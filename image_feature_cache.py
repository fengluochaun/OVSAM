"""Small disk cache for frozen SAM3 image features.

The training and evaluation entry points use this module to avoid encoding the
same image once per text prompt.  Features are stored as CPU tensors and moved
back to the caller's device when loaded.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch


def resolve_feature_cache_dir(explicit_dir: str | None, base_dir: str | None = None) -> str:
    """Resolve a cache directory relative to the checkpoint directory."""
    if explicit_dir:
        path = Path(explicit_dir)
        if not path.is_absolute() and base_dir:
            path = Path(base_dir) / path
    else:
        path = Path(base_dir or ".") / "image_feature_cache"
    return str(path.expanduser().resolve())


def _cache_path(cache_dir: str, image_path: str) -> str:
    normalized = os.path.abspath(os.path.normpath(str(image_path)))
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{digest}.pt")


def _load_feature_file(path: str, device: str | torch.device) -> List[torch.Tensor] | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        features = payload["features"] if isinstance(payload, dict) else payload
        if not isinstance(features, (list, tuple)) or len(features) == 0:
            return None
        return [feature.to(device=device, non_blocking=True) for feature in features]
    except (OSError, RuntimeError, KeyError, TypeError, ValueError, EOFError):
        return None


def _save_feature_file(path: str, features: Sequence[torch.Tensor]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    payload = {"features": [feature.detach().to(device="cpu").contiguous() for feature in features]}
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


@torch.no_grad()
def get_batched_cached_image_features(
    model: torch.nn.Module,
    image_tensors: Sequence[torch.Tensor],
    image_paths: Sequence[str],
    device: str | torch.device,
    cache_dir: str,
    enable_disk_cache: bool = True,
) -> Tuple[List[torch.Tensor], Dict[str, int]]:
    """Return one feature pyramid per input image, using a disk cache.

    The returned list is flattened by feature level, matching the detector's
    ``forward_decoder`` input: each tensor has shape ``[B, C, H, W]``.
    """
    if len(image_tensors) != len(image_paths):
        raise ValueError("image_tensors and image_paths must have the same length")
    if len(image_tensors) == 0:
        raise ValueError("image_tensors is empty")

    # The current callers pass one image at a time.  Keeping this API batched
    # also covers the validation path in train.py.
    n = len(image_tensors)
    cached_by_image: List[List[torch.Tensor] | None] = [None] * n
    missing: List[int] = []
    cache_hits = 0

    if enable_disk_cache:
        os.makedirs(cache_dir, exist_ok=True)
        for index, image_path in enumerate(image_paths):
            loaded = _load_feature_file(_cache_path(cache_dir, image_path), device)
            if loaded is None:
                missing.append(index)
            else:
                cached_by_image[index] = loaded
                cache_hits += 1
    else:
        missing = list(range(n))

    cache_misses = len(missing)
    if missing:
        images = torch.stack([image_tensors[index] for index in missing], dim=0).to(device)
        # The frozen SAM3 encoder is inference-only.  Keeping its feature
        # pyramid in fp16 substantially reduces cache size and GPU residency;
        # the decoder is already executed under autocast by the callers.
        use_cuda_amp = torch.is_tensor(images) and images.is_cuda
        with torch.autocast(device_type="cuda", enabled=use_cuda_amp, dtype=torch.float16):
            fresh_by_level = model.forward_image(images, return_image_aux=False)
        fresh_by_level = [feature.detach() for feature in fresh_by_level]

        for local_index, image_index in enumerate(missing):
            per_image = [feature[local_index : local_index + 1] for feature in fresh_by_level]
            cached_by_image[image_index] = [feature.to(device=device) for feature in per_image]
            if enable_disk_cache:
                _save_feature_file(
                    _cache_path(cache_dir, image_paths[image_index]),
                    per_image,
                )

    if any(features is None for features in cached_by_image):
        raise RuntimeError("Failed to produce image features for every input image")

    # The detector's single-image evaluation path expects [level][B,...], not
    # a nested per-image list.  Concatenate in the original input order.
    num_levels = len(cached_by_image[0])  # type: ignore[arg-type]
    merged: List[torch.Tensor] = []
    for level in range(num_levels):
        merged.append(torch.cat([features[level] for features in cached_by_image], dim=0))  # type: ignore[index]

    return merged, {"cache_hits": cache_hits, "cache_misses": cache_misses}


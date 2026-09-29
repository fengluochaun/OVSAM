import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class InstanceRecord:
    category_id: int
    phrase: str
    bbox_xyxy: List[float]
    area: float


@dataclass
class SampleRecord:
    case_id: str
    slice_z: int
    image_path: str
    height: int
    width: int
    instances: List[InstanceRecord]


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _normalize_path_str(path: str) -> str:
    """
    Convert mixed Windows/Linux separators into current OS separator.
    """
    return path.replace("\\", os.sep).replace("/", os.sep)


def _resolve_image_path(image_path: str, ann_path: str, image_root: Optional[str] = None) -> str:
    """
    Robustly resolve image path saved in annotation json.

    Priority:
    1) absolute path -> normalize and return
    2) image_root / image_path
    3) parent(ann_dir) / image_path   # useful when ann is under .../ann/
    4) ann_dir / image_path
    """
    image_path = _normalize_path_str(image_path)

    if os.path.isabs(image_path):
        return image_path

    candidates = []

    if image_root is not None:
        candidates.append(os.path.join(image_root, image_path))

    ann_dir = os.path.dirname(os.path.abspath(ann_path))
    ann_parent = os.path.dirname(ann_dir)

    candidates.append(os.path.join(ann_parent, image_path))
    candidates.append(os.path.join(ann_dir, image_path))
    candidates.append(image_path)

    for p in candidates:
        p = os.path.normpath(p)
        if os.path.exists(p):
            return p

    # fallback: return normalized first candidate for clearer error printing
    return os.path.normpath(candidates[0] if len(candidates) > 0 else image_path)


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return torch.from_numpy(arr)


class OVSAMDataset(Dataset):
    """
    A generic dataset for OVSAM.

    JSON format expectation (ODVG-style):
    {
        "organ_names": [...],
        "items": [
            {
                "case_id": "FLARE22_Tr_0001",
                "slice_z": 123,
                "image": "/abs/path/to/image.png",
                "height": 512,
                "width": 512,
                "instances": [
                    {
                        "category_id": 1,
                        "phrase": "liver",
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "area": 12345
                    },
                    ...
                ]
            },
            ...
        ]
    }

    Modes:
    - if prompt_sampler is None:
        returns a raw per-image sample
    - if prompt_sampler is provided:
        returns a processed training/eval task sample
    """

    def __init__(
        self,
        ann_path: str,
        image_root: Optional[str] = None,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        prompt_sampler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        return_raw_if_empty: bool = False,
        min_instances: int = 1,
    ):
        super().__init__()
        self.ann_path = ann_path
        self.image_root = image_root
        self.transform = transform if transform is not None else _pil_to_tensor
        self.prompt_sampler = prompt_sampler
        self.return_raw_if_empty = return_raw_if_empty
        self.min_instances = min_instances

        meta = _load_json(ann_path)
        self.organ_names: List[str] = meta.get("organ_names", [])
        raw_items: List[Dict[str, Any]] = meta.get("items", [])

        self.samples: List[SampleRecord] = []
        for item in raw_items:
            instances = [
                InstanceRecord(
                    category_id=int(obj["category_id"]),
                    phrase=str(obj["phrase"]),
                    bbox_xyxy=[float(x) for x in obj["bbox_xyxy"]],
                    area=float(obj.get("area", 0.0)),
                )
                for obj in item.get("instances", [])
            ]

            if len(instances) < self.min_instances:
                continue

            image_path = _resolve_image_path(
                image_path=item["image"],
                ann_path=self.ann_path,
                image_root=self.image_root,
            )

            self.samples.append(
                SampleRecord(
                    case_id=str(item.get("case_id", "")),
                    slice_z=int(item.get("slice_z", -1)),
                    image_path=image_path,
                    height=int(item["height"]),
                    width=int(item["width"]),
                    instances=instances,
                )
            )

        self.category_to_name = {
            idx: name for idx, name in enumerate(self.organ_names)
        }

    def __len__(self) -> int:
        return len(self.samples)

    def _read_image(self, image_path: str) -> Image.Image:
        img = Image.open(image_path).convert("RGB")
        return img

    def _to_raw_dict(self, record: SampleRecord, image_tensor: torch.Tensor) -> Dict[str, Any]:
        return {
            "case_id": record.case_id,
            "slice_z": record.slice_z,
            "image_path": record.image_path,
            "height": record.height,
            "width": record.width,
            "image": image_tensor,
            "instances": [
                {
                    "category_id": inst.category_id,
                    "phrase": inst.phrase,
                    "bbox_xyxy": inst.bbox_xyxy,
                    "area": inst.area,
                }
                for inst in record.instances
            ],
            "organ_names": self.organ_names,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.samples[idx]
        img = self._read_image(record.image_path)
        image_tensor = self.transform(img)
        raw_sample = self._to_raw_dict(record, image_tensor)

        if self.prompt_sampler is None:
            return raw_sample

        sampled = self.prompt_sampler(raw_sample)
        if sampled is None:
            if self.return_raw_if_empty:
                return raw_sample
            # Fallback: return raw and let upper layer skip
            return raw_sample
        return sampled


def ovsam_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Keep the batch flexible because:
    - number of targets may differ
    - prompt strings have variable length
    - some fields are list-based (merge prompts, negatives, etc.)

    The trainer/model can decide how to tensorize downstream.
    """
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return {}

    out: Dict[str, Any] = {}

    # stack images if possible
    if all(("image" in x and torch.is_tensor(x["image"])) for x in batch):
        shapes = [tuple(x["image"].shape) for x in batch]
        if len(set(shapes)) == 1:
            out["images"] = torch.stack([x["image"] for x in batch], dim=0)
        else:
            out["images"] = [x["image"] for x in batch]

    # keep everything else as list
    keys = set()
    for x in batch:
        keys.update(x.keys())

    for k in keys:
        if k == "image":
            continue
        out[k] = [x.get(k, None) for x in batch]

    return out

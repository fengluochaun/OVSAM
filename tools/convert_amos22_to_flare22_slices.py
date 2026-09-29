import argparse
import os
import re
import shutil
import struct
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


DEFAULT_SRC_ROOT = (
    "/data3/users/zhaojun/.cache/kagglehub/datasets/"
    "jvora25/amos-22/versions/1/amos22"
)
DEFAULT_OUT_ROOT = "/data3/users/zhaojun/project/OVSAM/data/amos22_flare22_style"

# AMOS22 label ids are kept unchanged in the exported 2D masks.
AMOS22_ORGAN_NAMES = [
    "background",
    "spleen",
    "right kidney",
    "left kidney",
    "gallbladder",
    "esophagus",
    "liver",
    "stomach",
    "aorta",
    "inferior vena cava",
    "pancreas",
    "right adrenal gland",
    "left adrenal gland",
    "duodenum",
    "bladder",
    "prostate or uterus",
]


def ensure_parent(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def reset_dir(path: str, overwrite: bool) -> None:
    if os.path.isdir(path):
        if not overwrite and os.listdir(path):
            raise FileExistsError(
                f"Output dir is not empty: {path}. "
                "Use --overwrite to replace generated images/masks."
            )
        if overwrite:
            shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def maybe_remove_file(path: str, overwrite: bool) -> None:
    if not os.path.exists(path):
        return
    if not overwrite:
        raise FileExistsError(
            f"Output file already exists: {path}. Use --overwrite to replace it."
        )
    os.remove(path)


def parse_amos_case_id(filename: str) -> Optional[int]:
    match = re.fullmatch(r"amos_(\d{4})\.nii", filename)
    if match is None:
        return None
    return int(match.group(1))


def list_case_ids(images_dir: str, labels_dir: str, ct_only: bool) -> List[int]:
    image_ids = {
        case_id
        for name in os.listdir(images_dir)
        for case_id in [parse_amos_case_id(name)]
        if case_id is not None
    }
    label_ids = {
        case_id
        for name in os.listdir(labels_dir)
        for case_id in [parse_amos_case_id(name)]
        if case_id is not None
    }
    case_ids = sorted(image_ids & label_ids)
    if ct_only:
        case_ids = [case_id for case_id in case_ids if case_id <= 500]
    return case_ids


def read_nifti(path: str) -> np.ndarray:
    if path.endswith(".nii.gz"):
        raise ValueError(f"Only uncompressed .nii is supported in this script: {path}")

    with open(path, "rb") as f:
        header = f.read(348)

    sizeof_hdr = struct.unpack("<I", header[:4])[0]
    endian = "<" if sizeof_hdr == 348 else ">"
    if endian == ">" and struct.unpack(">I", header[:4])[0] != 348:
        raise ValueError(f"Invalid NIfTI header: {path}")

    dims = struct.unpack(endian + "8h", header[40:56])
    ndim = int(dims[0])
    shape = tuple(int(x) for x in dims[1 : 1 + ndim])
    datatype = int(struct.unpack(endian + "h", header[70:72])[0])
    vox_offset = int(struct.unpack(endian + "f", header[108:112])[0])
    scl_slope = float(struct.unpack(endian + "f", header[112:116])[0])
    scl_inter = float(struct.unpack(endian + "f", header[116:120])[0])

    dtype_map = {
        2: np.uint8,
        4: np.int16,
        8: np.int32,
        16: np.float32,
        64: np.float64,
        256: np.int8,
        512: np.uint16,
        768: np.uint32,
    }
    if datatype not in dtype_map:
        raise ValueError(f"Unsupported NIfTI datatype {datatype} in {path}")

    dtype = np.dtype(dtype_map[datatype]).newbyteorder(endian)
    arr = np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=vox_offset,
        shape=shape,
        order="F",
    )
    arr = np.asarray(arr)

    if scl_slope not in (0.0, 1.0):
        arr = arr.astype(np.float32) * scl_slope
    if scl_inter != 0.0:
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        arr = arr + scl_inter

    return arr


def count_present_organs(mask_slice: np.ndarray) -> int:
    labels = np.unique(mask_slice)
    return int(np.sum(labels > 0))


def find_keep_range(mask_volume: np.ndarray, min_organs: int) -> Optional[Tuple[int, int]]:
    qualifying = []
    for z in range(mask_volume.shape[2]):
        if count_present_organs(mask_volume[:, :, z]) >= min_organs:
            qualifying.append(z)
    if not qualifying:
        return None
    return qualifying[0], qualifying[-1]


def window_ct_to_uint8(ct_slice: np.ndarray, low: float, high: float) -> np.ndarray:
    clipped = np.clip(ct_slice.astype(np.float32), low, high)
    clipped = (clipped - low) / max(high - low, 1e-6)
    clipped = np.clip(clipped * 255.0, 0.0, 255.0)
    return clipped.astype(np.uint8)


def to_pil_gray(slice_2d: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(slice_2d), mode="L")


def resize_image_pair(
    image: Image.Image,
    mask: Image.Image,
    output_size: Optional[int],
) -> Tuple[Image.Image, Image.Image]:
    if output_size is None:
        return image, mask
    size = (output_size, output_size)
    if image.size == size and mask.size == size:
        return image, mask
    return (
        image.resize(size, resample=Image.BILINEAR),
        mask.resize(size, resample=Image.NEAREST),
    )


def write_organ_names(out_root: str, organ_names: Sequence[str], overwrite: bool) -> str:
    path = os.path.join(out_root, "organ_names.txt")
    maybe_remove_file(path, overwrite)
    with open(path, "w", encoding="utf-8") as f:
        for name in organ_names:
            f.write(name + "\n")
    return path


def iter_split_specs(src_root: str) -> Iterable[Tuple[str, str, str, str]]:
    yield ("imagesTr", "labelsTr", "train", "AMOS22_Tr")
    yield ("imagesVa", "labelsVa", "test", "AMOS22_Va")


def export_split(
    src_root: str,
    out_root: str,
    image_subdir: str,
    label_subdir: str,
    out_split: str,
    case_prefix: str,
    min_organs: int,
    output_size: Optional[int],
    window_low: float,
    window_high: float,
    ct_only: bool,
    limit_cases: Optional[int],
    transpose_xy: bool,
    flip_vertical: bool,
) -> Dict[str, int]:
    images_dir = os.path.join(src_root, image_subdir)
    labels_dir = os.path.join(src_root, label_subdir)
    out_images_dir = os.path.join(out_root, "images", out_split)
    out_masks_dir = os.path.join(out_root, "masks", out_split)

    case_ids = list_case_ids(images_dir, labels_dir, ct_only=ct_only)
    if limit_cases is not None:
        case_ids = case_ids[:limit_cases]

    stats = {
        "cases_total": len(case_ids),
        "cases_written": 0,
        "cases_skipped": 0,
        "slices_written": 0,
    }

    for case_id in case_ids:
        image_path = os.path.join(images_dir, f"amos_{case_id:04d}.nii")
        mask_path = os.path.join(labels_dir, f"amos_{case_id:04d}.nii")

        image_volume = read_nifti(image_path)
        mask_volume = read_nifti(mask_path).astype(np.uint8)

        if image_volume.shape[:3] != mask_volume.shape[:3]:
            raise ValueError(
                f"Image/mask shape mismatch for case {case_id:04d}: "
                f"{image_volume.shape} vs {mask_volume.shape}"
            )

        keep_range = find_keep_range(mask_volume, min_organs=min_organs)
        if keep_range is None:
            stats["cases_skipped"] += 1
            print(
                f"[WARN] skip case amos_{case_id:04d}: "
                f"no slice has >= {min_organs} organs"
            )
            continue

        z_start, z_end = keep_range
        case_name = f"{case_prefix}_{case_id:04d}"

        for z in range(z_start, z_end + 1):
            image_slice = image_volume[:, :, z]
            mask_slice = mask_volume[:, :, z]
            if transpose_xy:
                # Optional orientation used by the original generic converter.
                image_slice = image_slice.T
                mask_slice = mask_slice.T
            if flip_vertical:
                # Match the in-plane orientation of the AMOS22 Table IV export.
                # The source first spatial axis is written as image rows.
                image_slice = np.flip(image_slice, axis=0)
                mask_slice = np.flip(mask_slice, axis=0)

            image_u8 = window_ct_to_uint8(image_slice, low=window_low, high=window_high)
            image_pil = to_pil_gray(image_u8).convert("RGB")
            mask_pil = to_pil_gray(mask_slice)
            image_pil, mask_pil = resize_image_pair(image_pil, mask_pil, output_size)

            filename = f"{case_name}_z{z:04d}.png"
            image_pil.save(os.path.join(out_images_dir, filename))
            mask_pil.save(os.path.join(out_masks_dir, filename))
            stats["slices_written"] += 1

        stats["cases_written"] += 1
        print(
            f"[{out_split}] {case_name}: keep z[{z_start}, {z_end}] "
            f"-> {z_end - z_start + 1} slices"
        )

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert AMOS22 CT volumes into an OVSAM/FLARE22-style 2D png dataset. "
            "Only images/, masks/ and organ_names.txt are generated."
        )
    )
    parser.add_argument("--src_root", default=DEFAULT_SRC_ROOT)
    parser.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--min_organs", type=int, default=5)
    parser.add_argument("--window_low", type=float, default=-160.0)
    parser.add_argument("--window_high", type=float, default=240.0)
    parser.add_argument(
        "--output_size",
        type=int,
        default=512,
        help="Square output size. Use 0 to keep original in-plane size.",
    )
    parser.add_argument(
        "--include_non_ct",
        action="store_false",
        dest="ct_only",
        help="Also export ids > 500 if present in the source root.",
    )
    parser.set_defaults(ct_only=True)
    parser.add_argument("--limit_cases", type=int, default=0)
    parser.add_argument(
        "--no_transpose",
        action="store_false",
        dest="transpose_xy",
        help="Keep the source in-plane orientation before optional flipping.",
    )
    parser.set_defaults(transpose_xy=True)
    parser.add_argument(
        "--flip_vertical",
        action="store_true",
        help="Flip each exported slice along its first spatial axis.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace generated images/, masks/ and organ_names.txt under out_root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_size = args.output_size if args.output_size > 0 else None
    limit_cases = args.limit_cases if args.limit_cases > 0 else None

    ensure_parent(args.out_root)
    reset_dir(os.path.join(args.out_root, "images", "train"), overwrite=args.overwrite)
    reset_dir(os.path.join(args.out_root, "images", "test"), overwrite=args.overwrite)
    reset_dir(os.path.join(args.out_root, "masks", "train"), overwrite=args.overwrite)
    reset_dir(os.path.join(args.out_root, "masks", "test"), overwrite=args.overwrite)
    organ_names_path = write_organ_names(
        out_root=args.out_root,
        organ_names=AMOS22_ORGAN_NAMES,
        overwrite=args.overwrite,
    )

    split_stats: Dict[str, Dict[str, int]] = {}
    for image_subdir, label_subdir, out_split, case_prefix in iter_split_specs(args.src_root):
        split_stats[out_split] = export_split(
            src_root=args.src_root,
            out_root=args.out_root,
            image_subdir=image_subdir,
            label_subdir=label_subdir,
            out_split=out_split,
            case_prefix=case_prefix,
            min_organs=args.min_organs,
            output_size=output_size,
            window_low=args.window_low,
            window_high=args.window_high,
            ct_only=args.ct_only,
            limit_cases=limit_cases,
            transpose_xy=args.transpose_xy,
            flip_vertical=args.flip_vertical,
        )

    print("\n[Done]")
    print(f"organ_names: {organ_names_path}")
    for split, stats in split_stats.items():
        print(
            f"{split}: cases_total={stats['cases_total']} "
            f"cases_written={stats['cases_written']} "
            f"cases_skipped={stats['cases_skipped']} "
            f"slices_written={stats['slices_written']}"
        )


if __name__ == "__main__":
    main()

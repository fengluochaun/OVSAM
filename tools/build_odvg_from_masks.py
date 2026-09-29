import os
import re
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    raise ImportError("Please install opencv-python first: pip install opencv-python")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_organ_names(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"organ_names.txt not found: {path}")
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                names.append(t)
    return names


def list_pngs(folder: str) -> List[str]:
    if not os.path.exists(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    files.sort()
    return files


def parse_case_and_slice(filename: str) -> Tuple[str, int]:
    """
    Expect names like:
      FLARE22_Tr_0034_z0058.png
    """
    stem = os.path.splitext(filename)[0]
    m = re.match(r"(.+)_z(\d+)$", stem)
    if m is None:
        return stem, -1
    case_id = m.group(1)
    slice_z = int(m.group(2))
    return case_id, slice_z


def largest_cc_bbox(mask01: np.ndarray) -> Tuple[Optional[List[int]], int]:
    """
    mask01: HxW uint8 {0,1}
    return bbox [x1,y1,x2,y2] and area of largest cc
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask01.astype(np.uint8), connectivity=8)
    if num <= 1:
        return None, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    x = int(stats[idx, cv2.CC_STAT_LEFT])
    y = int(stats[idx, cv2.CC_STAT_TOP])
    w = int(stats[idx, cv2.CC_STAT_WIDTH])
    h = int(stats[idx, cv2.CC_STAT_HEIGHT])
    area = int(stats[idx, cv2.CC_STAT_AREA])
    return [x, y, x + w - 1, y + h - 1], area


def all_fg_bbox(mask01: np.ndarray) -> Tuple[Optional[List[int]], int]:
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None, 0
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    area = int(mask01.sum())
    return [x1, y1, x2, y2], area


def build_odvg_for_split(
    root: str,
    split: str,
    organ_names: List[str],
    min_area: int = 200,
    bbox_mode: str = "largest_cc",
    save_absolute_path: bool = False,
) -> Dict:
    images_dir = os.path.join(root, "images", split)
    masks_dir = os.path.join(root, "masks", split)

    image_files = list_pngs(images_dir)
    if len(image_files) == 0:
        raise RuntimeError(f"No png images found in {images_dir}")

    items = []
    missing_masks = 0
    empty_items = 0

    for fn in image_files:
        img_path = os.path.join(images_dir, fn)
        mask_path = os.path.join(masks_dir, fn)

        if not os.path.exists(mask_path):
            missing_masks += 1
            continue

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if img.size != mask.size:
            mask = mask.resize(img.size, resample=Image.NEAREST)

        img_np = np.array(img)
        mask_np = np.array(mask).astype(np.uint8)

        H, W = mask_np.shape
        case_id, slice_z = parse_case_and_slice(fn)

        instances = []
        present_ids = [int(x) for x in np.unique(mask_np) if int(x) != 0]

        for cid in present_ids:
            if cid >= len(organ_names):
                continue

            binm = (mask_np == cid).astype(np.uint8)

            if bbox_mode == "largest_cc":
                bbox, area = largest_cc_bbox(binm)
            elif bbox_mode == "all_fg":
                bbox, area = all_fg_bbox(binm)
            else:
                raise ValueError(f"Unknown bbox_mode: {bbox_mode}")

            if bbox is None or area < min_area:
                continue

            instances.append({
                "category_id": cid,
                "phrase": organ_names[cid],
                "bbox_xyxy": bbox,
                "area": int(area)
            })

        if len(instances) == 0:
            empty_items += 1
            continue

        if save_absolute_path:
            image_field = os.path.abspath(img_path)
        else:
            # always save POSIX-style relative path
            image_field = f"images/{split}/{fn}"

        items.append({
            "case_id": case_id,
            "slice_z": slice_z,
            "image": image_field,
            "height": H,
            "width": W,
            "instances": instances
        })

    print(f"[{split}] total images      : {len(image_files)}")
    print(f"[{split}] missing masks     : {missing_masks}")
    print(f"[{split}] empty items skip  : {empty_items}")
    print(f"[{split}] valid items       : {len(items)}")

    return {
        "organ_names": organ_names,
        "items": items
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data3/users/zhaojun/project/OVSAM/data/cardiacUDC_A4C", help="processed dataset root")
    ap.add_argument("--organ_names", default="", help="path to organ_names.txt; default=root/organ_names.txt")
    ap.add_argument("--out_dir", default="", help="default=root/ann")
    ap.add_argument("--min_area", type=int, default=200)
    ap.add_argument("--bbox_mode", choices=["largest_cc", "all_fg"], default="largest_cc")
    ap.add_argument("--save_absolute_path", action="store_true")
    args = ap.parse_args()

    root = args.root
    organ_names_path = args.organ_names if args.organ_names else os.path.join(root, "organ_names.txt")
    out_dir = args.out_dir if args.out_dir else os.path.join(root, "ann")

    ensure_dir(out_dir)
    organ_names = load_organ_names(organ_names_path)

    odvg_train = build_odvg_for_split(
        root=root,
        split="train",
        organ_names=organ_names,
        min_area=args.min_area,
        bbox_mode=args.bbox_mode,
        save_absolute_path=args.save_absolute_path,
    )

    odvg_test = build_odvg_for_split(
        root=root,
        split="test",
        organ_names=organ_names,
        min_area=args.min_area,
        bbox_mode=args.bbox_mode,
        save_absolute_path=args.save_absolute_path,
    )

    train_out = os.path.join(out_dir, "odvg_train.json")
    test_out = os.path.join(out_dir, "odvg_test.json")

    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(odvg_train, f, ensure_ascii=False, indent=2)

    with open(test_out, "w", encoding="utf-8") as f:
        json.dump(odvg_test, f, ensure_ascii=False, indent=2)

    print("\n[Done]")
    print("Saved:")
    print(" -", train_out)
    print(" -", test_out)


if __name__ == "__main__":
    main()

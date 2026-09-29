import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from PIL import Image


ROOT_DEFAULT = "/data3/users/zhaojun/project/OVSAM/data/cardiacUDC_A4C"


if hasattr(Image, "Transpose"):
    ROTATE_90_CCW = Image.Transpose.ROTATE_90
else:
    ROTATE_90_CCW = Image.ROTATE_90


def rotate_bbox_left90(box_xyxy: List[int], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    return [
        y1,
        width - 1 - x2,
        y2,
        width - 1 - x1,
    ]


def rotate_pngs_in_dir(folder: Path) -> int:
    paths = sorted(folder.glob("*.png"))
    for path in paths:
        image = Image.open(path)
        rotated = image.transpose(ROTATE_90_CCW)
        rotated.save(path)
    return len(paths)


def rotate_odvg_file(path: Path) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    item_count = 0
    instance_count = 0
    for item in data.get("items", []):
        item_count += 1
        old_h = int(item["height"])
        old_w = int(item["width"])
        item["height"] = old_w
        item["width"] = old_h

        for inst in item.get("instances", []):
            inst["bbox_xyxy"] = rotate_bbox_left90(inst["bbox_xyxy"], width=old_w, height=old_h)
            instance_count += 1

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "items": item_count,
        "instances": instance_count,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    image_train = root / "images" / "train"
    image_test = root / "images" / "test"
    mask_train = root / "masks" / "train"
    mask_test = root / "masks" / "test"
    ann_train = root / "ann" / "odvg_train.json"
    ann_test = root / "ann" / "odvg_test.json"

    required = [image_train, image_test, mask_train, mask_test, ann_train, ann_test]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    num_image_train = rotate_pngs_in_dir(image_train)
    num_image_test = rotate_pngs_in_dir(image_test)
    num_mask_train = rotate_pngs_in_dir(mask_train)
    num_mask_test = rotate_pngs_in_dir(mask_test)
    train_stats = rotate_odvg_file(ann_train)
    test_stats = rotate_odvg_file(ann_test)

    print(f"[images/train] rotated {num_image_train}")
    print(f"[images/test]  rotated {num_image_test}")
    print(f"[masks/train]  rotated {num_mask_train}")
    print(f"[masks/test]   rotated {num_mask_test}")
    print(
        f"[odvg_train] items={train_stats['items']} instances={train_stats['instances']}"
    )
    print(
        f"[odvg_test]  items={test_stats['items']} instances={test_stats['instances']}"
    )


if __name__ == "__main__":
    main()

import argparse
import json
import os
from typing import Dict, List


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rewrite_image_paths(coco: Dict, image_dir: str) -> Dict:
    image_dir = os.path.abspath(image_dir)
    images: List[Dict] = []

    for image in coco.get("images", []):
        image_copy = dict(image)
        file_name = image_copy.get("file_name", "")
        if not file_name:
            raise ValueError(f"Image entry missing file_name: {image_copy}")

        if os.path.isabs(file_name):
            abs_path = file_name
        else:
            abs_path = os.path.abspath(os.path.join(image_dir, os.path.basename(file_name)))

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Image file not found for COCO entry: {abs_path}")

        image_copy["file_name"] = abs_path
        images.append(image_copy)

    coco_copy = dict(coco)
    coco_copy["images"] = images
    return coco_copy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a MedSAM3 zero-shot evaluation directory from an existing COCO annotation file."
    )
    parser.add_argument(
        "--coco",
        required=True,
        help="Path to the source COCO annotation json, e.g. data/flare22/labels/test_coco.json",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing the referenced images, e.g. data/flare22/images/test",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory to place _annotations.coco.json for MedSAM3 validation",
    )
    args = parser.parse_args()

    with open(args.coco, "r", encoding="utf-8") as f:
        coco = json.load(f)

    prepared = rewrite_image_paths(coco, args.image_dir)

    ensure_dir(args.out_dir)
    out_path = os.path.join(args.out_dir, "_annotations.coco.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prepared, f, ensure_ascii=False)

    print(f"Saved MedSAM3 evaluation annotation to: {out_path}")
    print(f"Images: {len(prepared.get('images', []))}")
    print(f"Annotations: {len(prepared.get('annotations', []))}")
    print(f"Categories: {len(prepared.get('categories', []))}")


if __name__ == "__main__":
    main()

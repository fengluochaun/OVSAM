from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', type=str, default='/data3/users/zhaojun/project/OVSAM/data/flare22')
    ap.add_argument('--source_split', type=str, default='test')
    ap.add_argument('--target_split', type=str, default='merge_test')
    ap.add_argument('--organ_ids', type=str, default='2,13')
    ap.add_argument('--organ_names', type=str, default='right kidney,left kidney')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root)
    images_src = root / 'images' / args.source_split
    masks_src = root / 'masks' / args.source_split
    images_dst = root / 'images' / args.target_split
    masks_dst = root / 'masks' / args.target_split
    images_dst.mkdir(parents=True, exist_ok=True)
    masks_dst.mkdir(parents=True, exist_ok=True)

    organ_ids = [int(x.strip()) for x in args.organ_ids.split(',') if x.strip()]
    organ_names = [x.strip() for x in args.organ_names.split(',') if x.strip()]
    if len(organ_ids) != len(organ_names):
        raise ValueError('organ_ids and organ_names must have the same length')

    for folder in (images_dst, masks_dst):
        for p in folder.glob('*.png'):
            p.unlink()

    selected = []
    for mask_path in sorted(masks_src.glob('*.png')):
        mask = np.array(Image.open(mask_path))
        if not all(np.any(mask == organ_id) for organ_id in organ_ids):
            continue
        image_path = images_src / mask_path.name
        if not image_path.exists():
            raise FileNotFoundError(f'missing image for {mask_path.name}')
        merged = np.isin(mask, organ_ids).astype(np.uint8)
        shutil.copy2(image_path, images_dst / image_path.name)
        Image.fromarray(merged).save(masks_dst / mask_path.name)
        selected.append(mask_path.name)

    summary = {
        'source_split': args.source_split,
        'target_split': args.target_split,
        'organ_ids': dict(zip(organ_names, organ_ids)),
        'num_slices': len(selected),
        'filenames': selected,
    }
    summary_path = root / f'{args.target_split}_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f'saved {len(selected)} slices to {images_dst} and {masks_dst}')
    print(f'summary: {summary_path}')


if __name__ == '__main__':
    main()

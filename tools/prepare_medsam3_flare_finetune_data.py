import argparse
import json
import os
import random
from typing import Any, Dict, List, Set, Tuple


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_case_id(file_name: str) -> str:
    stem = os.path.splitext(os.path.basename(file_name))[0]
    if '_z' not in stem:
        return stem
    return stem.rsplit('_z', 1)[0]


def load_coco(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def attach_absolute_paths(coco: Dict[str, Any], image_dir: str) -> Dict[str, Any]:
    image_dir = os.path.abspath(image_dir)
    images = []
    for image in coco['images']:
        image_copy = dict(image)
        file_name = image_copy['file_name']
        abs_path = file_name if os.path.isabs(file_name) else os.path.join(image_dir, os.path.basename(file_name))
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f'image not found: {abs_path}')
        image_copy['file_name'] = os.path.abspath(abs_path)
        image_copy['case_id'] = parse_case_id(file_name)
        images.append(image_copy)

    coco_copy = dict(coco)
    coco_copy['images'] = images
    return coco_copy


def split_cases(case_ids: List[str], valid_ratio: float, seed: int) -> Tuple[Set[str], Set[str]]:
    case_ids = sorted(case_ids)
    rng = random.Random(seed)
    rng.shuffle(case_ids)
    num_valid = max(1, int(round(len(case_ids) * valid_ratio)))
    valid_cases = set(sorted(case_ids[:num_valid]))
    train_cases = set(sorted(case_ids[num_valid:]))
    return train_cases, valid_cases


def filter_coco_by_cases(coco: Dict[str, Any], keep_cases: Set[str]) -> Dict[str, Any]:
    images = [img for img in coco['images'] if img['case_id'] in keep_cases]
    keep_image_ids = {int(img['id']) for img in images}
    annotations = [ann for ann in coco['annotations'] if int(ann['image_id']) in keep_image_ids]

    return {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'categories': coco['categories'],
        'images': images,
        'annotations': annotations,
    }


def write_split(coco: Dict[str, Any], out_dir: str) -> None:
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, '_annotations.coco.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False)


def summarize_split(name: str, coco: Dict[str, Any]) -> None:
    case_ids = sorted({img['case_id'] for img in coco['images']})
    print(f'[{name}] cases={len(case_ids)} images={len(coco["images"])} annotations={len(coco["annotations"])}')


def main() -> None:
    ap = argparse.ArgumentParser(description='Prepare FLARE22 data for official MedSAM3 LoRA fine-tuning.')
    ap.add_argument('--train_coco', default='/data3/users/zhaojun/project/OVSAM/data/flare22/annotations/instances_train.json')
    ap.add_argument('--test_coco', default='/data3/users/zhaojun/project/OVSAM/data/flare22/annotations/instances_test.json')
    ap.add_argument('--train_image_dir', default='/data3/users/zhaojun/project/OVSAM/data/flare22/images/train')
    ap.add_argument('--test_image_dir', default='/data3/users/zhaojun/project/OVSAM/data/flare22/images/test')
    ap.add_argument('--out_root', default='/data3/users/zhaojun/project/OVSAM/data/flare22/medsam3_finetune_train_valid')
    ap.add_argument('--eval_test_dir', default='/data3/users/zhaojun/project/OVSAM/data/flare22/medsam3_eval_test')
    ap.add_argument('--valid_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--valid_source', choices=['train_split', 'test'], default='train_split')
    args = ap.parse_args()

    train_coco = attach_absolute_paths(load_coco(args.train_coco), args.train_image_dir)
    test_coco = attach_absolute_paths(load_coco(args.test_coco), args.test_image_dir)

    all_train_cases = sorted({img['case_id'] for img in train_coco['images']})
    all_test_cases = sorted({img['case_id'] for img in test_coco['images']})

    if args.valid_source == 'train_split':
        train_cases, valid_cases = split_cases(all_train_cases, args.valid_ratio, args.seed)
        train_split = filter_coco_by_cases(train_coco, train_cases)
        valid_split = filter_coco_by_cases(train_coco, valid_cases)
    else:
        train_cases = set(all_train_cases)
        valid_cases = set(all_test_cases)
        train_split = train_coco
        valid_split = test_coco

    write_split(train_split, os.path.join(args.out_root, 'train'))
    write_split(valid_split, os.path.join(args.out_root, 'valid'))
    write_split(test_coco, args.eval_test_dir)

    split_meta = {
        'seed': args.seed,
        'valid_ratio': args.valid_ratio,
        'valid_source': args.valid_source,
        'train_cases': sorted(train_cases),
        'valid_cases': sorted(valid_cases),
        'num_train_cases': len(train_cases),
        'num_valid_cases': len(valid_cases),
    }
    if args.valid_source == 'test':
        split_meta['note'] = 'valid split is identical to the held-out test split'
    with open(os.path.join(args.out_root, 'split_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(split_meta, f, ensure_ascii=False, indent=2)

    summarize_split('train', train_split)
    summarize_split('valid', valid_split)
    summarize_split('test_eval', test_coco)
    print(f'[done] fine-tune data: {args.out_root}')
    print(f'[done] held-out test eval data: {args.eval_test_dir}')


if __name__ == '__main__':
    main()

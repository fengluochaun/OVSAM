import os
import sys
import json
import math
import time
import random
import logging
import argparse
from datetime import timedelta
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm import tqdm
# CUDA_VISIBLE_DEVICES=0,2,3,4 torchrun --standalone --nproc_per_node=4 train.py

# =========================================================
# Make project root importable
# =========================================================
def add_project_root_to_path():
    this_file = os.path.abspath(__file__)
    cur_dir = os.path.dirname(this_file)

    candidates = [
        cur_dir,
        os.path.dirname(cur_dir),
        os.path.dirname(os.path.dirname(cur_dir)),
        os.path.dirname(os.path.dirname(os.path.dirname(cur_dir))),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "datasets")) and os.path.exists(os.path.join(c, "models")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return c
    return None


PROJECT_ROOT = add_project_root_to_path()
if PROJECT_ROOT is None:
    print("[WARN] Could not confidently find project root. Make sure datasets/ and models/ are importable.")


from datasets.dataset import OVSAMDataset
from datasets.prompt_sampler import load_prompt_bank, PromptBankSampler
from image_feature_cache import get_batched_cached_image_features, resolve_feature_cache_dir
from models.detector.detector_builder import build_detector
from losses.detection_loss import build_detection_criterion
from models.detector.matcher import box_cxcywh_to_xyxy


# =========================================================
# Distributed helpers
# =========================================================
def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process():
    return get_rank() == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def setup_distributed(args):
    """
    torchrun 启动:
      CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py ...

    也兼容单卡 python 直接运行。
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(hours=2),
        )
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        torch.cuda.set_device(0 if torch.cuda.is_available() else "cpu")


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed: int, rank: int = 0):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 正式训练先别强制 deterministic，太容易拖速度
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =========================================================
# Logging / IO
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def setup_logger(save_dir: str):
    logger = logging.getLogger("train_ddp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    if is_main_process():
        fh = logging.FileHandler(os.path.join(save_dir, "train.log"), mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger


def log_jsonl(save_dir: str, record: Dict[str, Any]):
    if not is_main_process():
        return
    path = os.path.join(save_dir, "history.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =========================================================
# Dataset helpers
# =========================================================
def choose_prompt_from_positive_task(task: Dict[str, Any]) -> str:
    return task["primary_prompt"]


def build_target_boxes_tensor(target_boxes: List[List[float]]) -> torch.Tensor:
    if len(target_boxes) == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.as_tensor(target_boxes, dtype=torch.float32).reshape(-1, 4)


def stack_padded_images(image_tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Stack variable-sized CHW tensors by zero-padding to the max H/W in the batch.

    Returns:
        batched_images: [B,C,Hmax,Wmax]
        padded_hw:      (Hmax, Wmax)
    """
    if len(image_tensors) == 0:
        raise ValueError("image_tensors is empty.")

    shapes = [tuple(x.shape) for x in image_tensors]
    if len(set(shapes)) == 1:
        h, w = int(image_tensors[0].shape[-2]), int(image_tensors[0].shape[-1])
        return torch.stack(image_tensors, dim=0), (h, w)

    max_h = max(int(x.shape[-2]) for x in image_tensors)
    max_w = max(int(x.shape[-1]) for x in image_tensors)
    c = int(image_tensors[0].shape[0])
    dtype = image_tensors[0].dtype

    out = torch.zeros((len(image_tensors), c, max_h, max_w), dtype=dtype)
    for i, x in enumerate(image_tensors):
        h, w = int(x.shape[-2]), int(x.shape[-1])
        out[i, :, :h, :w] = x
    return out, (max_h, max_w)


def expand_train_prompt_tasks(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    positive_tasks = sample.get("positive_tasks", None)
    if positive_tasks is None:
        positive_tasks = [{
            "primary_prompt": sample.get("primary_prompt", ""),
            "aux_prompt": sample.get("aux_prompt", None),
            "target_boxes": sample.get("target_boxes", []),
            "task_type": sample.get("task_type", ""),
            "canonical_name": sample.get("canonical_name", ""),
        }]

    tasks = []
    for pos_task in positive_tasks:
        tasks.append({
            "prompt": choose_prompt_from_positive_task(pos_task),
            "target_boxes": pos_task.get("target_boxes", []),
            "task_type": pos_task.get("task_type", ""),
            "canonical_name": pos_task.get("canonical_name", ""),
            "is_negative": False,
        })

    for neg in sample.get("negative_prompts", []):
        if isinstance(neg, dict):
            neg_prompt = neg.get("prompt", "")
            neg_canonical_name = neg.get("canonical_name", "")
        else:
            neg_prompt = str(neg)
            neg_canonical_name = ""

        if neg_prompt == "":
            continue

        tasks.append({
            "prompt": neg_prompt,
            "target_boxes": [],
            "task_type": "negative",
            "canonical_name": neg_canonical_name,
            "is_negative": True,
        })

    return tasks


def choose_prompt_from_eval_sample(sample: Dict[str, Any], prompt_type: str = "canonical", index: int = 0) -> str:
    if prompt_type == "canonical":
        queries = sample.get("canonical_queries", [])
    elif prompt_type == "heldout":
        queries = sample.get("heldout_queries", [])
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")

    if len(queries) == 0:
        raise RuntimeError(f"No queries found for prompt_type={prompt_type}")

    index = min(index, len(queries) - 1)
    return queries[index]["prompt"]


def build_train_dataset(args):
    prompt_bank = load_prompt_bank(args.prompt_bank)
    sampler = PromptBankSampler(
        prompt_bank=prompt_bank,
        mode="train",
        positive_sampling=args.train_positive_sampling,
        train_use_template=bool(args.train_use_template),
        p_negative=args.p_negative,
        max_negatives=args.max_negatives,
        seed=args.seed,
    )

    if args.train_positive_sampling == "single":
        train_batch_unit = "image"
    elif args.train_positive_sampling == "all":
        train_batch_unit = "task"
    else:
        raise ValueError(f"Unknown train_positive_sampling: {args.train_positive_sampling}")

    base_dataset = OVSAMDataset(
        ann_path=args.train_ann_path,
        image_root=args.image_root,
        prompt_sampler=(sampler if train_batch_unit == "image" else None),
        return_raw_if_empty=True,
    )

    if train_batch_unit == "image":
        return base_dataset

    if train_batch_unit == "task":
        return TrainTaskDataset(
            base_dataset=base_dataset,
            prompt_sampler=sampler,
            positive_sampling=args.train_positive_sampling,
        )

    raise ValueError(f"Unknown train_batch_unit: {train_batch_unit}")


def build_val_dataset(args):
    prompt_bank = load_prompt_bank(args.prompt_bank)
    sampler = PromptBankSampler(
        prompt_bank=prompt_bank,
        mode="eval",
        eval_use_template=bool(getattr(args, "eval_use_template", 0)),
        p_negative=args.p_negative,
        max_negatives=args.max_negatives,
        seed=args.seed,
    )

    dataset = OVSAMDataset(
        ann_path=args.val_ann_path,
        image_root=args.image_root,
        prompt_sampler=sampler,
        return_raw_if_empty=True,
    )
    return dataset

# =========================================================
# Validation task dataset (expand one image to multiple organ tasks)
# =========================================================
def _pick_one_query_per_canonical(queries: List[Dict[str, Any]], prompt_index: int = 0) -> List[Dict[str, Any]]:
    """
    For each canonical organ, only keep one prompt description.
    If there are multiple descriptions for the same canonical organ,
    choose the prompt_index-th one (clamped).
    """
    grouped = defaultdict(list)
    for q in queries:
        cname = q.get("canonical_name", "unknown")
        grouped[cname].append(q)

    picked = []
    for cname, qs in grouped.items():
        idx = min(prompt_index, len(qs) - 1)
        picked.append(qs[idx])
    return picked


class ValTaskDataset(Dataset):
    """
    Expand base eval dataset:
      one image -> multiple organ-level eval tasks
    but only keep one prompt description per organ.
    """
    def __init__(
        self,
        base_dataset: OVSAMDataset,
        eval_prompt_type: str = "canonical",
        eval_prompt_index: int = 0,
    ):
        self.base_dataset = base_dataset
        self.eval_prompt_type = eval_prompt_type
        self.eval_prompt_index = eval_prompt_index
        self.tasks: List[Dict[str, Any]] = []

        for sample_idx in range(len(base_dataset)):
            sample = base_dataset[sample_idx]

            if eval_prompt_type == "canonical":
                queries = sample.get("canonical_queries", [])
            elif eval_prompt_type == "heldout":
                queries = sample.get("heldout_queries", [])
            else:
                raise ValueError(f"Unknown eval_prompt_type: {eval_prompt_type}")

            queries = _pick_one_query_per_canonical(
                queries=queries,
                prompt_index=eval_prompt_index,
            )

            for q in queries:
                self.tasks.append({
                    "sample_idx": sample_idx,
                    "query": q,
                })

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        sample = self.base_dataset[task["sample_idx"]]
        q = task["query"]

        return {
            "image": sample["image"],
            "image_path": sample.get("image_path", ""),
            "prompt": q["prompt"],
            "target_boxes": q["target_boxes"],   # list of xyxy abs boxes
            "image_size": (int(sample["height"]), int(sample["width"])),
            "case_id": sample.get("case_id", ""),
            "slice_z": sample.get("slice_z", -1),
            "canonical_name": q.get("canonical_name", sample.get("canonical_name", "unknown")),
            "eval_prompt_type": self.eval_prompt_type,
        }


class TrainTaskDataset(Dataset):
    """
    Expand base train dataset to task-level samples.

    - positive_sampling=single:
        one task slot per image, and the present organ is still sampled randomly
    - positive_sampling=all:
        one task slot per (image, canonical organ) pair

    Negative prompt sampling remains unchanged, but is attached per task sample
    instead of being expanded from a whole-image batch. This keeps the effective
    prompt count per batch much more stable.
    """
    def __init__(
        self,
        base_dataset: OVSAMDataset,
        prompt_sampler: PromptBankSampler,
        positive_sampling: str = "single",
    ):
        self.base_dataset = base_dataset
        self.prompt_sampler = prompt_sampler
        self.positive_sampling = positive_sampling
        self.task_specs: List[Dict[str, Any]] = []

        for sample_idx, record in enumerate(base_dataset.samples):
            if positive_sampling == "single":
                self.task_specs.append({
                    "sample_idx": sample_idx,
                    "canonical_name": None,
                })
                continue

            if positive_sampling != "all":
                raise ValueError(f"Unknown positive_sampling: {positive_sampling}")

            organ_names = sorted({inst.phrase for inst in record.instances})
            for canonical_name in organ_names:
                self.task_specs.append({
                    "sample_idx": sample_idx,
                    "canonical_name": canonical_name,
                })

    def __len__(self):
        return len(self.task_specs)

    def __getitem__(self, idx):
        task = self.task_specs[idx]
        sample = self.base_dataset[task["sample_idx"]]
        instances = sample["instances"]

        if self.positive_sampling == "single":
            chosen_inst = self.prompt_sampler.rng.choice(instances)
            canonical_name = chosen_inst["phrase"]
            matched_instances = [chosen_inst]
        else:
            canonical_name = task["canonical_name"]
            matched_instances = [x for x in instances if x["phrase"] == canonical_name]
            if len(matched_instances) == 0:
                raise RuntimeError(
                    f"No instances found for canonical_name={canonical_name} "
                    f"in sample_idx={task['sample_idx']}"
                )

        positive_task = self.prompt_sampler._build_positive_train_task(
            canonical_name=canonical_name,
            matched_instances=matched_instances,
            all_instances=instances,
        )

        negative_prompts = []
        present_names = [x["phrase"] for x in instances]
        if self.prompt_sampler.rng.random() < self.prompt_sampler.p_negative:
            negative_prompts = self.prompt_sampler._sample_negative_prompts(present_names)

        return {
            "image": sample["image"],
            "case_id": sample["case_id"],
            "slice_z": sample["slice_z"],
            "image_path": sample["image_path"],
            "height": sample["height"],
            "width": sample["width"],
            "instances": instances,
            "positive_tasks": [positive_task],
            "task_type": positive_task["task_type"],
            "canonical_name": positive_task["canonical_name"],
            "primary_prompt": positive_task["primary_prompt"],
            "aux_prompt": positive_task["aux_prompt"],
            "negative_prompts": negative_prompts,
            "target_boxes": positive_task["target_boxes"],
            "target_category_ids": positive_task["target_category_ids"],
            "target_instance_indices": positive_task["target_instance_indices"],
        }


def collate_as_list(batch):
    return batch


def build_dataloader(dataset, args, train: bool):
    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=train,
        drop_last=train,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=train,
        collate_fn=collate_as_list,
        persistent_workers=(args.num_workers > 0),
    )
    return loader, sampler


def prepare_batch(
    batch_samples: List[Dict[str, Any]],
    device: str,
    mode: str = "train",
    eval_prompt_type: str = "canonical",
    eval_prompt_index: int = 0,
):
    images = []
    prompts = []
    targets = []
    num_negative_tasks = 0

    for s in batch_samples:
        if mode == "train":
            prompt_tasks = expand_train_prompt_tasks(sample=s)
        else:
            prompt = choose_prompt_from_eval_sample(
                s,
                prompt_type=eval_prompt_type,
                index=eval_prompt_index,
            )

            if eval_prompt_type == "canonical":
                queries = s.get("canonical_queries", [])
            elif eval_prompt_type == "heldout":
                queries = s.get("heldout_queries", [])
            else:
                raise ValueError(f"Unknown eval_prompt_type: {eval_prompt_type}")

            if len(queries) == 0:
                raise RuntimeError("No eval queries found in sample.")
            q = queries[min(eval_prompt_index, len(queries) - 1)]
            prompt_tasks = [{
                "prompt": prompt,
                "target_boxes": q["target_boxes"],
                "task_type": s.get("task_type", ""),
                "canonical_name": s.get("canonical_name", ""),
                "is_negative": False,
            }]

        for task in prompt_tasks:
            images.append(s["image"])
            prompts.append(task["prompt"])

            target_boxes_tensor = build_target_boxes_tensor(task["target_boxes"])
            target = {
                "boxes": target_boxes_tensor.to(device),   # absolute xyxy
                "image_size": None,
                "case_id": s.get("case_id", ""),
                "slice_z": s.get("slice_z", -1),
                "task_type": task.get("task_type", s.get("task_type", "")),
                "canonical_name": task.get("canonical_name", s.get("canonical_name", "")),
            }
            targets.append(target)
            num_negative_tasks += int(task.get("is_negative", False))

    images, padded_hw = stack_padded_images(images)
    padded_hw = (int(padded_hw[0]), int(padded_hw[1]))
    for tgt in targets:
        tgt["image_size"] = padded_hw
    images = images.to(device, non_blocking=True)
    return images, prompts, targets, num_negative_tasks


def union_boxes_xyxy(boxes: List[List[float]]) -> torch.Tensor:
    """
    Union multiple xyxy boxes into one xyxy box.
    """
    boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
    x1 = boxes_t[:, 0].min()
    y1 = boxes_t[:, 1].min()
    x2 = boxes_t[:, 2].max()
    y2 = boxes_t[:, 3].max()
    return torch.stack([x1, y1, x2, y2], dim=0)


def prepare_val_task_batch(
    batch_samples: List[Dict[str, Any]],
    device: str,
    move_images_to_device: bool = True,
):
    images = []
    prompts = []
    gt_boxes_abs = []
    image_sizes = []
    image_paths = []
    metas = []

    for s in batch_samples:
        images.append(s["image"])
        prompts.append(s["prompt"])

        gt_union = union_boxes_xyxy(s["target_boxes"])
        gt_boxes_abs.append(gt_union.to(device))

        image_sizes.append(s["image_size"])
        image_paths.append(s.get("image_path", ""))
        metas.append({
            "case_id": s.get("case_id", ""),
            "slice_z": s.get("slice_z", -1),
            "canonical_name": s.get("canonical_name", "unknown"),
            "eval_prompt_type": s.get("eval_prompt_type", "unknown"),
            "image_path": s.get("image_path", ""),
        })

    images, padded_hw = stack_padded_images(images)
    padded_hw = (int(padded_hw[0]), int(padded_hw[1]))
    image_sizes = [padded_hw for _ in image_sizes]
    if move_images_to_device:
        images = images.to(device, non_blocking=True)
    gt_boxes_abs = torch.stack(gt_boxes_abs, dim=0).to(device, non_blocking=True)
    return images, prompts, gt_boxes_abs, image_sizes, image_paths, metas


# =========================================================
# Config builders
# =========================================================
def build_detector_cfg(args) -> Dict[str, Any]:
    return {
        "sam3_ckpt": args.sam3_ckpt,
        "image_backbone_name": args.image_backbone_name,

        # 两个 encoder 都冻结
        "freeze_image_encoder": True,
        "freeze_text_encoder": True,

        "return_image_last_hidden_state": False,
        "return_image_position_encoding": False,

        # text adapter
        "use_medical_lexicon_adapter": bool(args.use_medical_lexicon_adapter),
        "adapter_bottleneck_dim": args.adapter_bottleneck_dim,
        "adapter_phrase_out_dim": args.adapter_phrase_out_dim,
        "adapter_dropout": args.adapter_dropout,
        "adapter_gate_init": args.adapter_gate_init,
        "max_length": args.max_length,

        # decoder
        "text_token_dim": 1024,
        "phrase_dim": (args.adapter_phrase_out_dim if bool(args.use_medical_lexicon_adapter) else 512),
        "embed_dim": args.embed_dim,
        "num_queries": args.num_queries,
        "num_decoder_layers": args.num_decoder_layers,
        "num_heads": args.num_heads,
        "num_points": args.num_points,
        "ffn_dim": args.ffn_dim,
        "dropout": args.dropout,
        "topk": args.topk,
        "fusion_num_layers": args.fusion_num_layers,
        "fusion_top_levels": args.fusion_top_levels,
        "fusion_use_text_bias": bool(args.fusion_use_text_bias),
        "fusion_gate_init": args.fusion_gate_init,
        "use_presence_branch": bool(args.use_presence_branch),
    }


def build_criterion_cfg(args) -> Dict[str, Any]:
    return {
        "target_box_format": "xyxy_abs",

        # matcher
        "matcher_cost_bbox": args.matcher_cost_bbox,
        "matcher_cost_giou": args.matcher_cost_giou,
        "matcher_cost_class": args.matcher_cost_class,

        # losses
        "loss_bbox_weight": args.loss_bbox_weight,
        "loss_giou_weight": args.loss_giou_weight,
        "loss_class_weight": args.loss_class_weight,
        "loss_presence_weight": args.loss_presence_weight,
        "use_presence_branch": bool(args.use_presence_branch),

        # aux
        "use_aux_loss": bool(args.use_aux_loss),
        "aux_loss_weight": args.aux_loss_weight,

        # BCE weights
        "class_pos_weight": args.class_pos_weight,
        "presence_pos_weight": args.presence_pos_weight,
    }


# =========================================================
# Frozen encoder guard
# =========================================================
def keep_frozen_encoders_in_eval(detector: nn.Module):
    model = unwrap_model(detector)

    if hasattr(model, "image_backbone") and hasattr(model.image_backbone, "vision_encoder"):
        model.image_backbone.vision_encoder.eval()

    if hasattr(model, "text_backbone") and hasattr(model.text_backbone, "text_encoder"):
        model.text_backbone.text_encoder.eval()


# =========================================================
# Param / grad helpers
# =========================================================
def get_trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_grad_norm(parameters) -> float:
    total_sq = 0.0
    has_grad = False
    for p in parameters:
        if p.grad is not None:
            g = p.grad.detach()
            total_sq += float((g * g).sum().item())
            has_grad = True
    if not has_grad:
        return 0.0
    return math.sqrt(total_sq)


# =========================================================
# Metric helpers
# =========================================================
def normalize_xyxy_abs(boxes_xyxy_abs: torch.Tensor, image_size) -> torch.Tensor:
    h, w = image_size
    out = boxes_xyxy_abs.clone().float()
    out[:, 0] /= float(w)
    out[:, 2] /= float(w)
    out[:, 1] /= float(h)
    out[:, 3] /= float(h)
    return out.clamp(0.0, 1.0)


def box_iou_diag(boxes1_xyxy: torch.Tensor, boxes2_xyxy: torch.Tensor) -> torch.Tensor:
    x1 = torch.max(boxes1_xyxy[:, 0], boxes2_xyxy[:, 0])
    y1 = torch.max(boxes1_xyxy[:, 1], boxes2_xyxy[:, 1])
    x2 = torch.min(boxes1_xyxy[:, 2], boxes2_xyxy[:, 2])
    y2 = torch.min(boxes1_xyxy[:, 3], boxes2_xyxy[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]).clamp(min=0) * \
            (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1]).clamp(min=0)
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]).clamp(min=0) * \
            (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1]).clamp(min=0)

    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def compute_matched_mean_iou(
    outputs: Dict[str, torch.Tensor],
    targets: List[Dict[str, Any]],
    indices: List[Tuple[torch.Tensor, torch.Tensor]],
) -> float:
    pred_boxes = outputs["pred_boxes"]
    pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)

    ious = []
    for b, (src_idx, tgt_idx) in enumerate(indices):
        if len(src_idx) == 0:
            continue

        tgt_boxes_abs = targets[b]["boxes"].float()
        tgt_xyxy_norm = normalize_xyxy_abs(tgt_boxes_abs, targets[b]["image_size"]).to(pred_boxes.device)

        src_xyxy = pred_boxes_xyxy[b, src_idx]
        tgt_xyxy = tgt_xyxy_norm[tgt_idx]

        pair_ious = box_iou_diag(src_xyxy, tgt_xyxy)
        ious.extend(pair_ious.detach().cpu().tolist())

    if len(ious) == 0:
        return 0.0
    return float(np.mean(ious))

def box_iou_one_to_many(boxes_xyxy: torch.Tensor, gt_box_xyxy: torch.Tensor) -> torch.Tensor:
    """
    boxes_xyxy: [K,4]
    gt_box_xyxy: [4]
    return: [K]
    """
    x1 = torch.maximum(boxes_xyxy[:, 0], gt_box_xyxy[0])
    y1 = torch.maximum(boxes_xyxy[:, 1], gt_box_xyxy[1])
    x2 = torch.minimum(boxes_xyxy[:, 2], gt_box_xyxy[2])
    y2 = torch.minimum(boxes_xyxy[:, 3], gt_box_xyxy[3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=0) * \
            (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=0)
    area2 = (gt_box_xyxy[2] - gt_box_xyxy[0]).clamp(min=0) * \
            (gt_box_xyxy[3] - gt_box_xyxy[1]).clamp(min=0)

    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def reduce_scalar(value: float, device: str) -> float:
    if get_world_size() == 1:
        return value
    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= get_world_size()
    return float(t.item())


def gather_class_metric_stats(local_stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if get_world_size() == 1:
        return local_stats
    gathered = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, local_stats)
    merged: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "n": 0.0,
            "top1_sum": 0.0,
            "oracle_sum": 0.0,
            "r50_sum": 0.0,
            "r75_sum": 0.0,
        }
    )
    for rank_stats in gathered:
        for canonical_name, stats in rank_stats.items():
            dst = merged[canonical_name]
            dst["n"] += float(stats["n"])
            dst["top1_sum"] += float(stats["top1_sum"])
            dst["oracle_sum"] += float(stats["oracle_sum"])
            dst["r50_sum"] += float(stats["r50_sum"])
            dst["r75_sum"] += float(stats["r75_sum"])
    return dict(merged)


# =========================================================
# Save checkpoint
# =========================================================
def save_checkpoint(
    save_path: str,
    detector: nn.Module,
    save_fp16: bool = True,
    save_frozen_encoders: bool = False,
):
    if not is_main_process():
        return

    model = unwrap_model(detector)
    state_dict = model.state_dict()
    compact_state_dict = {}
    frozen_encoder_prefixes = (
        "image_backbone.vision_encoder.",
        "text_backbone.text_encoder.",
    )
    for name, tensor in state_dict.items():
        if (not save_frozen_encoders) and any(
            name.startswith(prefix) for prefix in frozen_encoder_prefixes
        ):
            continue
        value = tensor.detach().cpu()
        if save_fp16 and torch.is_floating_point(value):
            value = value.to(dtype=torch.float16)
        compact_state_dict[name] = value

    torch.save(compact_state_dict, save_path)


# =========================================================
# Epoch loops
# =========================================================
def train_one_epoch(
    epoch: int,
    detector: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler,
    args,
):
    detector.train()
    criterion.train()
    keep_frozen_encoders_in_eval(detector)

    total_loss = 0.0
    total_bbox = 0.0
    total_giou = 0.0
    total_cls = 0.0
    total_presence = 0.0
    total_iou = 0.0
    total_negative_tasks = 0
    total_prompt_tasks = 0
    n_steps = 0


    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"Train {epoch:03d}",
        dynamic_ncols=True,
        leave=False,
        disable=not is_main_process(),
    )

    for step, batch_samples in enumerate(progress):
        images, prompts, targets, num_negative_tasks = prepare_batch(
            batch_samples=batch_samples,
            device=args.device,
            mode="train",
        )

        with autocast("cuda", enabled=bool(args.amp)):
            outputs = detector(
                images=images,
                prompts=prompts,
                topk=None,
                return_image_features=False,
                return_text_outputs=False,
                return_aux=True,
                return_image_aux=False,
            )

            loss, loss_dict, indices = criterion(outputs, targets)
            mean_iou = compute_matched_mean_iou(outputs, targets, indices)

            loss_to_backward = loss / args.accum_steps

        scaler.scale(loss_to_backward).backward()

        if (step + 1) % args.accum_steps == 0:
            if args.grad_clip is not None and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(get_trainable_params(detector), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().item())
        total_bbox += float(loss_dict["loss_bbox"].detach().item())
        total_giou += float(loss_dict["loss_giou"].detach().item())
        total_cls += float(loss_dict["loss_class"].detach().item())
        total_presence += float(loss_dict["loss_presence"].detach().item())
        total_iou += mean_iou
        total_negative_tasks += int(num_negative_tasks)
        total_prompt_tasks += int(len(targets))
        n_steps += 1

        if is_main_process():
            progress.set_postfix({
                "loss": f"{total_loss / n_steps:.4f}",
                "miou": f"{total_iou / n_steps:.4f}",
                "cls": f"{total_cls / n_steps:.4f}",
                "neg": f"{total_negative_tasks / max(total_prompt_tasks, 1):.2f}",
            })


    # 处理最后不足 accum_steps 的残留梯度
    if n_steps > 0 and (n_steps % args.accum_steps != 0):
        if args.grad_clip is not None and args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(get_trainable_params(detector), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    if scheduler is not None:
        scheduler.step()

    stats = {
        "loss_total": total_loss / max(n_steps, 1),
        "loss_bbox": total_bbox / max(n_steps, 1),
        "loss_giou": total_giou / max(n_steps, 1),
        "loss_class": total_cls / max(n_steps, 1),
        "loss_presence": total_presence / max(n_steps, 1),
        "matched_mean_iou": total_iou / max(n_steps, 1),
        "negative_task_ratio": total_negative_tasks / max(total_prompt_tasks, 1),
        "num_negative_tasks": float(total_negative_tasks),
        "num_prompt_tasks": float(total_prompt_tasks),
        "grad_norm": get_grad_norm(get_trainable_params(detector)),
        "lr": optimizer.param_groups[0]["lr"],
    }


    for k in list(stats.keys()):
        stats[k] = reduce_scalar(stats[k], device=args.device)

    return stats


@torch.no_grad()
def evaluate(
    epoch: int,
    detector: nn.Module,
    loader: DataLoader,
    args,
):
    """
    Test-like validation:
    - expand one image into multiple organ-level tasks
    - for each organ, only use one prompt description
    - evaluate real inference top1 / oracle-topk metrics
    """
    detector.eval()
    keep_frozen_encoders_in_eval(detector)

    total_top1_iou = 0.0
    total_oracle_iou = 0.0
    total_r50 = 0.0
    total_r75 = 0.0
    total_n = 0
    total_cache_hits = 0
    total_cache_misses = 0
    class_metric_stats: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "n": 0.0,
            "top1_sum": 0.0,
            "oracle_sum": 0.0,
            "r50_sum": 0.0,
            "r75_sum": 0.0,
        }
    )

    model_for_eval = unwrap_model(detector)

    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"Val   {epoch:03d}",
        dynamic_ncols=True,
        leave=False,
        disable=not is_main_process(),
    )

    for batch_samples in progress:
        use_disk_cache = bool(args.use_disk_image_feature_cache)
        images, prompts, gt_boxes_abs, image_sizes, image_paths, metas = prepare_val_task_batch(
            batch_samples=batch_samples,
            device=args.device,
            move_images_to_device=not use_disk_cache,
        )

        if use_disk_cache:
            multi_scale_feats, cache_stats = get_batched_cached_image_features(
                model=model_for_eval,
                image_tensors=[images[i] for i in range(images.shape[0])],
                image_paths=image_paths,
                device=args.device,
                cache_dir=args.disk_image_feature_cache_dir,
                enable_disk_cache=True,
            )
            total_cache_hits += int(cache_stats["cache_hits"])
            total_cache_misses += int(cache_stats["cache_misses"])

            with autocast("cuda", enabled=bool(args.amp)):
                text_outputs = model_for_eval.forward_text(
                    prompts=prompts,
                    batch_size=len(prompts),
                    return_raw_text=False,
                    return_text_aux=True,
                )
                outputs = model_for_eval.forward_decoder(
                    multi_scale_feats=multi_scale_feats,
                    text_outputs=text_outputs,
                    topk=args.topk,
                    return_aux=False,
                )
        else:
            with autocast("cuda", enabled=bool(args.amp)):
                outputs = detector(
                    images=images,
                    prompts=prompts,
                    topk=args.topk,
                    return_image_features=False,
                    return_text_outputs=False,
                    return_aux=False,
                    return_image_aux=False,
                )

        # [B,K,4] normalized cxcywh -> normalized xyxy
        pred_topk_xyxy_norm = box_cxcywh_to_xyxy(outputs["topk_boxes"]).clamp(0.0, 1.0)

        B, K = pred_topk_xyxy_norm.shape[:2]
        for b in range(B):
            H, W = image_sizes[b]
            pred_topk_abs = pred_topk_xyxy_norm[b].clone()
            pred_topk_abs[:, 0] *= W
            pred_topk_abs[:, 2] *= W
            pred_topk_abs[:, 1] *= H
            pred_topk_abs[:, 3] *= H

            gt_box = gt_boxes_abs[b]  # [4]
            ious = box_iou_one_to_many(pred_topk_abs, gt_box)

            top1_iou = float(ious[0].item())
            oracle_iou = float(ious.max().item())

            total_top1_iou += top1_iou
            total_oracle_iou += oracle_iou
            total_r50 += float(top1_iou >= 0.5)
            total_r75 += float(top1_iou >= 0.75)
            total_n += 1

            canonical_name = metas[b].get("canonical_name", "unknown")
            class_stats = class_metric_stats[canonical_name]
            class_stats["n"] += 1.0
            class_stats["top1_sum"] += top1_iou
            class_stats["oracle_sum"] += oracle_iou
            class_stats["r50_sum"] += float(top1_iou >= 0.5)
            class_stats["r75_sum"] += float(top1_iou >= 0.75)

        if is_main_process() and total_n > 0:
            postfix = {
                "top1": f"{total_top1_iou / total_n:.4f}",
                "r50": f"{total_r50 / total_n:.4f}",
                "oracle": f"{total_oracle_iou / total_n:.4f}",
            }
            if use_disk_cache:
                total_cache = total_cache_hits + total_cache_misses
                postfix["cache"] = f"{total_cache_hits}/{max(total_cache, 1)}"
            progress.set_postfix(postfix)

    global_class_metric_stats = gather_class_metric_stats(dict(class_metric_stats))
    class_summaries = []
    total_n = 0.0
    for canonical_name in sorted(global_class_metric_stats.keys()):
        s = global_class_metric_stats[canonical_name]
        n = float(s["n"])
        if n <= 0:
            continue
        total_n += n
        class_summaries.append({
            "det_mIoU_top1": float(s["top1_sum"] / n),
            "det_mIoU_oracle_topk": float(s["oracle_sum"] / n),
            "det_Recall@0.5_top1": float(s["r50_sum"] / n),
            "det_Recall@0.75_top1": float(s["r75_sum"] / n),
        })

    if len(class_summaries) == 0:
        stats = {
            "val_N": 0.0,
            "det_mIoU_top1": 0.0,
            "det_mIoU_oracle_topk": 0.0,
            "det_Recall@0.5_top1": 0.0,
            "det_Recall@0.75_top1": 0.0,
        }
    else:
        stats = {
            "val_N": float(total_n),
            "det_mIoU_top1": float(sum(x["det_mIoU_top1"] for x in class_summaries) / len(class_summaries)),
            "det_mIoU_oracle_topk": float(sum(x["det_mIoU_oracle_topk"] for x in class_summaries) / len(class_summaries)),
            "det_Recall@0.5_top1": float(sum(x["det_Recall@0.5_top1"] for x in class_summaries) / len(class_summaries)),
            "det_Recall@0.75_top1": float(sum(x["det_Recall@0.75_top1"] for x in class_summaries) / len(class_summaries)),
        }

    return stats


# =========================================================
# Main
# =========================================================
@record
def main():
    ap = argparse.ArgumentParser()

    # data
    # ap.add_argument("--train_ann_path", default="data/cardiacUDC_A4C/ann/odvg_train.json")
    # ap.add_argument("--val_ann_path", default="data/cardiacUDC_A4C/ann/odvg_test.json")
    # ap.add_argument("--image_root", default="data/cardiacUDC_A4C")
    # ap.add_argument("--prompt_bank", default="data/cardiacUDC_A4C/prompt_bank.json")

    ap.add_argument("--train_ann_path", default="data/flare22/ann/odvg_train.json")
    ap.add_argument("--val_ann_path", default="data/flare22/ann/odvg_test.json")
    ap.add_argument("--image_root", default="data/flare22")
    ap.add_argument("--prompt_bank", default="data/flare22/prompt_bank.json")

    # training control
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size_per_gpu", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--accum_steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)

    # prompt sampling
    ap.add_argument("--train_positive_sampling", choices=["single", "all"], default="all")
    ap.add_argument("--train_use_template", type=int, default=0)
    ap.add_argument("--p_negative", type=float, default=0.20)
    ap.add_argument("--max_negatives", type=int, default=1)

    # eval prompt
    ap.add_argument("--eval_prompt_type", choices=["canonical", "heldout"], default="canonical")
    ap.add_argument("--eval_prompt_index", type=int, default=0)
    ap.add_argument("--eval_use_template", type=int, default=0)

    # detector
    ap.add_argument("--sam3_ckpt", default="/data3/users/zhaojun/project/sam3")
    ap.add_argument("--image_backbone_name", choices=["sam3_fpn"], default="sam3_fpn")

    # text adapter
    ap.add_argument("--use_medical_lexicon_adapter", type=int, default=1)
    ap.add_argument("--adapter_bottleneck_dim", type=int, default=256)
    ap.add_argument("--adapter_phrase_out_dim", type=int, default=512)
    ap.add_argument("--adapter_dropout", type=float, default=0.1)
    ap.add_argument("--adapter_gate_init", type=float, default=0.1)
    ap.add_argument("--max_length", type=int, default=32)

    # decoder
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--num_queries", type=int, default=100)
    ap.add_argument("--num_decoder_layers", type=int, default=6)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_points", type=int, default=5)
    ap.add_argument("--ffn_dim", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--fusion_num_layers", type=int, default=2)
    ap.add_argument("--fusion_top_levels", type=int, default=1)
    ap.add_argument("--fusion_use_text_bias", type=int, default=1)
    ap.add_argument("--fusion_gate_init", type=float, default=0.1)
    ap.add_argument("--use_presence_branch", type=int, default=1)

    # matcher
    ap.add_argument("--matcher_cost_bbox", type=float, default=5.0)
    ap.add_argument("--matcher_cost_giou", type=float, default=2.0)
    ap.add_argument("--matcher_cost_class", type=float, default=1.0)

    # losses
    ap.add_argument("--loss_bbox_weight", type=float, default=5.0)
    ap.add_argument("--loss_giou_weight", type=float, default=2.0)
    ap.add_argument("--loss_class_weight", type=float, default=1.0)
    ap.add_argument("--loss_presence_weight", type=float, default=1.0)

    # aux
    ap.add_argument("--use_aux_loss", type=int, default=1)
    ap.add_argument("--aux_loss_weight", type=float, default=1.0)

    # BCE weights
    ap.add_argument("--class_pos_weight", type=float, default=1.0)
    ap.add_argument("--presence_pos_weight", type=float, default=1.0)

    # optimization
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", type=int, default=1)

    # save / log
    ap.add_argument("--save_dir", default="./runs_5point")
    ap.add_argument("--save_frozen_encoders", type=int, default=0)
    ap.add_argument("--use_disk_image_feature_cache", type=int, default=0)
    ap.add_argument("--disk_image_feature_cache_dir", default="./image_feature_cache")

    args = ap.parse_args()

    setup_distributed(args)
    args.device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed, rank=args.rank)

    if is_main_process():
        ensure_dir(args.save_dir)
    barrier()

    if bool(args.use_disk_image_feature_cache):
        args.disk_image_feature_cache_dir = resolve_feature_cache_dir(
            explicit_dir=args.disk_image_feature_cache_dir,
            base_dir=args.save_dir,
        )
        if is_main_process():
            ensure_dir(args.disk_image_feature_cache_dir)
    barrier()

    logger = setup_logger(args.save_dir)

    if is_main_process():
        logger.info("=" * 100)
        logger.info("CONFIG")
        logger.info("=" * 100)
        logger.info(json.dumps(vars(args), indent=2, ensure_ascii=False))

    # -------------------------------------------------
    # datasets / loaders
    # -------------------------------------------------
    train_dataset = build_train_dataset(args)

    # val: first build image-level base dataset, then expand to organ-level task dataset
    val_base_dataset = build_val_dataset(args)
    val_dataset = ValTaskDataset(
        base_dataset=val_base_dataset,
        eval_prompt_type=args.eval_prompt_type,
        eval_prompt_index=args.eval_prompt_index,
    )

    train_loader, train_sampler = build_dataloader(train_dataset, args, train=True)
    val_loader, val_sampler = build_dataloader(val_dataset, args, train=False)

    if is_main_process():
        logger.info("=" * 100)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 100)
        logger.info(f"train size     : {len(train_dataset)}")
        logger.info(f"val image size : {len(val_base_dataset)}")
        logger.info(f"val task size  : {len(val_dataset)}")


    # -------------------------------------------------
    # model / criterion / optimizer / scheduler
    # -------------------------------------------------
    detector_cfg = build_detector_cfg(args)
    detector = build_detector(detector_cfg).to(args.device)
    model_for_setup = detector

    criterion_cfg = build_criterion_cfg(args)
    criterion = build_detection_criterion(criterion_cfg).to(args.device)

    optimizer = torch.optim.AdamW(
        [p for p in detector.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.1,
    )

    scaler = GradScaler("cuda", enabled=bool(args.amp))

    if args.world_size > 1:
        detector = DDP(
            detector,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    if is_main_process():
        model_for_log = unwrap_model(detector)
        total_params, trainable_params = count_params(model_for_log)

        logger.info("=" * 100)
        logger.info("MODEL SUMMARY")
        logger.info("=" * 100)
        logger.info(f"detector: {model_for_log.__class__.__name__}")
        logger.info(f"image_backbone: {model_for_log.image_backbone.__class__.__name__}")
        logger.info(f"text_backbone : {model_for_log.text_backbone.__class__.__name__}")
        logger.info(f"decoder       : {model_for_log.decoder.__class__.__name__}")
        logger.info(f"total_params     : {total_params}")
        logger.info(f"trainable_params : {trainable_params}")

    best_metric = -1.0
    best_epoch = -1
    start_time = time.time()

    # -------------------------------------------------
    # training loop
    # -------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            epoch=epoch,
            detector=detector,
            criterion=criterion,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            args=args,
        )

        val_stats = evaluate(
            epoch=epoch,
            detector=detector,
            loader=val_loader,
            args=args,
        )


        metric = val_stats["det_mIoU_top1"]

        is_best = metric > best_metric
        if is_best:
            best_metric = metric
            best_epoch = epoch
            save_checkpoint(
                save_path=os.path.join(args.save_dir, "best.pt"),
                detector=detector,
                save_frozen_encoders=bool(args.save_frozen_encoders),
            )

        if is_main_process():
            elapsed = (time.time() - start_time) / 60.0
            logger.info(
                f"[Epoch {epoch:04d}/{args.epochs:04d}] "
                f"train_loss={train_stats['loss_total']:.4f} "
                f"train_match_iou={train_stats['matched_mean_iou']:.4f} | "
                f"bbox={train_stats['loss_bbox']:.4f} "
                f"giou={train_stats['loss_giou']:.4f} "
                f"cls={train_stats['loss_class']:.4f} "
                f"pres={train_stats['loss_presence']:.4f} "
                f"neg={train_stats['negative_task_ratio']:.2f} | "
                f"val_det_mIoU={val_stats['det_mIoU_top1']:.4f} "
                f"val_det_R50={val_stats['det_Recall@0.5_top1']:.4f} "
                f"val_det_R75={val_stats['det_Recall@0.75_top1']:.4f} "
                f"val_oracle_topk={val_stats['det_mIoU_oracle_topk']:.4f} "
                f"val_N={int(val_stats['val_N'])} | "
                f"lr={train_stats['lr']:.6f} "
                f"grad={train_stats['grad_norm']:.4f} | "
                f"best_det_mIoU={best_metric:.4f}@{best_epoch} | "
                f"time={elapsed:.1f}m"
            )



            log_jsonl(args.save_dir, {
                "epoch": epoch,
                "train": train_stats,
                "val": val_stats,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
            })

    save_checkpoint(
        save_path=os.path.join(args.save_dir, "last.pt"),
        detector=detector,
        save_frozen_encoders=bool(args.save_frozen_encoders),
    )

    if is_main_process():
        logger.info("=" * 100)
        logger.info("DONE")
        logger.info("=" * 100)
        logger.info(f"Best val matched IoU: {best_metric:.4f} at epoch {best_epoch}")
        logger.info(f"Saved dir: {args.save_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    main()

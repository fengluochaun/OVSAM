import os
import sys
import csv
import json
import argparse
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist

# CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --standalone --nproc_per_node=4 test.py

# =========================================================
# project root
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


# =========================================================
# SAM3 imports
# =========================================================
try:
    from transformers import Sam3Processor, Sam3Model
except Exception as e:
    raise RuntimeError(f"Cannot import Sam3Processor / Sam3Model from transformers: {e}")


def import_sam3tracker():
    """
    Different transformers builds may expose different class names.
    Try common import paths.
    """
    try:
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        return Sam3TrackerModel, Sam3TrackerProcessor
    except Exception:
        raise ImportError(
            "Cannot import Sam3TrackerModel/Sam3TrackerProcessor from transformers. "
            "Please ensure your transformers version includes sam3_tracker."
        )


# =========================================================
# basic utils
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def init_distributed(device_arg: str) -> Tuple[bool, int, int, int, str]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1

    if is_distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed test requires CUDA.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
    else:
        device = device_arg

    return is_distributed, rank, local_rank, world_size, device


def is_main_process(rank: int) -> bool:
    return rank == 0


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def load_checkpoint_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    return strip_module_prefix(state_dict)


def infer_fusion_cfg_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    fusion_layer_ids = set()
    text_bias_level_count = 0

    for key in state_dict.keys():
        if key.startswith("decoder.fusion_encoder.layers."):
            parts = key.split(".")
            if len(parts) > 3 and parts[3].isdigit():
                fusion_layer_ids.add(int(parts[3]))

    gates_key = "decoder.fusion_encoder.text_conditioner.level_gates"
    if gates_key in state_dict:
        gates = state_dict[gates_key]
        text_bias_level_count = int(gates.shape[0]) if gates.ndim > 0 else 1

    return {
        "fusion_num_layers": (max(fusion_layer_ids) + 1) if len(fusion_layer_ids) > 0 else 0,
        "fusion_top_levels": max(text_bias_level_count, 1),
        "fusion_use_text_bias": int(text_bias_level_count > 0),
    }


def apply_inferred_decoder_cfg(args, state_dict: Dict[str, torch.Tensor], verbose: bool = True) -> None:
    fusion_cfg = infer_fusion_cfg_from_state_dict(state_dict)
    args.fusion_num_layers = fusion_cfg["fusion_num_layers"]
    args.fusion_top_levels = fusion_cfg["fusion_top_levels"]
    args.fusion_use_text_bias = fusion_cfg["fusion_use_text_bias"]
    args.use_medical_lexicon_adapter = int(any(k.startswith("text_backbone.adapter.") for k in state_dict.keys()))
    args.use_presence_branch = int(
        any(
            ("presence_" in k) or ("presence_token" in k)
            for k in state_dict.keys()
        )
    )
    if not bool(args.use_medical_lexicon_adapter):
        args.adapter_phrase_out_dim = 512
    if verbose:
        print(
            "[INFO] inferred decoder cfg from checkpoint:"
            f" fusion_num_layers={args.fusion_num_layers}"
            f" fusion_top_levels={args.fusion_top_levels}"
            f" fusion_use_text_bias={args.fusion_use_text_bias}"
            f" use_medical_lexicon_adapter={args.use_medical_lexicon_adapter}"
            f" use_presence_branch={args.use_presence_branch}"
        )


def load_checkpoint_to_model(
    model,
    ckpt_path: str,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    verbose: bool = True,
):
    if state_dict is None:
        state_dict = load_checkpoint_state_dict(ckpt_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    allowed_missing_prefixes = (
        "image_backbone.vision_encoder.",
        "text_backbone.text_encoder.",
    )
    invalid_missing = [
        k for k in missing_keys
        if not any(k.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if unexpected_keys or invalid_missing:
        raise RuntimeError(
            "Failed to load checkpoint strictly enough. "
            f"unexpected_keys={list(unexpected_keys)[:20]} "
            f"invalid_missing_keys={invalid_missing[:20]} "
            f"ckpt_path={ckpt_path}"
        )
    if verbose:
        print(f"[INFO] Loaded checkpoint: {ckpt_path}")
        if len(missing_keys) > 0:
            print(
                "[INFO] Missing frozen SAM3 encoder keys were kept from --sam3_ckpt: "
                f"{len(missing_keys)}"
            )


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    image_tensor: [3,H,W], either [0,1] or [0,255]
    """
    x = image_tensor.detach().cpu().float()
    if x.max() <= 1.5:
        x = x.clamp(0, 1) * 255.0
    else:
        x = x.clamp(0, 255.0)
    x = x.byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x, mode="RGB")


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def norm_xyxy_to_abs(box_xyxy_norm: torch.Tensor, H: int, W: int) -> torch.Tensor:
    out = box_xyxy_norm.clone()
    out[..., 0] *= W
    out[..., 2] *= W
    out[..., 1] *= H
    out[..., 3] *= H
    return out


def clamp_box_xyxy(box: np.ndarray, H: int, W: int) -> np.ndarray:
    b = box.copy().astype(np.float32)
    b[0] = np.clip(b[0], 0, W - 1)
    b[2] = np.clip(b[2], 0, W - 1)
    b[1] = np.clip(b[1], 0, H - 1)
    b[3] = np.clip(b[3], 0, H - 1)
    if b[2] < b[0]:
        b[2] = b[0]
    if b[3] < b[1]:
        b[3] = b[1]
    return b


def union_boxes_xyxy(boxes: List[List[float]]) -> np.ndarray:
    arr = np.asarray(boxes, dtype=np.float32)
    x1 = arr[:, 0].min()
    y1 = arr[:, 1].min()
    x2 = arr[:, 2].max()
    y2 = arr[:, 3].max()
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
    union = area1 + area2 - inter
    return inter / (union + 1e-8)


def dice_iou_mask(pred01: np.ndarray, gt01: np.ndarray) -> Tuple[float, float]:
    pred = pred01.astype(bool)
    gt = gt01.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    dice = float(2 * inter) / float(pred.sum() + gt.sum() + 1e-8)
    iou = float(inter) / float(union + 1e-8)
    return dice, iou


# =========================================================
# dataset / eval task helpers
# =========================================================
def build_eval_dataset(args):
    prompt_bank = load_prompt_bank(args.prompt_bank)
    sampler = PromptBankSampler(
        prompt_bank=prompt_bank,
        mode="eval",
        eval_use_template=bool(getattr(args, "eval_use_template", 0)),
        p_negative=0.0,
        p_consistency=0.0,
        max_negatives=0,
        seed=args.seed,
    )
    dataset = OVSAMDataset(
        ann_path=args.ann_path,
        image_root=args.image_root,
        prompt_sampler=sampler,
        return_raw_if_empty=True,
    )
    return dataset


def collect_eval_tasks(sample: Dict[str, Any], eval_modes: List[str]) -> List[Dict[str, Any]]:
    tasks = []

    if "canonical" in eval_modes:
        for q in sample.get("canonical_queries", []):
            q2 = dict(q)
            q2["eval_group"] = "canonical"
            tasks.append(q2)

    if "heldout" in eval_modes:
        for q in sample.get("heldout_queries", []):
            q2 = dict(q)
            q2["eval_group"] = "heldout"
            tasks.append(q2)

    return tasks


def extract_present_phrase_to_cat(sample: Dict[str, Any]) -> Dict[str, int]:
    phrase2cat = {}
    for inst in sample.get("instances", []):
        phrase = inst.get("phrase", None)
        cat_id = inst.get("category_id", None)
        if phrase is not None and cat_id is not None:
            phrase2cat[phrase] = int(cat_id)
    return phrase2cat


def resolve_mask_path(sample: Dict[str, Any], masks_root: str) -> str:
    """
    Your masks are under:
      data/flare22/masks/train
      data/flare22/masks/test
    Infer split from image_path.
    """
    img_path = sample.get("image_path", "")
    img_name = os.path.basename(img_path)

    split = None
    norm = img_path.replace("\\", "/")
    if "/images/train/" in norm:
        split = "train"
    elif "/images/test/" in norm:
        split = "test"
    else:
        # fallback from ann path style names if needed
        if "train" in norm.lower():
            split = "train"
        else:
            split = "test"

    p = os.path.join(masks_root, split, img_name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Mask not found: {p}")
    return p


def build_gt_binary_mask(
    sample: Dict[str, Any],
    task: Dict[str, Any],
    label_map_u8: np.ndarray,
) -> np.ndarray:
    cat_ids = []

    target_cat_ids = task.get("target_cat_ids", None)
    if target_cat_ids is not None:
        for cid in target_cat_ids:
            cid = int(cid)
            if cid >= 0:
                cat_ids.append(cid)

    if len(cat_ids) == 0:
        phrase2cat = extract_present_phrase_to_cat(sample)
        eval_group = task.get("eval_group", "")
        canonical_name = task.get("canonical_name", "")

        if canonical_name in phrase2cat:
            cat_ids.append(int(phrase2cat[canonical_name]))

    if len(cat_ids) == 0:
        raise RuntimeError(
            f"Cannot build GT binary mask for task={task.get('canonical_name')} "
            f"prompt={task.get('prompt')}"
        )

    gt01 = np.zeros_like(label_map_u8, dtype=np.uint8)
    for cid in cat_ids:
        gt01 = np.maximum(gt01, (label_map_u8 == cid).astype(np.uint8))
    return gt01


# =========================================================
# detector helpers
# =========================================================
def build_detector_cfg(args) -> Dict[str, Any]:
    return {
        "sam3_ckpt": args.sam3_ckpt,
        "image_backbone_name": args.image_backbone_name,

        "freeze_image_encoder": True,
        "freeze_text_encoder": True,

        "return_image_last_hidden_state": False,
        "return_image_position_encoding": False,

        "use_medical_lexicon_adapter": bool(args.use_medical_lexicon_adapter),
        "adapter_bottleneck_dim": args.adapter_bottleneck_dim,
        "adapter_phrase_out_dim": args.adapter_phrase_out_dim,
        "adapter_dropout": args.adapter_dropout,
        "adapter_gate_init": args.adapter_gate_init,
        "max_length": args.max_length,

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


@torch.no_grad()
def run_detector_one(
    detector,
    image_tensor: torch.Tensor,      # [3,H,W], used for geometry and optional image encode
    prompt: str,
    topk: int,
    device: str,
    amp: bool = True,
    precomputed_image_feats: Optional[List[torch.Tensor]] = None,
) -> Dict[str, Any]:
    with torch.autocast(device_type="cuda", enabled=(amp and "cuda" in device)):
        if precomputed_image_feats is None:
            image = image_tensor.unsqueeze(0).to(device)
            outputs = detector(
                images=image,
                prompts=[prompt],
                topk=topk,
                return_image_features=False,
                return_text_outputs=False,
                return_aux=False,
                return_image_aux=False,
            )
        else:
            text_outputs = detector.forward_text(
                prompts=[prompt],
                batch_size=1,
                return_raw_text=False,
                return_text_aux=True,
            )
            outputs = detector.forward_decoder(
                multi_scale_feats=precomputed_image_feats,
                text_outputs=text_outputs,
                topk=topk,
                return_aux=False,
            )

    H, W = image_tensor.shape[-2], image_tensor.shape[-1]
    topk_boxes = outputs["topk_boxes"][0]      # [K,4], normalized cxcywh
    topk_scores = outputs["topk_scores"][0]    # [K]

    topk_xyxy = cxcywh_to_xyxy(topk_boxes).clamp(0, 1)
    topk_xyxy_abs = norm_xyxy_to_abs(topk_xyxy, H=H, W=W).detach().cpu().numpy()

    return {
        "topk_boxes_abs": topk_xyxy_abs,
        "topk_scores": topk_scores.detach().cpu().numpy(),
    }


# =========================================================
# SAM3 helpers
# =========================================================
class Sam3BoxSegmenter:
    def __init__(self, ckpt: str, mode: str = "tracker", device: str = "cuda"):
        self.device = device
        self.mode = mode

        if mode == "tracker":
            try:
                Sam3TrackerModel, Sam3TrackerProcessor = import_sam3tracker()
                self.processor = Sam3TrackerProcessor.from_pretrained(ckpt)
                self.model = Sam3TrackerModel.from_pretrained(ckpt).to(device)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Sam3TrackerModel/Sam3TrackerProcessor: {e}")

        elif mode == "model":
            self.processor = Sam3Processor.from_pretrained(ckpt)
            self.model = Sam3Model.from_pretrained(ckpt).to(device)

        else:
            raise ValueError(f"Unknown sam3 mode: {mode}")

        self.model.eval()

    @torch.no_grad()
    def predict_masks(
        self,
        image_pil: Image.Image,
        boxes_xyxy_abs: np.ndarray,
        text_prompt: Optional[str] = None,
        pred_thr: float = 0.5,
    ) -> np.ndarray:
        W, H = image_pil.size
        boxes_xyxy_abs = np.asarray(boxes_xyxy_abs, dtype=np.float32).reshape(-1, 4)
        if len(boxes_xyxy_abs) == 0:
            return np.zeros((0, H, W), dtype=np.uint8)

        clamped_boxes = [clamp_box_xyxy(box, H=H, W=W) for box in boxes_xyxy_abs]
        input_boxes = [[list(map(int, box.tolist())) for box in clamped_boxes]]

        if self.mode == "tracker":
            inputs = self.processor(
                images=image_pil,
                input_boxes=input_boxes,
                return_tensors="pt",
            )
        else:
            processor_kwargs = {
                "images": image_pil,
                "input_boxes": input_boxes,
                "input_boxes_labels": [[1] * len(clamped_boxes)],
                "return_tensors": "pt",
            }
            if text_prompt is not None and str(text_prompt).strip():
                processor_kwargs["text"] = str(text_prompt)
            inputs = self.processor(
                **processor_kwargs,
            )

        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        outputs = self.model(**inputs)

        pm = getattr(outputs, "pred_masks", None)
        if pm is None:
            raise RuntimeError("SAM3 output has no pred_masks.")

        if not torch.is_tensor(pm):
            pm = torch.tensor(pm, device=self.device)
        pm = pm.float()

        use_text_guided_masks = (
            self.mode == "model"
            and text_prompt is not None
            and str(text_prompt).strip() != ""
        )
        if use_text_guided_masks:
            pred_logits = getattr(outputs, "pred_logits", None)
            if pred_logits is None:
                raise RuntimeError("SAM3 model output has no pred_logits for text-guided mask selection.")
            if not torch.is_tensor(pred_logits):
                pred_logits = torch.tensor(pred_logits, device=self.device)
            scores = pred_logits.float().sigmoid()

            presence_logits = getattr(outputs, "presence_logits", None)
            if presence_logits is not None:
                if not torch.is_tensor(presence_logits):
                    presence_logits = torch.tensor(presence_logits, device=self.device)
                scores = scores * presence_logits.float().sigmoid()
            scores = scores[0]

            if pm.ndim == 4:
                pred_masks = pm[0]
            elif pm.ndim == 3:
                pred_masks = pm
            else:
                raise RuntimeError(
                    f"Unexpected pred_masks shape for text-guided model mode: {tuple(pm.shape)}"
                )

            k = min(int(len(clamped_boxes)), int(scores.shape[0]))
            _, topk_idx = torch.topk(scores, k=k, dim=0)
            logits = pred_masks[topk_idx]
        else:
            scores = getattr(outputs, "iou_scores", None)
            if scores is not None:
                if not torch.is_tensor(scores):
                    scores = torch.tensor(scores, device=self.device)
                scores = scores.float()

            if pm.ndim == 5:
                num_boxes = pm.shape[1]
                if scores is not None:
                    best_idx = scores[0].argmax(dim=-1)
                else:
                    best_idx = torch.zeros(num_boxes, dtype=torch.long, device=pm.device)
                logits = pm[0, torch.arange(num_boxes, device=pm.device), best_idx]
            elif pm.ndim == 4:
                logits = pm[0]
            elif pm.ndim == 3:
                logits = pm
            else:
                raise RuntimeError(f"Unexpected pred_masks shape: {tuple(pm.shape)}")

        if logits.ndim == 2:
            logits = logits.unsqueeze(0)

        prob = torch.sigmoid(logits[:, None, ...])
        prob_up = F.interpolate(
            prob,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[:, 0]

        pred01 = (prob_up >= pred_thr).detach().cpu().numpy().astype(np.uint8)
        return pred01

    @torch.no_grad()
    def predict_mask(
        self,
        image_pil: Image.Image,
        box_xyxy_abs: np.ndarray,
        text_prompt: Optional[str] = None,
        pred_thr: float = 0.5,
    ) -> np.ndarray:
        pred01 = self.predict_masks(
            image_pil=image_pil,
            boxes_xyxy_abs=np.asarray(box_xyxy_abs, dtype=np.float32)[None, :],
            text_prompt=text_prompt,
            pred_thr=pred_thr,
        )
        return pred01[0]

    @torch.no_grad()
    def predict_union_mask(
        self,
        image_pil: Image.Image,
        boxes_xyxy_abs: np.ndarray,
        text_prompt: Optional[str] = None,
        pred_thr: float = 0.5,
    ) -> np.ndarray:
        W, H = image_pil.size
        if len(boxes_xyxy_abs) == 0:
            return np.zeros((H, W), dtype=np.uint8)

        pred_masks = self.predict_masks(
            image_pil=image_pil,
            boxes_xyxy_abs=np.asarray(boxes_xyxy_abs, dtype=np.float32),
            text_prompt=text_prompt,
            pred_thr=pred_thr,
        )
        return pred_masks.max(axis=0)


# =========================================================
# metric meters
# =========================================================
class MetricMeter:
    def __init__(self):
        self.n = 0
        self.det_iou_top1_sum = 0.0
        self.det_iou_oracle_sum = 0.0
        self.det_r50 = 0
        self.det_r75 = 0

        self.seg_top1_dice_sum = 0.0
        self.seg_top1_iou_sum = 0.0
        self.pred_empty_top1 = 0

        self.seg_topk_union_dice_sum = 0.0
        self.seg_topk_union_iou_sum = 0.0
        self.pred_empty_topk_union = 0

    def update(
        self,
        det_iou_top1: float,
        det_iou_oracle: float,
        seg_top1_dice: float,
        seg_top1_iou: float,
        pred_empty_top1: bool,
        seg_topk_union_dice: float,
        seg_topk_union_iou: float,
        pred_empty_topk_union: bool,
    ):
        self.n += 1
        self.det_iou_top1_sum += det_iou_top1
        self.det_iou_oracle_sum += det_iou_oracle
        self.det_r50 += int(det_iou_top1 >= 0.5)
        self.det_r75 += int(det_iou_top1 >= 0.75)

        self.seg_top1_dice_sum += seg_top1_dice
        self.seg_top1_iou_sum += seg_top1_iou
        self.pred_empty_top1 += int(pred_empty_top1)

        self.seg_topk_union_dice_sum += seg_topk_union_dice
        self.seg_topk_union_iou_sum += seg_topk_union_iou
        self.pred_empty_topk_union += int(pred_empty_topk_union)

    def summary(self):
        if self.n == 0:
            return {
                "N": 0,
                "det_mIoU_top1": 0.0,
                "det_mIoU_oracle_topk": 0.0,
                "det_Recall@0.5_top1": 0.0,
                "det_Recall@0.75_top1": 0.0,
                "seg_top1_mDice": 0.0,
                "seg_top1_mIoU": 0.0,
                "predEmpty_top1": 0,
                "seg_topk_union_mDice": 0.0,
                "seg_topk_union_mIoU": 0.0,
                "predEmpty_topk_union": 0,
                "seg_mDice": 0.0,
                "seg_mIoU": 0.0,
                "predEmpty": 0,
            }

        return {
            "N": self.n,
            "det_mIoU_top1": self.det_iou_top1_sum / self.n,
            "det_mIoU_oracle_topk": self.det_iou_oracle_sum / self.n,
            "det_Recall@0.5_top1": self.det_r50 / self.n,
            "det_Recall@0.75_top1": self.det_r75 / self.n,
            "seg_top1_mDice": self.seg_top1_dice_sum / self.n,
            "seg_top1_mIoU": self.seg_top1_iou_sum / self.n,
            "predEmpty_top1": self.pred_empty_top1,
            "seg_topk_union_mDice": self.seg_topk_union_dice_sum / self.n,
            "seg_topk_union_mIoU": self.seg_topk_union_iou_sum / self.n,
            "predEmpty_topk_union": self.pred_empty_topk_union,
            "seg_mDice": self.seg_top1_dice_sum / self.n,
            "seg_mIoU": self.seg_top1_iou_sum / self.n,
            "predEmpty": self.pred_empty_top1,
        }


def build_macro_total_from_class_summary(
    class_summary: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    if len(class_summary) == 0:
        return MetricMeter().summary()

    values = list(class_summary.values())
    metric_keys = [
        "det_mIoU_top1",
        "det_mIoU_oracle_topk",
        "det_Recall@0.5_top1",
        "det_Recall@0.75_top1",
        "seg_top1_mDice",
        "seg_top1_mIoU",
        "seg_topk_union_mDice",
        "seg_topk_union_mIoU",
        "seg_mDice",
        "seg_mIoU",
    ]
    count_keys = [
        "predEmpty_top1",
        "predEmpty_topk_union",
        "predEmpty",
    ]

    out = {
        "N": int(sum(int(v.get("N", 0)) for v in values)),
        "num_classes": int(len(values)),
    }
    for k in metric_keys:
        out[k] = float(sum(float(v[k]) for v in values) / len(values))
    for k in count_keys:
        out[k] = int(sum(int(v.get(k, 0)) for v in values))
    return out


def build_summaries_from_records(
    records: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    total_meter = MetricMeter()
    group_meter = defaultdict(MetricMeter)
    class_meter = defaultdict(MetricMeter)
    class_group_meter = defaultdict(MetricMeter)

    for rec in records:
        total_meter.update(
            det_iou_top1=float(rec["det_iou_top1"]),
            det_iou_oracle=float(rec["det_iou_oracle_topk"]),
            seg_top1_dice=float(rec["seg_dice_top1"]),
            seg_top1_iou=float(rec["seg_iou_top1"]),
            pred_empty_top1=bool(rec["pred_empty_top1"]),
            seg_topk_union_dice=float(rec["seg_dice_topk_union"]),
            seg_topk_union_iou=float(rec["seg_iou_topk_union"]),
            pred_empty_topk_union=bool(rec["pred_empty_topk_union"]),
        )
        group_meter[rec["eval_group"]].update(
            float(rec["det_iou_top1"]),
            float(rec["det_iou_oracle_topk"]),
            float(rec["seg_dice_top1"]),
            float(rec["seg_iou_top1"]),
            bool(rec["pred_empty_top1"]),
            float(rec["seg_dice_topk_union"]),
            float(rec["seg_iou_topk_union"]),
            bool(rec["pred_empty_topk_union"]),
        )
        class_meter[rec["canonical_name"]].update(
            float(rec["det_iou_top1"]),
            float(rec["det_iou_oracle_topk"]),
            float(rec["seg_dice_top1"]),
            float(rec["seg_iou_top1"]),
            bool(rec["pred_empty_top1"]),
            float(rec["seg_dice_topk_union"]),
            float(rec["seg_iou_topk_union"]),
            bool(rec["pred_empty_topk_union"]),
        )
        class_group_meter[f"{rec['eval_group']}::{rec['canonical_name']}"].update(
            float(rec["det_iou_top1"]),
            float(rec["det_iou_oracle_topk"]),
            float(rec["seg_dice_top1"]),
            float(rec["seg_iou_top1"]),
            bool(rec["pred_empty_top1"]),
            float(rec["seg_dice_topk_union"]),
            float(rec["seg_iou_topk_union"]),
            bool(rec["pred_empty_topk_union"]),
        )

    group_summary = {k: v.summary() for k, v in sorted(group_meter.items())}
    class_summary = {k: v.summary() for k, v in sorted(class_meter.items())}
    class_group_summary = {k: v.summary() for k, v in sorted(class_group_meter.items())}
    total_summary = {"ALL": build_macro_total_from_class_summary(class_summary)}
    return total_summary, group_summary, class_summary, class_group_summary


def print_summary_table(title: str, summary_dict: Dict[str, Dict[str, float]]):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    for name, s in summary_dict.items():
        print(
            f"{name:<32s} "
            f"N={s['N']:<5d} "
            f"det_mIoU={s['det_mIoU_top1']:.4f} "
            f"det_R50={s['det_Recall@0.5_top1']:.4f} "
            f"det_R75={s['det_Recall@0.75_top1']:.4f} "
            f"det_oracle@topk={s['det_mIoU_oracle_topk']:.4f} | "
            f"seg_top1_mDice={s['seg_top1_mDice']:.4f} "
            f"seg_top1_mIoU={s['seg_top1_mIoU']:.4f} "
            f"predEmpty_top1={s['predEmpty_top1']} | "
            f"seg_topk_union_mDice={s['seg_topk_union_mDice']:.4f} "
            f"seg_topk_union_mIoU={s['seg_topk_union_mIoU']:.4f} "
            f"predEmpty_topk_union={s['predEmpty_topk_union']}"
        )


# =========================================================
# main
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--ann_path", default="data/flare22/ann/odvg_test.json")
    ap.add_argument("--image_root", default="data/flare22")
    ap.add_argument("--prompt_bank", default="data/flare22/prompt_bank.json")
    ap.add_argument("--masks_root", default="data/flare22/masks")

    # model ckpts
    ap.add_argument("--det_ckpt", default="runs_2/best.pt", help="trained detector checkpoint, e.g. best.pt")
    ap.add_argument("--sam3_ckpt", default="/data3/users/zhaojun/project/sam3")

    # eval modes
    ap.add_argument("--eval_modes", default="canonical", help="comma-separated from {canonical,heldout}")
    ap.add_argument("--eval_use_template", type=int, default=1)
    ap.add_argument("--sam_mode", choices=["tracker", "model"], default="tracker")
    ap.add_argument("--sam3_use_text_prompt", type=int, default=0, help="when 1, pass the current text prompt into SAM3 together with boxes; requires --sam_mode model")
    ap.add_argument("--pred_thr", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=5)

    # detector cfg (must match train)
    ap.add_argument("--image_backbone_name", choices=["sam3_fpn"], default="sam3_fpn")
    ap.add_argument("--use_medical_lexicon_adapter", type=int, default=0)
    ap.add_argument("--adapter_bottleneck_dim", type=int, default=256)
    ap.add_argument("--adapter_phrase_out_dim", type=int, default=512)
    ap.add_argument("--adapter_dropout", type=float, default=0.1)
    ap.add_argument("--adapter_gate_init", type=float, default=0.1)
    ap.add_argument("--max_length", type=int, default=32)

    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--num_queries", type=int, default=100)
    ap.add_argument("--num_decoder_layers", type=int, default=6)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_points", type=int, default=4)
    ap.add_argument("--ffn_dim", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--fusion_num_layers", type=int, default=0)
    ap.add_argument("--fusion_top_levels", type=int, default=1)
    ap.add_argument("--fusion_use_text_bias", type=int, default=0)
    ap.add_argument("--fusion_gate_init", type=float, default=0.1)
    ap.add_argument("--use_presence_branch", type=int, default=1)

    # misc
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max_n", type=int, default=-1, help="debug only")
    ap.add_argument("--save_dir", default="runs_2/canonical_use_template")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--use_disk_image_feature_cache", type=int, default=0)
    ap.add_argument("--disk_image_feature_cache_dir", default="./image_feature_cache")

    args = ap.parse_args()
    is_distributed, rank, local_rank, world_size, args.device = init_distributed(args.device)
    set_seed(args.seed + rank)
    ensure_dir(args.save_dir)

    if bool(args.use_disk_image_feature_cache):
        args.disk_image_feature_cache_dir = resolve_feature_cache_dir(
            explicit_dir=args.disk_image_feature_cache_dir,
            base_dir=os.path.dirname(os.path.abspath(args.det_ckpt)),
        )
        ensure_dir(args.disk_image_feature_cache_dir)

    eval_modes = [x.strip() for x in args.eval_modes.split(",") if x.strip()]
    if is_main_process(rank):
        print("[INFO] eval_modes =", eval_modes)
        print("[INFO] device     =", args.device)
        print("[INFO] sam3 text  =", bool(args.sam3_use_text_prompt))
        if is_distributed:
            print(f"[INFO] distributed = True | world_size={world_size}")

    if bool(args.sam3_use_text_prompt) and args.sam_mode != "model":
        raise ValueError("--sam3_use_text_prompt=1 requires --sam_mode model because tracker mode does not support text inputs.")

    # dataset
    dataset = build_eval_dataset(args)
    if args.max_n > 0:
        total_indices = list(range(min(args.max_n, len(dataset))))
    else:
        total_indices = list(range(len(dataset)))
    local_indices = total_indices[rank::world_size]

    if is_main_process(rank):
        print(f"[INFO] dataset size = {len(dataset)}")
        print(f"[INFO] eval count   = {len(total_indices)}")
        if is_distributed:
            print(f"[INFO] local eval count on rank0 = {len(local_indices)}")

    state_dict = load_checkpoint_state_dict(args.det_ckpt)
    apply_inferred_decoder_cfg(args, state_dict, verbose=is_main_process(rank))

    # detector
    det_cfg = build_detector_cfg(args)
    detector = build_detector(det_cfg).to(args.device)
    load_checkpoint_to_model(detector, args.det_ckpt, state_dict=state_dict, verbose=is_main_process(rank))
    detector.eval()

    # sam3
    sam3_segmenter = Sam3BoxSegmenter(
        ckpt=args.sam3_ckpt,
        mode=args.sam_mode,
        device=args.device,
    )

    records = []
    num_skipped_no_gtmask = 0
    total_cache_hits = 0
    total_cache_misses = 0

    pbar = tqdm(total=len(local_indices), desc="Testing", ncols=120) if is_main_process(rank) else None

    for idx in local_indices:
        sample = dataset[idx]
        tasks = collect_eval_tasks(sample, eval_modes=eval_modes)

        if len(tasks) == 0:
            if pbar is not None:
                pbar.update(1)
            continue

        image_tensor = sample["image"]
        image_pil = tensor_to_pil(image_tensor)
        H, W = image_tensor.shape[-2], image_tensor.shape[-1]
        cached_image_feats = None

        if bool(args.use_disk_image_feature_cache):
            cached_image_feats, cache_stats = get_batched_cached_image_features(
                model=detector,
                image_tensors=[image_tensor],
                image_paths=[sample["image_path"]],
                device=args.device,
                cache_dir=args.disk_image_feature_cache_dir,
                enable_disk_cache=True,
            )
            total_cache_hits += int(cache_stats["cache_hits"])
            total_cache_misses += int(cache_stats["cache_misses"])

        mask_path = resolve_mask_path(sample, masks_root=args.masks_root)
        label_map_u8 = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        if label_map_u8.shape[0] != H or label_map_u8.shape[1] != W:
            label_map_u8 = np.array(
                Image.fromarray(label_map_u8).resize((W, H), resample=Image.NEAREST)
            )

        for task in tasks:
            eval_group = task.get("eval_group", "unknown")
            canonical_name = task.get("canonical_name", "unknown")
            prompt = task.get("prompt", "")

            target_boxes = task.get("target_boxes", [])
            if len(target_boxes) == 0:
                continue

            gt_box = union_boxes_xyxy(target_boxes)
            gt_box = clamp_box_xyxy(gt_box, H=H, W=W)

            det_out = run_detector_one(
                detector=detector,
                image_tensor=image_tensor,
                prompt=prompt,
                topk=args.topk,
                device=args.device,
                amp=bool(args.amp),
                precomputed_image_feats=cached_image_feats,
            )

            topk_boxes_abs = det_out["topk_boxes_abs"]
            topk_scores = det_out["topk_scores"]

            pred_box_top1 = clamp_box_xyxy(topk_boxes_abs[0], H=H, W=W)
            det_iou_top1 = iou_xyxy(pred_box_top1, gt_box)
            det_iou_oracle = max(iou_xyxy(clamp_box_xyxy(b, H, W), gt_box) for b in topk_boxes_abs)

            try:
                gt_mask01 = build_gt_binary_mask(
                    sample=sample,
                    task=task,
                    label_map_u8=label_map_u8,
                )
            except Exception as e:
                num_skipped_no_gtmask += 1
                print(f"[WARN] skip no GT mask | idx={idx} case={sample.get('case_id')} task={canonical_name} err={e}")
                continue

            pred_masks_topk = sam3_segmenter.predict_masks(
                image_pil=image_pil,
                boxes_xyxy_abs=topk_boxes_abs,
                text_prompt=(prompt if bool(args.sam3_use_text_prompt) else None),
                pred_thr=args.pred_thr,
            )
            pred_mask_top1 = pred_masks_topk[0]
            pred_mask_topk_union = pred_masks_topk.max(axis=0)

            seg_dice_top1, seg_iou_top1 = dice_iou_mask(pred_mask_top1, gt_mask01)
            seg_dice_topk_union, seg_iou_topk_union = dice_iou_mask(pred_mask_topk_union, gt_mask01)
            pred_empty_top1 = bool(pred_mask_top1.sum() == 0)
            pred_empty_topk_union = bool(pred_mask_topk_union.sum() == 0)

            records.append({
                "idx": idx,
                "case_id": sample.get("case_id", ""),
                "slice_z": sample.get("slice_z", -1),
                "image_path": sample.get("image_path", ""),
                "mask_path": mask_path,
                "eval_group": eval_group,
                "canonical_name": canonical_name,
                "prompt": prompt,
                "det_score_top1": float(topk_scores[0]),
                "det_scores_top5": json.dumps([float(x) for x in topk_scores[:5].tolist()], ensure_ascii=False),
                "det_iou_top1": float(det_iou_top1),
                "det_iou_oracle_topk": float(det_iou_oracle),
                "seg_dice_top1": float(seg_dice_top1),
                "seg_iou_top1": float(seg_iou_top1),
                "pred_empty_top1": int(pred_empty_top1),
                "seg_dice_topk_union": float(seg_dice_topk_union),
                "seg_iou_topk_union": float(seg_iou_topk_union),
                "pred_empty_topk_union": int(pred_empty_topk_union),
                "gt_box_xyxy": gt_box.tolist(),
                "pred_box_top1_xyxy": pred_box_top1.tolist(),
                "gt_pixels": int(gt_mask01.sum()),
                "pred_pixels_top1": int(pred_mask_top1.sum()),
                "pred_pixels_topk_union": int(pred_mask_topk_union.sum()),
            })

        if pbar is not None and bool(args.use_disk_image_feature_cache):
            total_cache = total_cache_hits + total_cache_misses
            pbar.set_postfix({"cache": f"{total_cache_hits}/{max(total_cache, 1)}"})
        if pbar is not None:
            local_total_summary, _, _, _ = build_summaries_from_records(records)
            local_total = local_total_summary["ALL"]
            pbar.update(1)
            pbar.set_postfix(
                totalN=local_total["N"],
                det_mIoU=f"{local_total['det_mIoU_top1']:.4f}",
                seg1=f"{local_total['seg_top1_mDice']:.4f}",
                segK=f"{local_total['seg_topk_union_mDice']:.4f}",
            )

    if pbar is not None:
        pbar.close()

    local_payload = {
        "records": records,
        "num_skipped_no_gtmask": num_skipped_no_gtmask,
        "cache_hits": total_cache_hits,
        "cache_misses": total_cache_misses,
    }
    gathered_payloads = [local_payload]
    if is_distributed:
        gathered_payloads = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_payloads, local_payload)

    if is_main_process(rank):
        all_records = []
        total_skipped_no_gtmask = 0
        all_cache_hits = 0
        all_cache_misses = 0
        for payload in gathered_payloads:
            all_records.extend(payload["records"])
            total_skipped_no_gtmask += int(payload["num_skipped_no_gtmask"])
            all_cache_hits += int(payload["cache_hits"])
            all_cache_misses += int(payload["cache_misses"])

        all_records.sort(key=lambda x: (x["idx"], x["eval_group"], x["canonical_name"], x["prompt"]))
        total_summary, group_summary, class_summary, class_group_summary = build_summaries_from_records(all_records)

        print_summary_table("TOTAL", total_summary)
        print_summary_table("BY EVAL GROUP", group_summary)
        print_summary_table("BY CLASS", class_summary)
        print_summary_table("BY EVAL GROUP + CLASS", class_group_summary)

        print("\n[INFO] skipped no GT mask             =", total_skipped_no_gtmask)
        print("[INFO] total records                  =", len(all_records))
        if bool(args.use_disk_image_feature_cache):
            print("[INFO] image feature cache hits       =", all_cache_hits)
            print("[INFO] image feature cache misses     =", all_cache_misses)

        with open(os.path.join(args.save_dir, "summary_total.json"), "w", encoding="utf-8") as f:
            json.dump(total_summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.save_dir, "summary_by_group.json"), "w", encoding="utf-8") as f:
            json.dump(group_summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.save_dir, "summary_by_class.json"), "w", encoding="utf-8") as f:
            json.dump(class_summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.save_dir, "summary_by_group_class.json"), "w", encoding="utf-8") as f:
            json.dump(class_group_summary, f, ensure_ascii=False, indent=2)

        csv_path = os.path.join(args.save_dir, "records.csv")
        if len(all_records) > 0:
            fieldnames = list(all_records[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_records)

        print(f"[INFO] saved results to: {args.save_dir}")

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

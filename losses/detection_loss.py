import torch
import torch.nn as nn
import torch.nn.functional as F

from models.detector.matcher import HungarianMatcher
from models.detector.matcher import box_cxcywh_to_xyxy
from models.detector.matcher import generalized_box_iou


# =========================================================
# Small box utils
# =========================================================
def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    boxes: [..., 4] in xyxy
    returns: [..., 4] in cxcywh
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    return torch.stack([cx, cy, w, h], dim=-1)


def normalize_xyxy_abs(boxes_xyxy_abs: torch.Tensor, image_size) -> torch.Tensor:
    """
    boxes_xyxy_abs: [N,4] absolute pixel xyxy
    image_size: (H, W)
    returns normalized xyxy in [0,1]
    """
    h, w = image_size
    out = boxes_xyxy_abs.clone().float()
    out[:, 0] /= float(w)
    out[:, 2] /= float(w)
    out[:, 1] /= float(h)
    out[:, 3] /= float(h)
    return out.clamp(0.0, 1.0)


def box_iou_diag(boxes1_xyxy: torch.Tensor, boxes2_xyxy: torch.Tensor) -> torch.Tensor:
    """
    Diagonal IoU between aligned box pairs.

    boxes1_xyxy: [N,4]
    boxes2_xyxy: [N,4]
    return:      [N]
    """
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


# =========================================================
# Criterion
# =========================================================
class DetectionCriterion(nn.Module):
    """
    OVSAM stage-1 detection loss.

    Supervision used here:
    1) box L1 loss
    2) GIoU loss
    3) prompt-conditioned class BCE
    4) prompt-level presence BCE
    5) auxiliary losses on intermediate decoder outputs

    Notes:
    - Later you can add refine loss outside this criterion:
          total_loss = det_loss + lambda_ref * refine_loss
    """

    def __init__(
        self,
        matcher: HungarianMatcher,
        weight_dict: dict,
        target_box_format: str = "xyxy_abs",
        use_aux_loss: bool = True,
        aux_loss_weight: float = 1.0,
        class_pos_weight: float = 1.0,
        presence_pos_weight: float = 1.0,
        use_presence_branch: bool = True,
    ):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.target_box_format = target_box_format
        self.use_aux_loss = use_aux_loss
        self.aux_loss_weight = aux_loss_weight

        self.class_pos_weight = class_pos_weight
        self.presence_pos_weight = presence_pos_weight
        self.use_presence_branch = bool(use_presence_branch)

    # -----------------------------------------------------
    # helpers
    # -----------------------------------------------------
    def _normalize_target_boxes(
        self,
        target_boxes: torch.Tensor,
        image_size,
    ):
        """
        Convert targets into normalized xyxy and cxcywh.
        """
        if self.target_box_format == "xyxy_abs":
            tgt_xyxy_norm = normalize_xyxy_abs(target_boxes, image_size=image_size)
            tgt_cxcywh_norm = box_xyxy_to_cxcywh(tgt_xyxy_norm)
        elif self.target_box_format == "xyxy_norm":
            tgt_xyxy_norm = target_boxes.float().clamp(0.0, 1.0)
            tgt_cxcywh_norm = box_xyxy_to_cxcywh(tgt_xyxy_norm)
        elif self.target_box_format == "cxcywh_norm":
            tgt_cxcywh_norm = target_boxes.float().clamp(0.0, 1.0)
            tgt_xyxy_norm = box_cxcywh_to_xyxy(tgt_cxcywh_norm).clamp(0.0, 1.0)
        else:
            raise ValueError(f"Unknown target_box_format: {self.target_box_format}")

        return tgt_xyxy_norm, tgt_cxcywh_norm

    def _get_num_targets(self, targets: list, device):
        """
        Total number of GT boxes across batch, for normalization.
        """
        num = 0
        for t in targets:
            boxes = t["boxes"]
            if torch.is_tensor(boxes):
                num += boxes.shape[0]
            else:
                num += len(boxes)
        num = max(num, 1)
        return torch.as_tensor(float(num), device=device)

    def _build_binary_targets_from_indices(
        self,
        logits: torch.Tensor,          # [B,Q]
        indices: list,
        positive_value: float = 1.0,
        negative_value: float = 0.0,
    ):
        """
        Build BCE targets for query-level supervision:
            matched queries -> 1
            unmatched queries -> 0
        """
        B, Q = logits.shape
        target = torch.full_like(logits, fill_value=negative_value)

        for b, (src_idx, _) in enumerate(indices):
            if len(src_idx) > 0:
                target[b, src_idx] = positive_value

        return target

    def _loss_class(
        self,
        outputs: dict,
        indices: list,
    ):
        """
        BCE loss over all queries:
          matched = 1, unmatched = 0
        """
        logits = outputs["pred_logits"]  # [B,Q]
        targets = self._build_binary_targets_from_indices(logits, indices)

        pos_weight = torch.tensor(
            self.class_pos_weight,
            device=logits.device,
            dtype=logits.dtype,
        )

        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="mean",
        )
        return loss

    def _loss_presence(
        self,
        outputs: dict,
        targets: list,
    ):
        """
        Prompt-level presence BCE:
          any GT box in this prompt task -> 1
          empty / negative prompt task   -> 0
        """
        if not self.use_presence_branch:
            return outputs["pred_logits"].sum() * 0.0

        logits = outputs["presence_logits"]
        if logits.dim() > 1:
            logits = logits.squeeze(-1)

        presence_targets = []
        for tgt in targets:
            boxes = tgt["boxes"]
            num_boxes = boxes.shape[0] if torch.is_tensor(boxes) else len(boxes)
            presence_targets.append(float(num_boxes > 0))
        presence_targets = torch.as_tensor(
            presence_targets,
            device=logits.device,
            dtype=logits.dtype,
        )

        pos_weight = torch.tensor(
            self.presence_pos_weight,
            device=logits.device,
            dtype=logits.dtype,
        )

        loss = F.binary_cross_entropy_with_logits(
            logits,
            presence_targets,
            pos_weight=pos_weight,
            reduction="mean",
        )
        return loss

    def _loss_boxes(
        self,
        outputs: dict,
        targets: list,
        indices: list,
        num_targets: torch.Tensor,
    ):
        """
        Compute matched box losses:
          - L1 on normalized cxcywh
          - GIoU on normalized xyxy
        """
        pred_boxes = outputs["pred_boxes"]                 # [B,Q,4] cxcywh norm
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)  # [B,Q,4] xyxy norm

        loss_bbox = pred_boxes.sum() * 0.0
        loss_giou = pred_boxes.sum() * 0.0

        total_matched = 0

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue

            tgt_boxes = targets[b]["boxes"]
            if not torch.is_tensor(tgt_boxes):
                tgt_boxes = torch.as_tensor(
                    tgt_boxes,
                    dtype=pred_boxes.dtype,
                    device=pred_boxes.device,
                )
            else:
                tgt_boxes = tgt_boxes.to(pred_boxes.device).float()

            image_size = targets[b].get("image_size", None)
            tgt_xyxy_norm, tgt_cxcywh_norm = self._normalize_target_boxes(
                tgt_boxes,
                image_size=image_size,
            )

            src_boxes = pred_boxes[b, src_idx]                   # [M,4]
            src_boxes_xyxy = pred_boxes_xyxy[b, src_idx]         # [M,4]

            tgt_boxes_cxcywh = tgt_cxcywh_norm[tgt_idx]          # [M,4]
            tgt_boxes_xyxy = tgt_xyxy_norm[tgt_idx]              # [M,4]

            # L1
            loss_bbox = loss_bbox + F.l1_loss(
                src_boxes,
                tgt_boxes_cxcywh,
                reduction="sum",
            )

            # GIoU
            giou = generalized_box_iou(src_boxes_xyxy, tgt_boxes_xyxy)  # [M,M]
            loss_giou = loss_giou + (1.0 - torch.diag(giou)).sum()

            total_matched += len(src_idx)

        loss_bbox = loss_bbox / num_targets
        loss_giou = loss_giou / num_targets

        return loss_bbox, loss_giou, total_matched

    def _compute_single_layer_losses(
        self,
        outputs: dict,
        targets: list,
        indices: list = None,
    ):
        """
        Compute losses for one decoder output (either final or one aux layer).
        """
        if indices is None:
            indices = self.matcher(outputs, targets)

        num_targets = self._get_num_targets(targets, device=outputs["pred_boxes"].device)

        # classification / prompt-presence BCEs
        loss_class = self._loss_class(outputs, indices)
        loss_presence = self._loss_presence(outputs, targets)

        # box losses on matched queries only
        loss_bbox, loss_giou, total_matched = self._loss_boxes(
            outputs=outputs,
            targets=targets,
            indices=indices,
            num_targets=num_targets,
        )

        loss_dict = {
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_class": loss_class,
            "loss_presence": loss_presence,
            "num_targets": num_targets.detach(),
            "num_matched": torch.as_tensor(float(total_matched), device=num_targets.device),
        }

        return loss_dict, indices

    def _weighted_sum(self, loss_dict: dict):
        """
        Compute weighted total detection loss from current loss dict.
        """
        total = 0.0
        for k, v in loss_dict.items():
            if k in self.weight_dict:
                total = total + self.weight_dict[k] * v
        return total

    # -----------------------------------------------------
    # forward
    # -----------------------------------------------------
    def forward(
        self,
        outputs: dict,
        targets: list,
    ):
        """
        Args:
            outputs:
                detector outputs dict from OVSAMDetector
            targets:
                list of dicts, each like
                {
                    "boxes": Tensor [N,4],         # default absolute pixel xyxy
                    "image_size": (H, W)
                }

        Returns:
            total_loss: scalar tensor
            loss_dict: dict of all losses
            indices: matcher assignments for final output
        """
        # -------------------------------------------------
        # 1) final decoder output
        # -------------------------------------------------
        main_outputs = {
            "pred_boxes": outputs["pred_boxes"],
            "pred_logits": outputs["pred_logits"],
        }
        if self.use_presence_branch:
            main_outputs["presence_logits"] = outputs.get("presence_logits", None)

        main_loss_dict, indices = self._compute_single_layer_losses(
            outputs=main_outputs,
            targets=targets,
            indices=None,
        )

        # weighted main loss
        total_loss = self._weighted_sum(main_loss_dict)

        # -------------------------------------------------
        # 2) auxiliary decoder outputs
        # -------------------------------------------------
        aux_loss_dict = {}
        if self.use_aux_loss and "aux_outputs" in outputs:
            # aux_outputs contains all decoder layers, and final output duplicates the last layer
            # so we skip the last aux layer here to avoid double-counting.
            aux_outputs = outputs["aux_outputs"][:-1]

            for i, aux_out in enumerate(aux_outputs):
                aux_layer_outputs = {
                    "pred_boxes": aux_out["pred_boxes"],
                    "pred_logits": aux_out["pred_logits"],
                }
                if self.use_presence_branch:
                    aux_layer_outputs["presence_logits"] = aux_out.get("presence_logits", None)

                aux_layer_loss_dict, _ = self._compute_single_layer_losses(
                    outputs=aux_layer_outputs,
                    targets=targets,
                    indices=None,   # recompute matching at each layer (standard DETR-style choice)
                )

                # rename keys for logging
                for k, v in aux_layer_loss_dict.items():
                    if k in ("num_targets", "num_matched"):
                        aux_loss_dict[f"{k}_aux_{i}"] = v
                    else:
                        aux_key = f"{k}_aux_{i}"
                        aux_loss_dict[aux_key] = v
                        if k in self.weight_dict:
                            total_loss = total_loss + self.aux_loss_weight * self.weight_dict[k] * v

        # -------------------------------------------------
        # 3) merge all losses
        # -------------------------------------------------
        loss_dict = {}
        loss_dict.update(main_loss_dict)
        loss_dict.update(aux_loss_dict)
        loss_dict["loss_total"] = total_loss

        return total_loss, loss_dict, indices


# =========================================================
# Builder
# =========================================================
def build_detection_criterion(cfg: dict = None):
    if cfg is None:
        cfg = {}

    matcher = cfg.get("matcher", None)
    if matcher is None:
        matcher = HungarianMatcher(
            cost_bbox=cfg.get("matcher_cost_bbox", 5.0),
            cost_giou=cfg.get("matcher_cost_giou", 2.0),
            cost_class=cfg.get("matcher_cost_class", 1.0),
            target_box_format=cfg.get("target_box_format", "xyxy_abs"),
        )

    weight_dict = {
        "loss_bbox": cfg.get("loss_bbox_weight", 5.0),
        "loss_giou": cfg.get("loss_giou_weight", 2.0),
        "loss_class": cfg.get("loss_class_weight", 1.0),
    }
    if cfg.get("use_presence_branch", True):
        weight_dict["loss_presence"] = cfg.get("loss_presence_weight", 1.0)

    criterion = DetectionCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        target_box_format=cfg.get("target_box_format", "xyxy_abs"),
        use_aux_loss=cfg.get("use_aux_loss", True),
        aux_loss_weight=cfg.get("aux_loss_weight", 1.0),
        class_pos_weight=cfg.get("class_pos_weight", 1.0),
        presence_pos_weight=cfg.get("presence_pos_weight", 1.0),
        use_presence_branch=cfg.get("use_presence_branch", True),
    )
    return criterion


# =========================================================
# minimal self-test
# =========================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, Q = 2, 100
    outputs = {
        "pred_boxes": torch.rand(B, Q, 4, device=device),
        "pred_logits": torch.randn(B, Q, device=device),
        "presence_logits": torch.randn(B, device=device),
        "aux_outputs": [],
    }
    outputs["pred_boxes"][..., 2:] = outputs["pred_boxes"][..., 2:] * 0.5

    targets = [
        {
            "boxes": torch.tensor([[140.0, 178.0, 402.0, 432.0]], device=device),
            "image_size": (512, 512),
        },
        {
            "boxes": torch.tensor([
                [120.0, 80.0, 220.0, 200.0],
                [260.0, 180.0, 360.0, 300.0],
            ], device=device),
            "image_size": (512, 512),
        },
    ]

    criterion = build_detection_criterion({
        "target_box_format": "xyxy_abs",
        "use_aux_loss": False,
    }).to(device)

    total_loss, loss_dict, indices = criterion(outputs, targets)

    print("total_loss:", float(total_loss))
    for k, v in loss_dict.items():
        if torch.is_tensor(v) and v.numel() == 1:
            print(f"{k}: {float(v)}")
        else:
            print(f"{k}: {v}")

    for b, (src_idx, tgt_idx) in enumerate(indices):
        print(f"[batch {b}] matched pairs: {len(src_idx)}")
        print("  src_idx:", src_idx.tolist())
        print("  tgt_idx:", tgt_idx.tolist())

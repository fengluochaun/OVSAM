import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    return torch.stack([cx, cy, w, h], dim=-1)


def normalize_xyxy_abs(boxes_xyxy_abs: torch.Tensor, image_size) -> torch.Tensor:
    h, w = image_size
    out = boxes_xyxy_abs.clone().float()
    out[:, 0] /= float(w)
    out[:, 2] /= float(w)
    out[:, 1] /= float(h)
    out[:, 3] /= float(h)
    return out.clamp(0.0, 1.0)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / (area + 1e-6)


class HungarianMatcher(nn.Module):
    """
    Matcher for the early OVSAM prompt-conditioned detector.

    Matching uses:
    - bbox L1
    - bbox GIoU
    - prompt-conditioned class score
    """

    def __init__(
        self,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        cost_class: float = 1.0,
        target_box_format: str = "xyxy_abs",
    ):
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_class = cost_class
        self.target_box_format = target_box_format

        if (
            cost_bbox == 0
            and cost_giou == 0
            and cost_class == 0
        ):
            raise ValueError("All matcher costs cannot be zero.")

    def _normalize_targets(self, target_boxes: torch.Tensor, image_size):
        if self.target_box_format == "xyxy_abs":
            if image_size is None:
                raise ValueError("image_size is required when target_box_format='xyxy_abs'")
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

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list):
        """
        Returns:
            indices: list of tuples (src_idx, tgt_idx), length=B
                src_idx: indices of selected predictions
                tgt_idx: indices of corresponding selected targets
        """
        device = outputs["pred_boxes"].device
        device_type = device.type

        # Matcher runs only for assignment. Force float32 to avoid AMP-induced
        # half precision instability in cdist / GIoU cost computation.
        with torch.amp.autocast(device_type=device_type, enabled=False):
            pred_boxes = outputs["pred_boxes"].float()          # [B,Q,4] normalized cxcywh
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)     # [B,Q,4]
            cls_logits = outputs["pred_logits"].float()         # [B,Q]

            bs, num_queries = pred_boxes.shape[:2]
            indices = []

            for b in range(bs):
                tgt = targets[b]
                tgt_boxes = tgt["boxes"]

                if not torch.is_tensor(tgt_boxes):
                    tgt_boxes = torch.as_tensor(
                        tgt_boxes,
                        dtype=pred_boxes.dtype,
                        device=pred_boxes.device,
                    )
                else:
                    tgt_boxes = tgt_boxes.to(pred_boxes.device).float()

                num_tgt = tgt_boxes.shape[0]
                if num_tgt == 0:
                    indices.append((
                        torch.empty(0, dtype=torch.int64, device=pred_boxes.device),
                        torch.empty(0, dtype=torch.int64, device=pred_boxes.device),
                    ))
                    continue

                image_size = tgt.get("image_size", None)
                tgt_xyxy_norm, tgt_cxcywh_norm = self._normalize_targets(
                    tgt_boxes,
                    image_size=image_size,
                )

                cost_bbox = torch.cdist(pred_boxes[b], tgt_cxcywh_norm, p=1)
                cost_giou = -generalized_box_iou(pred_boxes_xyxy[b], tgt_xyxy_norm)
                cost_cls = -cls_logits[b].sigmoid().unsqueeze(1).expand(num_queries, num_tgt)

                C = (
                    self.cost_bbox * cost_bbox
                    + self.cost_giou * cost_giou
                    + self.cost_class * cost_cls
                )

                if not torch.isfinite(C).all():
                    raise RuntimeError(
                        "HungarianMatcher produced invalid cost entries "
                        f"on batch index {b}: "
                        f"pred_boxes_finite={torch.isfinite(pred_boxes[b]).all().item()} "
                        f"pred_logits_finite={torch.isfinite(cls_logits[b]).all().item()} "
                        f"tgt_boxes_finite={torch.isfinite(tgt_boxes).all().item()} "
                        f"cost_bbox_finite={torch.isfinite(cost_bbox).all().item()} "
                        f"cost_giou_finite={torch.isfinite(cost_giou).all().item()} "
                        f"cost_cls_finite={torch.isfinite(cost_cls).all().item()}"
                    )

                C_np = C.detach().cpu().numpy()
                src_idx, tgt_idx = linear_sum_assignment(C_np)

                src_idx = torch.as_tensor(src_idx, dtype=torch.int64, device=pred_boxes.device)
                tgt_idx = torch.as_tensor(tgt_idx, dtype=torch.int64, device=pred_boxes.device)
                indices.append((src_idx, tgt_idx))

        return indices

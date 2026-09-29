from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from models.text.text_backbone import OVSAMTextBackbone
from models.detector.deformable_prompt_decoder import DeformablePromptDecoder
from models.vision.sam3_image_backbone import SAM3FPNImageBackbone


# =========================================================
# Main stage-1 detector
# =========================================================
class OVSAMDetector(nn.Module):
    """
    Full stage-1 detector for OVSAM.

    Pipeline:
        images -> image_backbone -> multi-scale image features
        prompts -> text_backbone -> token_feats + phrase_feat
        (image feats, token_feats, phrase_feat) -> decoder -> top-K boxes + scores
    """
    def __init__(
        self,
        image_backbone: nn.Module,
        text_backbone: OVSAMTextBackbone,
        decoder: DeformablePromptDecoder,
    ):
        super().__init__()
        self.image_backbone = image_backbone
        self.text_backbone = text_backbone
        self.decoder = decoder

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _normalize_prompts(
        self,
        prompts: Union[str, Sequence[str]],
        batch_size: int,
    ) -> List[str]:
        if isinstance(prompts, str):
            return [prompts for _ in range(batch_size)]

        prompts = list(prompts)
        if len(prompts) == 1 and batch_size > 1:
            return prompts * batch_size

        if len(prompts) != batch_size:
            raise ValueError(
                f"Prompt count ({len(prompts)}) does not match batch size ({batch_size})."
            )
        return prompts

    def forward_image(
        self,
        images: torch.Tensor,
        return_image_aux: bool = False,
    ):
        """
        Image backbone forward.

        Some backbones may return:
        - just multi_scale_feats
        - or a dict containing multi_scale_feats + aux data

        This wrapper normalizes the behavior.
        """
        outputs = self.image_backbone(images, return_aux=return_image_aux) \
            if "return_aux" in self.image_backbone.forward.__code__.co_varnames \
            else self.image_backbone(images)

        if isinstance(outputs, dict):
            return outputs
        else:
            if return_image_aux:
                return {"multi_scale_feats": outputs}
            return outputs

    def forward_text(
        self,
        prompts: Union[str, Sequence[str]],
        batch_size: int,
        return_raw_text: bool = True,
        return_text_aux: bool = True,
    ) -> Dict[str, torch.Tensor]:
        prompts = self._normalize_prompts(prompts, batch_size=batch_size)
        text_out = self.text_backbone(
            prompts,
            return_raw=return_raw_text,
            return_aux=return_text_aux,
        )
        return text_out

    def forward_decoder(
        self,
        multi_scale_feats: List[torch.Tensor],
        text_outputs: Dict[str, torch.Tensor],
        topk: Optional[int] = None,
        return_aux: bool = True,
    ) -> Dict[str, torch.Tensor]:
        decoder_out = self.decoder(
            multi_scale_feats=multi_scale_feats,
            token_feats=text_outputs["token_feats"],
            phrase_feat=text_outputs["phrase_feat"],
            text_padding_mask=text_outputs["attention_mask"],
            topk=topk,
            return_aux=return_aux,
        )
        return decoder_out

    def forward(
        self,
        images: torch.Tensor,
        prompts: Union[str, Sequence[str]],
        topk: Optional[int] = None,
        return_image_features: bool = False,
        return_text_outputs: bool = False,
        return_aux: bool = True,
        return_image_aux: bool = False,
        precomputed_text_outputs: Optional[Dict[str, torch.Tensor]] = None,
        precomputed_image_features: Optional[List[torch.Tensor]] = None,
        prompt_image_indices: Optional[torch.Tensor] = None,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        if images.ndim != 4:
            raise ValueError(f"Expected images shape [B,3,H,W], got {tuple(images.shape)}")

        image_batch_size = images.shape[0]

        # -------------------------------------------------
        # 1) image branch
        # -------------------------------------------------
        if precomputed_image_features is None:
            image_branch_out = self.forward_image(images, return_image_aux=return_image_aux)
            if isinstance(image_branch_out, dict):
                multi_scale_feats = image_branch_out["multi_scale_feats"]
            else:
                multi_scale_feats = image_branch_out
                image_branch_out = {"multi_scale_feats": multi_scale_feats}
        else:
            multi_scale_feats = precomputed_image_features
            image_branch_out = {"multi_scale_feats": multi_scale_feats}

        if prompt_image_indices is not None:
            if not torch.is_tensor(prompt_image_indices):
                prompt_image_indices = torch.as_tensor(prompt_image_indices, dtype=torch.long, device=images.device)
            prompt_image_indices = prompt_image_indices.to(device=multi_scale_feats[0].device, dtype=torch.long)
            if prompt_image_indices.ndim != 1:
                raise ValueError(
                    "prompt_image_indices must be a 1D tensor mapping each prompt task to one image."
                )
            if prompt_image_indices.numel() == 0:
                raise ValueError("prompt_image_indices is empty.")
            if int(prompt_image_indices.min().item()) < 0 or int(prompt_image_indices.max().item()) >= image_batch_size:
                raise ValueError(
                    f"prompt_image_indices out of range for image batch size {image_batch_size}."
                )
            multi_scale_feats = [
                feat.index_select(0, prompt_image_indices)
                for feat in multi_scale_feats
            ]
            batch_size = int(prompt_image_indices.numel())
        else:
            batch_size = image_batch_size

        # -------------------------------------------------
        # 2) text branch
        # -------------------------------------------------
        if precomputed_text_outputs is None:
            text_outputs = self.forward_text(
                prompts=prompts,
                batch_size=batch_size,
                return_raw_text=return_text_outputs,
                return_text_aux=return_text_outputs,
            )
        else:
            text_outputs = precomputed_text_outputs

        # -------------------------------------------------
        # 3) decoder
        # -------------------------------------------------
        decoder_outputs = self.forward_decoder(
            multi_scale_feats=multi_scale_feats,
            text_outputs=text_outputs,
            topk=topk,
            return_aux=return_aux,
        )

        out = dict(decoder_outputs)
        if text_outputs.get("offset_mapping", None) is not None:
            out["text_offset_mapping"] = text_outputs["offset_mapping"]

        if return_image_features:
            out["multi_scale_feats"] = multi_scale_feats

        if return_image_aux:
            out["image_outputs"] = image_branch_out

        if return_text_outputs:
            out["text_outputs"] = text_outputs

        return out


# =========================================================
# Config helpers
# =========================================================
def _get_cfg(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    return cfg[key] if key in cfg else default


def build_image_backbone(cfg: Dict[str, Any]) -> nn.Module:
    """
    Build stage-1 image backbone.

    Supported:
        - sam3_fpn
    """
    name = _get_cfg(cfg, "image_backbone_name", "sam3_fpn")

    if name == "sam3_fpn":
        sam3_ckpt = _get_cfg(cfg, "sam3_ckpt", None)
        if sam3_ckpt is None:
            raise ValueError("cfg['sam3_ckpt'] is required for image_backbone_name='sam3_fpn'.")

        return SAM3FPNImageBackbone(
            sam3_ckpt=sam3_ckpt,
            freeze_image_encoder=_get_cfg(cfg, "freeze_image_encoder", True),
            return_last_hidden_state=_get_cfg(cfg, "return_image_last_hidden_state", False),
            return_position_encoding=_get_cfg(cfg, "return_image_position_encoding", False),
        )

    raise NotImplementedError(
        f"Unknown image_backbone_name='{name}'. "
        f"Please implement it in build_image_backbone()."
    )


def build_text_backbone(cfg: Dict[str, Any]) -> OVSAMTextBackbone:
    name = _get_cfg(cfg, "text_backbone_name", "sam3_text")
    if name not in {"sam3_text", "sam3"}:
        raise NotImplementedError(
            f"Unknown text_backbone_name='{name}'. "
            f"Supported: sam3_text."
        )

    sam3_ckpt = _get_cfg(cfg, "sam3_ckpt", None)
    if sam3_ckpt is None:
        raise ValueError("cfg['sam3_ckpt'] is required to build OVSAMTextBackbone.")

    text_backbone = OVSAMTextBackbone(
        sam3_ckpt=sam3_ckpt,
        freeze_text_encoder=_get_cfg(cfg, "freeze_text_encoder", True),
        use_medical_lexicon_adapter=_get_cfg(cfg, "use_medical_lexicon_adapter", True),
        adapter_bottleneck_dim=_get_cfg(cfg, "adapter_bottleneck_dim", 256),
        adapter_phrase_out_dim=_get_cfg(cfg, "adapter_phrase_out_dim", 512),
        adapter_dropout=_get_cfg(cfg, "adapter_dropout", 0.1),
        adapter_gate_init=_get_cfg(cfg, "adapter_gate_init", 0.1),
        max_length=_get_cfg(cfg, "max_length", 32),
    )
    return text_backbone


def build_decoder(cfg: Dict[str, Any], image_backbone: nn.Module) -> DeformablePromptDecoder:
    if not hasattr(image_backbone, "out_channels"):
        raise AttributeError(
            "image_backbone must define `out_channels`, e.g. [256, 256, 256, 256]."
        )

    decoder = DeformablePromptDecoder(
        image_in_channels=image_backbone.out_channels,
        text_token_dim=_get_cfg(cfg, "text_token_dim", 1024),
        phrase_dim=_get_cfg(cfg, "phrase_dim", 512),
        embed_dim=_get_cfg(cfg, "embed_dim", 256),
        num_queries=_get_cfg(cfg, "num_queries", 100),
        num_decoder_layers=_get_cfg(cfg, "num_decoder_layers", 6),
        num_heads=_get_cfg(cfg, "num_heads", 8),
        num_points=_get_cfg(cfg, "num_points", 4),
        ffn_dim=_get_cfg(cfg, "ffn_dim", 1024),
        dropout=_get_cfg(cfg, "dropout", 0.1),
        topk=_get_cfg(cfg, "topk", 10),
        fusion_num_layers=_get_cfg(cfg, "fusion_num_layers", 0),
        fusion_top_levels=_get_cfg(cfg, "fusion_top_levels", 1),
        fusion_use_text_bias=_get_cfg(cfg, "fusion_use_text_bias", False),
        fusion_gate_init=_get_cfg(cfg, "fusion_gate_init", 0.1),
        use_phrase_query_conditioning=_get_cfg(cfg, "use_medical_lexicon_adapter", True),
        use_presence_branch=_get_cfg(cfg, "use_presence_branch", True),
    )
    return decoder


# =========================================================
# Public builder
# =========================================================
def build_detector(cfg: Dict[str, Any]) -> OVSAMDetector:
    image_backbone = build_image_backbone(cfg)
    text_backbone = build_text_backbone(cfg)
    decoder = build_decoder(cfg, image_backbone=image_backbone)

    model = OVSAMDetector(
        image_backbone=image_backbone,
        text_backbone=text_backbone,
        decoder=decoder,
    )
    return model


# =========================================================
# Self-test
# =========================================================
if __name__ == "__main__":
    cfg = {
        "sam3_ckpt": "/data3/users/zhaojun/project/sam3",
        "image_backbone_name": "sam3_fpn",   # <--- now testing real SAM3 image backbone
        "freeze_image_encoder": True,
        "freeze_text_encoder": True,

        "adapter_bottleneck_dim": 256,
        "adapter_phrase_out_dim": 512,
        "max_length": 32,

        "text_token_dim": 1024,
        "phrase_dim": 512,
        "embed_dim": 256,
        "num_queries": 100,
        "num_decoder_layers": 6,
        "num_heads": 8,
        "num_points": 4,
        "ffn_dim": 1024,
        "dropout": 0.1,
        "topk": 10,
        "dn_num_groups": 5,
        "dn_box_noise_scale": 0.4,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_detector(cfg).to(device)

    images = torch.randn(2, 3, 512, 512).to(device)
    prompts = ["a CT image of liver", "a CT image of spleen"]

    with torch.no_grad():
        out = model(
            images=images,
            prompts=prompts,
            topk=10,
            return_image_features=True,
            return_text_outputs=False,
            return_aux=True,
            return_image_aux=True,
        )

    print("pred_boxes         :", out["pred_boxes"].shape)
    print("pred_logits        :", out["pred_logits"].shape)
    print("presence_logits    :", out["presence_logits"].shape)
    print("topk_boxes         :", out["topk_boxes"].shape)
    print("topk_scores        :", out["topk_scores"].shape)

    if "multi_scale_feats" in out:
        for i, feat in enumerate(out["multi_scale_feats"]):
            print(f"multi_scale_feats[{i}]:", feat.shape)

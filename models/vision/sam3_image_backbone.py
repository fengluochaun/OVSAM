import os
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from transformers import Sam3Model, Sam3Processor


class SAM3FPNImageBackbone(nn.Module):
    """
    OVSAM image backbone built from SAM3 vision encoder.

    What it does:
    - Reuse SAM3's vision_encoder directly
    - Extract multi-scale FPN features from:
        vision_outputs.fpn_hidden_states
    - Return a feature pyramid compatible with the stage-1 decoder

    Important design note:
    ----------------------
    This implementation is optimized for the current OVSAM setting:
      - SAM3 image encoder is frozen
      - input images come from dataset as [B, 3, H, W] float tensors in [0,1]
      - preprocessing is delegated to Sam3Processor

    Because we convert tensors to PIL for Sam3Processor, this is NOT intended
    for end-to-end gradient flow into the image encoder.
    That is fine for the current "freeze foundation model" design.
    """

    def __init__(
        self,
        sam3_ckpt: str,
        freeze_image_encoder: bool = True,
        return_last_hidden_state: bool = False,
        return_position_encoding: bool = False,
    ):
        super().__init__()
        self.sam3_ckpt = sam3_ckpt
        self.freeze_image_encoder = freeze_image_encoder
        self.return_last_hidden_state = return_last_hidden_state
        self.return_position_encoding = return_position_encoding

        sam3_model = Sam3Model.from_pretrained(sam3_ckpt)
        self.vision_encoder = sam3_model.vision_encoder
        del sam3_model

        self.processor = Sam3Processor.from_pretrained(sam3_ckpt)

        # From your probe:
        # fpn_hidden_states:
        #   [B, 256, 288, 288]
        #   [B, 256, 144, 144]
        #   [B, 256, 72, 72]
        #   [B, 256, 36, 36]
        self.out_channels = [256, 256, 256, 256]

        if self.freeze_image_encoder:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            self.vision_encoder.eval()

    @property
    def device(self) -> torch.device:
        return next(self.vision_encoder.parameters()).device

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------
    def _tensor_to_pil(self, image: torch.Tensor) -> Image.Image:
        """
        Convert a single image tensor [3,H,W] in [0,1] or [0,255] to PIL RGB.
        """
        if image.ndim != 3:
            raise ValueError(f"Expected single image shape [3,H,W], got {tuple(image.shape)}")

        x = image.detach().cpu().float()

        # if values likely in [0,1], scale to [0,255]
        if x.max() <= 1.5:
            x = x.clamp(0, 1) * 255.0
        else:
            x = x.clamp(0, 255)

        x = x.byte().permute(1, 2, 0).numpy()  # CHW -> HWC
        return Image.fromarray(x, mode="RGB")

    def _prepare_pixel_values(
        self,
        images: Union[torch.Tensor, Sequence[Image.Image]],
    ) -> Dict[str, torch.Tensor]:
        """
        Convert raw images into SAM3 pixel_values via Sam3Processor.

        Supported input:
        - torch.Tensor [B,3,H,W]
        - list/tuple of PIL images

        Returns:
            processed dict with at least:
              pixel_values: [B,3,1008,1008]
              original_sizes: [B,2]
        """
        if torch.is_tensor(images):
            if images.ndim == 3:
                images = images.unsqueeze(0)
            if images.ndim != 4:
                raise ValueError(f"Expected image tensor [B,3,H,W], got {tuple(images.shape)}")

            pil_images = [self._tensor_to_pil(images[i]) for i in range(images.shape[0])]
            processed = self.processor(images=pil_images, return_tensors="pt")

        elif isinstance(images, (list, tuple)):
            if len(images) == 0:
                raise ValueError("Input image list is empty.")
            if not isinstance(images[0], Image.Image):
                raise TypeError("When images is a list/tuple, elements must be PIL.Image.")
            processed = self.processor(images=list(images), return_tensors="pt")

        else:
            raise TypeError(
                "Unsupported images type. Expect torch.Tensor [B,3,H,W] or list of PIL.Image."
            )

        processed = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in processed.items()
        }
        return processed

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(
        self,
        images: Union[torch.Tensor, Sequence[Image.Image]],
        input_is_preprocessed: bool = False,
        return_aux: bool = False,
    ):
        """
        Args:
            images:
                - raw tensor [B,3,H,W] in [0,1] or [0,255]
                - OR already preprocessed pixel_values if input_is_preprocessed=True
            input_is_preprocessed:
                if True, `images` is assumed to be pixel_values [B,3,1008,1008]
            return_aux:
                whether to also return last_hidden_state / position_encoding / original_sizes

        Returns:
            if return_aux=False:
                multi_scale_feats: list of 4 tensors
            else:
                dict {
                    "multi_scale_feats": [...],
                    "pixel_values": ...,
                    "original_sizes": ...,
                    "last_hidden_state": ...,
                    "fpn_position_encoding": ...
                }
        """
        # -------------------------------------------------
        # 1) Prepare pixel_values
        # -------------------------------------------------
        if input_is_preprocessed:
            if not torch.is_tensor(images):
                raise TypeError("When input_is_preprocessed=True, images must be a tensor.")
            pixel_values = images.to(self.device)
            original_sizes = None

            if not self.freeze_image_encoder:
                # this path allows future end-to-end training if you supply
                # differentiable preprocessed pixel_values externally
                pass
        else:
            if not self.freeze_image_encoder:
                raise RuntimeError(
                    "Current SAM3FPNImageBackbone preprocessing uses PIL/processor and is intended "
                    "for frozen image encoder only. "
                    "If you want to train image encoder end-to-end, pass preprocessed pixel_values "
                    "with input_is_preprocessed=True and implement a differentiable preprocessing path."
                )
            processed = self._prepare_pixel_values(images)
            pixel_values = processed["pixel_values"]
            original_sizes = processed.get("original_sizes", None)

        # -------------------------------------------------
        # 2) Vision encoder forward
        # -------------------------------------------------
        if self.freeze_image_encoder:
            with torch.no_grad():
                vision_outputs = self.vision_encoder(
                    pixel_values=pixel_values,
                    return_dict=True,
                )
        else:
            vision_outputs = self.vision_encoder(
                pixel_values=pixel_values,
                return_dict=True,
            )

        # Probe showed these exist:
        # - last_hidden_state: [B, 5184, 1024]
        # - fpn_hidden_states: tuple of 4 tensors, each [B,256,H,W]
        multi_scale_feats = list(vision_outputs.fpn_hidden_states)

        if not return_aux:
            return multi_scale_feats

        out = {
            "multi_scale_feats": multi_scale_feats,
            "pixel_values": pixel_values,
            "original_sizes": original_sizes,
        }

        if self.return_last_hidden_state and hasattr(vision_outputs, "last_hidden_state"):
            out["last_hidden_state"] = vision_outputs.last_hidden_state

        if self.return_position_encoding and hasattr(vision_outputs, "fpn_position_encoding"):
            out["fpn_position_encoding"] = vision_outputs.fpn_position_encoding

        return out

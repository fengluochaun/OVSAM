import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Basic utilities
# =========================================================
def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Numerically stable inverse of sigmoid.
    Used for iterative box refinement:
        new_box = sigmoid(inverse_sigmoid(old_box) + delta)
    """
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


class MLP(nn.Module):
    """
    Generic multi-layer perceptron.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.0
    ):
        super().__init__()
        assert num_layers >= 1
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DotProductScoring(nn.Module):
    """
    Prompt-conditioned query scoring borrowed from SAM3's detector branch.

    It mean-pools valid prompt tokens, projects both prompt and query states into
    a shared subspace, and scores each query with a scaled dot product.
    """
    def __init__(
        self,
        d_model: int,
        d_proj: int,
        clamp_logits: bool = True,
        clamp_max_val: float = 12.0,
    ):
        super().__init__()
        self.prompt_proj = nn.Linear(d_model, d_proj)
        self.query_proj = nn.Linear(d_model, d_proj)
        self.scale = 1.0 / math.sqrt(float(d_proj))
        self.clamp_logits = bool(clamp_logits)
        self.clamp_max_val = float(clamp_max_val)

    def _mean_pool_text(
        self,
        text_tokens: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if text_padding_mask is None:
            return text_tokens.mean(dim=1)

        valid = text_padding_mask.float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        return (text_tokens * valid).sum(dim=1) / denom

    def forward(
        self,
        query_states: torch.Tensor,
        text_tokens: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pooled_prompt = self._mean_pool_text(text_tokens, text_padding_mask)     # [B,D]
        proj_prompt = self.prompt_proj(pooled_prompt)                            # [B,Dp]
        proj_query = self.query_proj(query_states)                               # [B,Q,Dp]

        logits = torch.matmul(proj_query, proj_prompt.unsqueeze(-1)).squeeze(-1)
        logits = logits * self.scale
        if self.clamp_logits:
            logits = logits.clamp(min=-self.clamp_max_val, max=self.clamp_max_val)
        return logits


# =========================================================
# Multi-scale image feature projection
# =========================================================
class MultiScaleFeatureProjector(nn.Module):
    """
    Project backbone multi-scale image features into a unified decoder dimension.

    Input:
        multi_scale_feats = [
            feat_lvl0: [B, C0, H0, W0],
            feat_lvl1: [B, C1, H1, W1],
            ...
        ]

    Output:
        projected_feats = [
            [B, D, H0, W0],
            [B, D, H1, W1],
            ...
        ]

    We also add one learnable level embedding per feature level, which is
    standard in DETR-style multi-scale decoders.
    """
    def __init__(
        self,
        in_channels_list: List[int],
        embed_dim: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_levels = len(in_channels_list)

        self.input_proj = nn.ModuleList()
        for in_ch in in_channels_list:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, embed_dim, kernel_size=1),
                    nn.GroupNorm(32, embed_dim),
                )
            )

        self.level_embeds = nn.Parameter(torch.Tensor(self.num_levels, embed_dim))
        nn.init.normal_(self.level_embeds, std=0.02)

    def forward(self, multi_scale_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(multi_scale_feats) == self.num_levels, \
            f"Expected {self.num_levels} feature levels, got {len(multi_scale_feats)}"

        outs = []
        for lvl, feat in enumerate(multi_scale_feats):
            # Keep feature maps in standard contiguous layout so Conv2d weight
            # grads do not flip to channels-last and trigger DDP bucket warnings.
            feat = feat.contiguous(memory_format=torch.contiguous_format)
            x = self.input_proj[lvl](feat)                    # [B, D, H, W]
            x = x + self.level_embeds[lvl][None, :, None, None]
            outs.append(x)
        return outs


# =========================================================
# Query position encoding from reference boxes
# =========================================================
class ReferenceBoxPositionEncoder(nn.Module):
    """
    Encode normalized reference boxes (cx, cy, w, h) into query positional embeddings.

    Why:
    - DAB-DETR style methods show that using box coordinates as part of the query
      greatly stabilizes iterative refinement.
    - Here we avoid a heavy sine encoder and use a learnable MLP over box coords.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.mlp = MLP(4, embed_dim, embed_dim, num_layers=3, dropout=0.0)

    def forward(self, ref_boxes: torch.Tensor) -> torch.Tensor:
        """
        ref_boxes: [B, Q, 4], normalized cxcywh in [0,1]
        returns:   [B, Q, D]
        """
        return self.mlp(ref_boxes)


# =========================================================
# Shallow image-text fusion before query decoding
# =========================================================
class PooledTextConditioner(nn.Module):
    """
    Add a prompt-conditioned residual bias to selected image feature levels.

    This is intentionally shallow: it mean-pools text tokens, projects them
    per level, and injects the result with a learnable scalar gate.
    """
    def __init__(
        self,
        embed_dim: int,
        num_conditioned_levels: int,
        gate_init: float = 0.1,
    ):
        super().__init__()
        self.level_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
            for _ in range(num_conditioned_levels)
        ])
        self.level_gates = nn.Parameter(torch.full((num_conditioned_levels,), float(gate_init)))

    def _pool_text(
        self,
        text_tokens: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if text_padding_mask is None:
            return text_tokens.mean(dim=1)

        valid = text_padding_mask.float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        return (text_tokens * valid).sum(dim=1) / denom

    def forward(
        self,
        multi_feats: List[torch.Tensor],
        text_tokens: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor],
        level_indices: List[int],
    ) -> List[torch.Tensor]:
        if len(level_indices) == 0:
            return multi_feats

        pooled_text = self._pool_text(text_tokens, text_padding_mask)
        outs = list(multi_feats)
        if len(level_indices) != len(self.level_projs):
            raise ValueError(
                f"Conditioned levels ({len(level_indices)}) do not match "
                f"text conditioner size ({len(self.level_projs)})."
            )
        for proj_idx, lvl in enumerate(level_indices):
            bias = self.level_projs[proj_idx](pooled_text)[:, :, None, None]
            outs[lvl] = outs[lvl] + self.level_gates[proj_idx] * bias
        return outs


class ShallowFusionEncoderLayer(nn.Module):
    """
    Lightweight image<-text fusion block on flattened image tokens.

    The block keeps the text branch frozen and only updates image tokens.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = image_tokens

        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            need_weights=False,
        )
        x = x + self.drop1(attn_out)

        x_norm = self.norm2(x)
        attn_out, _ = self.text_cross_attn(
            x_norm,
            text_tokens,
            text_tokens,
            key_padding_mask=(~text_padding_mask.bool()) if text_padding_mask is not None else None,
            need_weights=False,
        )
        x = x + self.drop2(attn_out)

        x = x + self.ffn(self.norm3(x))
        return x


class ShallowPromptFusionEncoder(nn.Module):
    """
    Prompt-conditioned fusion on the top semantic feature levels only.

    It keeps OVSAM's multi-scale decoder intact while giving the highest-level
    image features a light text-aware refinement before queries are initialized.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        num_levels: int,
        num_layers: int = 1,
        top_levels: int = 1,
        use_text_bias: bool = True,
        text_bias_gate_init: float = 0.1,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.top_levels = max(int(top_levels), 0)
        self.use_text_bias = bool(use_text_bias)
        self.level_indices = self._build_target_levels()

        self.text_conditioner = None
        if self.use_text_bias and len(self.level_indices) > 0:
            self.text_conditioner = PooledTextConditioner(
                embed_dim=embed_dim,
                num_conditioned_levels=len(self.level_indices),
                gate_init=text_bias_gate_init,
            )

        self.layers = nn.ModuleList([
            ShallowFusionEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(max(int(num_layers), 0))
        ])

    def _build_target_levels(self) -> List[int]:
        n = min(self.top_levels, self.num_levels)
        if n <= 0:
            return []
        return list(range(self.num_levels - n, self.num_levels))

    def forward(
        self,
        multi_feats: List[torch.Tensor],
        text_tokens: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        level_indices = self.level_indices
        if len(level_indices) == 0:
            return multi_feats

        outs = list(multi_feats)
        if self.text_conditioner is not None:
            outs = self.text_conditioner(
                multi_feats=outs,
                text_tokens=text_tokens,
                text_padding_mask=text_padding_mask,
                level_indices=level_indices,
            )

        if len(self.layers) == 0:
            return outs

        for lvl in level_indices:
            feat = outs[lvl]
            B, D, H, W = feat.shape
            tokens = feat.flatten(2).transpose(1, 2).contiguous()  # [B, HW, D]
            for layer in self.layers:
                tokens = layer(
                    image_tokens=tokens,
                    text_tokens=text_tokens,
                    text_padding_mask=text_padding_mask,
                )
            outs[lvl] = tokens.transpose(1, 2).reshape(B, D, H, W).contiguous()
        return outs


# =========================================================
# Pure PyTorch deformable-style cross attention
# =========================================================
class MultiScaleDeformableCrossAttention(nn.Module):
    """
    A pure PyTorch deformable-style cross attention.

    This is NOT the optimized CUDA op from Deformable DETR, but a functional,
    readable reference implementation for OVSAM.

    Design:
    - Each query predicts a small set of sampling offsets for each head, level, point.
    - Sampling points are centered around the current reference box.
    - Features are bilinearly sampled from each feature map via grid_sample.
    - Attention weights aggregate sampled features.

    Input:
        query:          [B, Q, D]
        reference_boxes:[B, Q, 4]   normalized cxcywh
        multi_feats:    list of [B, D, H_l, W_l]

    Output:
        out:            [B, Q, D]
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        # Predict sampling offsets and attention weights from the query
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.zeros_(self.sampling_offsets.weight)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)

        # A radial initialization for offsets, inspired by Deformable DETR
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)  # [H, 2]
        grid_init = grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]  # normalize

        # [H, L, P, 2]
        grid = grid_init[:, None, None, :].repeat(1, self.num_levels, self.num_points, 1)
        for p in range(self.num_points):
            grid[:, :, p, :] *= (p + 1)

        self.sampling_offsets.bias = nn.Parameter(grid.reshape(-1))

        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        reference_boxes: torch.Tensor,
        multi_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        query:          [B, Q, D]
        reference_boxes:[B, Q, 4] normalized cxcywh
        multi_feats:    list of [B, D, H_l, W_l]
        """
        B, Q, D = query.shape
        H = self.num_heads
        L = self.num_levels
        P = self.num_points
        Dh = self.head_dim

        assert len(multi_feats) == L, f"Expected {L} feature levels, got {len(multi_feats)}"

        # -------------------------------------------------
        # 1) Predict offsets and attention weights
        # -------------------------------------------------
        offsets = self.sampling_offsets(query).view(B, Q, H, L, P, 2)          # [B,Q,H,L,P,2]
        attn = self.attention_weights(query).view(B, Q, H, L, P)               # [B,Q,H,L,P]
        attn = F.softmax(attn.flatten(-2), dim=-1).view(B, Q, H, L, P)         # normalize over L*P

        # -------------------------------------------------
        # 2) Generate sampling locations around reference boxes
        #    reference_boxes are [cx,cy,w,h], normalized in [0,1]
        # -------------------------------------------------
        ref_xy = reference_boxes[..., :2].unsqueeze(2).unsqueeze(3).unsqueeze(4)   # [B,Q,1,1,1,2]
        ref_wh = reference_boxes[..., 2:].clamp(min=1e-4).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        # Tanh keeps offsets bounded; box size scales the search region
        sampling_locations = ref_xy + torch.tanh(offsets) * 0.5 * ref_wh           # [B,Q,H,L,P,2]
        sampling_locations = sampling_locations.clamp(0.0, 1.0)

        # -------------------------------------------------
        # 3) Sample from each feature level using grid_sample
        # -------------------------------------------------
        out = torch.zeros(B, Q, H, Dh, device=query.device, dtype=query.dtype)

        for lvl, feat in enumerate(multi_feats):
            # feat: [B, D, Hl, Wl]
            Bf, Df, Hl, Wl = feat.shape
            assert Bf == B and Df == D

            # Split feature channels by head:
            # [B, D, Hl, Wl] -> [B, H, Dh, Hl, Wl] -> [B*H, Dh, Hl, Wl]
            feat_h = feat.view(B, H, Dh, Hl, Wl).reshape(B * H, Dh, Hl, Wl)

            # sampling grid for this level:
            # [B, Q, H, P, 2] -> [B, H, Q, P, 2] -> [B*H, Q, P, 2]
            lvl_grid = sampling_locations[:, :, :, lvl, :, :]                    # [B,Q,H,P,2]
            lvl_grid = lvl_grid.permute(0, 2, 1, 3, 4).contiguous()              # [B,H,Q,P,2]
            lvl_grid = lvl_grid.view(B * H, Q, P, 2)

            # grid_sample expects normalized coordinates in [-1, 1]
            lvl_grid = lvl_grid * 2.0 - 1.0

            # Sample:
            # output: [B*H, Dh, Q, P]
            sampled = F.grid_sample(
                feat_h,
                lvl_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

            # reshape -> [B, Q, H, P, Dh]
            sampled = sampled.view(B, H, Dh, Q, P).permute(0, 3, 1, 4, 2).contiguous()

            # attention weights for this level: [B,Q,H,P,1]
            lvl_attn = attn[:, :, :, lvl, :].unsqueeze(-1)

            # weighted sum over points
            out = out + (sampled * lvl_attn).sum(dim=3)

        # -------------------------------------------------
        # 4) Merge heads and project out
        # -------------------------------------------------
        out = out.reshape(B, Q, D)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


# =========================================================
# One decoder layer
# =========================================================
class TextConditionedDeformableDecoderLayer(nn.Module):
    """
    One OVSAM decoder layer.

    Order:
    1) Query self-attention
    2) Image deformable cross-attention
    3) Text cross-attention
    4) FFN

    Why this order:
    - self-attn lets queries communicate and avoid duplicate predictions
    - deformable image cross-attn gathers local geometric evidence around reference boxes
    - text cross-attn re-aligns queries with the prompt semantics
    - FFN refines the query state
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.1,
        ffn_dim: int = 1024,
        use_presence_branch: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_presence_branch = bool(use_presence_branch)

        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(dropout)

        # Deformable cross-attention to image features
        self.image_cross_attn = MultiScaleDeformableCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop2 = nn.Dropout(dropout)

        # Cross-attention to text tokens
        self.text_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.drop3 = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm4 = nn.LayerNorm(embed_dim)

        self.presence_query_attn = None
        self.presence_text_attn = None
        self.presence_norm1 = None
        self.presence_norm2 = None
        self.presence_norm3 = None
        self.presence_drop1 = None
        self.presence_drop2 = None
        self.presence_ffn = None
        if self.use_presence_branch:
            self.presence_query_attn = nn.MultiheadAttention(
                embed_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.presence_text_attn = nn.MultiheadAttention(
                embed_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.presence_norm1 = nn.LayerNorm(embed_dim)
            self.presence_norm2 = nn.LayerNorm(embed_dim)
            self.presence_norm3 = nn.LayerNorm(embed_dim)
            self.presence_drop1 = nn.Dropout(dropout)
            self.presence_drop2 = nn.Dropout(dropout)
            self.presence_ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim),
                nn.Dropout(dropout),
            )

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        reference_boxes: torch.Tensor,
        multi_feats: List[torch.Tensor],
        text_tokens: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
        presence_token: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        query:            [B, Q, D]
        query_pos:        [B, Q, D]
        reference_boxes:  [B, Q, 4]
        multi_feats:      list of [B, D, H, W]
        text_tokens:      [B, T, D]
        """
        # -----------------------------------------
        # 1) Query self-attention
        # -----------------------------------------
        q = query + query_pos
        x, _ = self.self_attn(q, q, query, need_weights=False)
        query = self.norm1(query + self.drop1(x))

        # -----------------------------------------
        # 2) Image deformable cross-attention
        # -----------------------------------------
        x = self.image_cross_attn(query + query_pos, reference_boxes, multi_feats)
        query = self.norm2(query + self.drop2(x))

        # -----------------------------------------
        # 3) Text cross-attention
        # -----------------------------------------
        x, _ = self.text_cross_attn(
            query + query_pos,
            text_tokens,
            text_tokens,
            key_padding_mask=(~text_padding_mask.bool()) if text_padding_mask is not None else None,
            need_weights=False,
        )
        query = self.norm3(query + self.drop3(x))

        # -----------------------------------------
        # 4) FFN
        # -----------------------------------------
        x = self.ffn(query)
        query = self.norm4(query + x)

        if self.use_presence_branch and presence_token is not None:
            p = presence_token

            p_norm = self.presence_norm1(p)
            x, _ = self.presence_query_attn(
                p_norm,
                query,
                query,
                need_weights=False,
            )
            p = p + self.presence_drop1(x)

            p_norm = self.presence_norm2(p)
            x, _ = self.presence_text_attn(
                p_norm,
                text_tokens,
                text_tokens,
                key_padding_mask=(~text_padding_mask.bool()) if text_padding_mask is not None else None,
                need_weights=False,
            )
            p = p + self.presence_drop2(x)
            p = self.presence_norm3(p + self.presence_ffn(p))
            presence_token = p

        return query, presence_token


# =========================================================
# Main OVSAM deformable prompt decoder
# =========================================================
class DeformablePromptDecoder(nn.Module):
    """
    OVSAM Stage-1 decoder:
        (multi-scale image feats, text token feats, phrase feat)
            -> top-K boxes + scores

    Key design points:
    - Text-conditioned queries
    - DAB-style iterative reference box refinement
    - Image/text alternating decoder blocks
    - SAM3-style dot-product query scoring
    - Prompt-level presence token for prompt existence calibration
    """
    def __init__(
        self,
        image_in_channels: List[int],
        text_token_dim: int = 1024,
        phrase_dim: int = 512,
        embed_dim: int = 256,
        num_queries: int = 100,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        num_points: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        topk: int = 10,
        fusion_num_layers: int = 0,
        fusion_top_levels: int = 1,
        fusion_use_text_bias: bool = False,
        fusion_gate_init: float = 0.1,
        use_phrase_query_conditioning: bool = True,
        use_presence_branch: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.num_levels = len(image_in_channels)
        self.topk = topk
        self.use_phrase_query_conditioning = bool(use_phrase_query_conditioning)
        self.use_presence_branch = bool(use_presence_branch)

        # -------------------------------------------------
        # 1) Project image/text into shared decoder space
        # -------------------------------------------------
        self.image_proj = MultiScaleFeatureProjector(
            in_channels_list=image_in_channels,
            embed_dim=embed_dim,
        )

        self.text_token_proj = nn.Sequential(
            nn.Linear(text_token_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.fusion_encoder = None
        if fusion_num_layers > 0 or fusion_use_text_bias:
            self.fusion_encoder = ShallowPromptFusionEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                num_levels=self.num_levels,
                num_layers=fusion_num_layers,
                top_levels=fusion_top_levels,
                use_text_bias=fusion_use_text_bias,
                text_bias_gate_init=fusion_gate_init,
            )

        # phrase_feat is used to condition the initial learnable queries.
        self.phrase_query_cond = None
        if self.use_phrase_query_conditioning:
            self.phrase_query_cond = nn.Sequential(
                nn.Linear(phrase_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )

        # -------------------------------------------------
        # 2) Learnable object queries and initial reference boxes
        # -------------------------------------------------
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.ref_box_embed = nn.Embedding(num_queries, 4)  # normalized cxcywh after sigmoid

        # -------------------------------------------------
        # 3) Position encoding from current reference boxes
        # -------------------------------------------------
        self.ref_box_pos_encoder = ReferenceBoxPositionEncoder(embed_dim)

        # -------------------------------------------------
        # 4) Decoder layers
        # -------------------------------------------------
        self.layers = nn.ModuleList([
            TextConditionedDeformableDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_levels=self.num_levels,
                num_points=num_points,
                dropout=dropout,
                ffn_dim=ffn_dim,
                use_presence_branch=self.use_presence_branch,
            )
            for _ in range(num_decoder_layers)
        ])

        # -------------------------------------------------
        # 5) Per-layer box refinement heads
        # -------------------------------------------------
        self.box_heads = nn.ModuleList([
            MLP(embed_dim, embed_dim, 4, num_layers=3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])

        # -------------------------------------------------
        # 6) Prompt-conditioned scoring heads
        # -------------------------------------------------
        self.class_scoring = DotProductScoring(
            d_model=embed_dim,
            d_proj=embed_dim,
            clamp_logits=True,
            clamp_max_val=12.0,
        )
        self.presence_token = None
        self.presence_head = None
        self.presence_out_norm = None
        self.presence_logit_max_val = 12.0
        if self.use_presence_branch:
            self.presence_token = nn.Embedding(1, embed_dim)
            self.presence_head = MLP(embed_dim, embed_dim, 1, num_layers=3, dropout=dropout)
            self.presence_out_norm = nn.LayerNorm(embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.query_embed.weight, std=0.02)
        with torch.no_grad():
            # Initialize reference boxes in logit space so the forward sigmoid
            # yields anchors that already cover the full image plane.
            q = self.num_queries
            num_cols = math.ceil(math.sqrt(q))
            num_rows = math.ceil(q / num_cols)

            ys = (torch.arange(num_rows, dtype=self.ref_box_embed.weight.dtype) + 0.5) / float(num_rows)
            xs = (torch.arange(num_cols, dtype=self.ref_box_embed.weight.dtype) + 0.5) / float(num_cols)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            centers = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)[:q]

            # Use grid-cell size as the initial anchor prior.
            sizes = centers.new_tensor([1.0 / float(num_cols), 1.0 / float(num_rows)]).unsqueeze(0).repeat(q, 1)
            ref_boxes = torch.cat([centers, sizes], dim=-1).clamp(min=1e-3, max=1 - 1e-3)
            self.ref_box_embed.weight.copy_(inverse_sigmoid(ref_boxes))
        if self.presence_token is not None:
            nn.init.normal_(self.presence_token.weight, std=0.02)

    def _init_queries(self, phrase_feat: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize:
        - query content from learned queries + phrase conditioning
        - reference boxes from learned embeddings

        phrase_feat: [B, 512]
        returns:
          query:      [B, Q, D]
          ref_boxes:  [B, Q, 4] normalized cxcywh
        """
        if phrase_feat is None or self.phrase_query_cond is None:
            raise ValueError("phrase-conditioned query initialization is disabled or phrase_feat is missing.")

        B = phrase_feat.shape[0]

        # learned query content
        query = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B,Q,D]

        # phrase-conditioned bias
        phrase_bias = self.phrase_query_cond(phrase_feat).unsqueeze(1)  # [B,1,D]
        query = query + phrase_bias

        # learned initial reference boxes
        ref_boxes = torch.sigmoid(self.ref_box_embed.weight).unsqueeze(0).repeat(B, 1, 1)  # [B,Q,4]
        return query, ref_boxes

    def _presence_logits(
        self,
        presence_token: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if (not self.use_presence_branch) or presence_token is None or self.presence_head is None:
            return None
        logits = self.presence_head(self.presence_out_norm(presence_token))
        logits = logits.squeeze(-1).squeeze(-1)
        return logits.clamp(
            min=-self.presence_logit_max_val,
            max=self.presence_logit_max_val,
        )

    def _compute_ranking_scores(
        self,
        pred_logits: torch.Tensor,
        presence_logits: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Ranking score:
            score = sigmoid(class) * sigmoid(presence)
        or when presence branch is disabled:
            score = sigmoid(class)

        return: [B,Q] in [0,1]
        """
        cls = pred_logits.sigmoid()
        if (not self.use_presence_branch) or presence_logits is None:
            return cls
        pres = presence_logits.sigmoid().unsqueeze(1)
        return cls * pres

    def _select_topk(
        self,
        boxes: torch.Tensor,
        pred_logits: torch.Tensor,
        presence_logits: Optional[torch.Tensor],
        k: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Select top-K boxes using fused ranking score:
            score = sigmoid(class) * sigmoid(presence)
        """
        if k is None:
            k = self.topk

        scores = self._compute_ranking_scores(
            pred_logits=pred_logits,
            presence_logits=presence_logits,
        )  # [B,Q]

        B, Q = scores.shape
        k = min(k, Q)

        topk_scores, topk_idx = torch.topk(scores, k=k, dim=1)  # [B,K]
        topk_boxes = torch.gather(
            boxes,
            dim=1,
            index=topk_idx.unsqueeze(-1).expand(-1, -1, 4)
        )

        return {
            "ranking_scores": scores,   # [B,Q]
            "topk_scores": topk_scores,
            "topk_indices": topk_idx,
            "topk_boxes": topk_boxes,
        }

    def forward(
        self,
        multi_scale_feats: List[torch.Tensor],
        token_feats: torch.Tensor,
        phrase_feat: Optional[torch.Tensor],
        text_padding_mask: Optional[torch.Tensor] = None,
        topk: Optional[int] = None,
        return_aux: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            multi_scale_feats: list of backbone feature maps
                               each [B, C_l, H_l, W_l]
            token_feats:       [B, T, 1024] from OVSAMTextBackbone
            phrase_feat:       optional [B, 512] from OVSAMTextBackbone
            text_padding_mask: [B, T], 1 for valid tokens
        Returns:
            dict:
                pred_boxes         [B,Q,4]
                pred_logits        [B,Q]
        presence_logits    [B] or None when disabled
                topk_boxes         [B,K,4]
                topk_scores        [B,K]
                ranking_scores     [B,Q]
                aux_outputs        list of intermediate layer outputs
        """
        # -------------------------------------------------
        # 1) Project image/text into decoder space
        # -------------------------------------------------
        image_feats = self.image_proj(multi_scale_feats)           # list of [B,D,H,W]
        text_tokens = self.text_token_proj(token_feats)            # [B,T,D]
        if self.fusion_encoder is not None:
            image_feats = self.fusion_encoder(
                multi_feats=image_feats,
                text_tokens=text_tokens,
                text_padding_mask=text_padding_mask,
            )

        # -------------------------------------------------
        # 2) Initialize queries and reference boxes
        # -------------------------------------------------
        batch_size = token_feats.shape[0]
        if phrase_feat is None:
            query = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            ref_boxes = torch.sigmoid(self.ref_box_embed.weight).unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            query, ref_boxes = self._init_queries(phrase_feat)         # [B,Q,D], [B,Q,4]
        presence_token = None
        if self.use_presence_branch and self.presence_token is not None:
            presence_token = self.presence_token.weight.unsqueeze(0).expand(query.shape[0], -1, -1)

        aux_outputs = []

        # -------------------------------------------------
        # 3) Decoder loop with iterative box refinement
        # -------------------------------------------------
        for lid, layer in enumerate(self.layers):
            query_pos = self.ref_box_pos_encoder(ref_boxes)        # [B,Q,D]

            # one decoder block
            query, presence_token = layer(
                query=query,
                query_pos=query_pos,
                reference_boxes=ref_boxes,
                multi_feats=image_feats,
                text_tokens=text_tokens,
                text_padding_mask=text_padding_mask,
                presence_token=presence_token,
            )

            # iterative box refinement
            delta_box = self.box_heads[lid](query)                 # [B,Q,4]
            ref_boxes = torch.sigmoid(inverse_sigmoid(ref_boxes) + delta_box)

            pred_logits = self.class_scoring(
                query_states=query,
                text_tokens=text_tokens,
                text_padding_mask=text_padding_mask,
            )                                                            # [B,Q]
            presence_logits = self._presence_logits(presence_token)       # [B] or None

            if return_aux:
                aux_outputs.append({
                    "pred_boxes": ref_boxes,
                    "pred_logits": pred_logits,
                    "presence_logits": presence_logits,
                    "query_states": query,
                })

        # -------------------------------------------------
        # 4) Final outputs
        # -------------------------------------------------
        final_boxes = ref_boxes
        final_pred_logits = pred_logits
        final_presence_logits = presence_logits

        topk_dict = self._select_topk(
            boxes=final_boxes,
            pred_logits=final_pred_logits,
            presence_logits=final_presence_logits,
            k=topk,
        )

        out = {
            "pred_boxes": final_boxes,                    # [B,Q,4], normalized cxcywh
            "pred_logits": final_pred_logits,            # [B,Q]
            "presence_logits": final_presence_logits,    # [B] or None
            "query_states": query,                       # [B,Q,D]
            **topk_dict,
        }

        if return_aux:
            out["aux_outputs"] = aux_outputs

        return out


# =========================================================
# Simple self-test
# =========================================================
if __name__ == "__main__":
    # Dummy test to verify tensor shapes
    B = 2
    T = 32

    # Example multi-scale image features from a backbone
    feats = [
        torch.randn(B, 256, 128, 128),
        torch.randn(B, 512, 64, 64),
        torch.randn(B, 1024, 32, 32),
        torch.randn(B, 1024, 16, 16),
    ]

    # Text features from OVSAMTextBackbone
    token_feats = torch.randn(B, T, 1024)
    phrase_feat = torch.randn(B, 512)
    text_padding_mask = torch.ones(B, T, dtype=torch.long)

    model = DeformablePromptDecoder(
        image_in_channels=[256, 512, 1024, 1024],
        text_token_dim=1024,
        phrase_dim=512,
        embed_dim=256,
        num_queries=100,
        num_decoder_layers=6,
        num_heads=8,
        num_points=4,
        ffn_dim=1024,
        dropout=0.1,
        topk=10,
    )

    outputs = model(
        multi_scale_feats=feats,
        token_feats=token_feats,
        phrase_feat=phrase_feat,
        text_padding_mask=text_padding_mask,
        topk=10,
        return_aux=True,
    )

    print("pred_boxes         :", outputs["pred_boxes"].shape)          # [B,Q,4]
    print("pred_logits        :", outputs["pred_logits"].shape)         # [B,Q]
    print("presence_logits    :", outputs["presence_logits"].shape)     # [B]
    print("ranking_scores     :", outputs["ranking_scores"].shape)      # [B,Q]
    print("topk_boxes         :", outputs["topk_boxes"].shape)          # [B,K,4]
    print("topk_scores        :", outputs["topk_scores"].shape)         # [B,K]
    print("num aux outputs    :", len(outputs["aux_outputs"]))

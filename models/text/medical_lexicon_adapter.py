import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBottleneckAdapter(nn.Module):
    """
    A lightweight residual bottleneck adapter:
        y = x + gate * Up(GELU(Down(LN(x))))
    Supports both token-level [B, T, D] and phrase-level [B, D].
    """
    def __init__(
        self,
        dim: int,
        bottleneck_dim: int = 256,
        dropout: float = 0.0,
        gate_init: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.down(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.up(x)
        return residual + self.gate * x


class AttentionPooling(nn.Module):
    """
    Attention pooling over token dimension:
        alpha = softmax(q^T LN(H))
        e = sum(alpha_t * H_t)
    Input:  H [B, T, D]
    Output: e [B, D]
    """
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, token_feats: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            token_feats: [B, T, D]
            attention_mask: [B, T], 1 for valid tokens, 0 for padding
        Returns:
            pooled: [B, D]
        """
        x = self.norm(token_feats)
        scores = torch.einsum("btd,d->bt", x, self.query)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bt,btd->bd", attn, token_feats)
        return pooled


class PhraseProjectionHead(nn.Module):
    """
    Project pooled phrase embedding from 1024 -> 512 (default),
    then L2-normalize for phrase-level conditioning.
    """
    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 1024,
        out_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = F.normalize(x, dim=-1)
        return x


class MedicalLexiconAdapter(nn.Module):
    """
    Medical Lexicon Adapter for frozen SAM3 text encoder outputs.

    Input:
        last_hidden_state: [B, T, 1024]
        attention_mask:    [B, T] or None

    Output:
        token_feats: [B, T, 1024] -> for decoder cross-attention and scoring
        phrase_feat: [B, 512]     -> for prompt-level conditioning
        aux: dict containing optional intermediate tensors
    """
    def __init__(
        self,
        token_dim: int = 1024,
        bottleneck_dim: int = 256,
        phrase_out_dim: int = 512,
        dropout: float = 0.0,
        gate_init: float = 0.1,
    ):
        super().__init__()

        self.token_adapter = ResidualBottleneckAdapter(
            dim=token_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
            gate_init=gate_init,
        )

        self.pool = AttentionPooling(dim=token_dim)

        self.phrase_adapter = ResidualBottleneckAdapter(
            dim=token_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
            gate_init=gate_init,
        )

        self.phrase_head = PhraseProjectionHead(
            in_dim=token_dim,
            hidden_dim=token_dim,
            out_dim=phrase_out_dim,
            dropout=dropout,
        )

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor = None,
        return_aux: bool = True,
    ):
        token_feats = self.token_adapter(last_hidden_state)
        pooled = self.pool(token_feats, attention_mask=attention_mask)
        pooled = self.phrase_adapter(pooled)
        phrase_feat = self.phrase_head(pooled)

        if not return_aux:
            return token_feats, phrase_feat

        aux = {
            "pooled_feat": pooled,
            "token_gate": self.token_adapter.gate.detach(),
            "phrase_gate": self.phrase_adapter.gate.detach(),
        }
        return token_feats, phrase_feat, aux

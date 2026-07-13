from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TransformerConfig:
    input_dim: int
    layers: int = 4
    d_model: int = 128
    heads: int = 4
    d_ffn: int = 512
    dropout: float = 0.1
    sequence_length: int = 36


class ClinicalTransformerLayer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(cfg.d_model, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ffn),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ffn, cfg.d_model),
        )

    def forward(self, x: torch.Tensor, replacement: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn))
        h_ffn = x
        a_ffn = self.ffn(h_ffn) if replacement is None else replacement
        x = self.norm2(x + self.dropout(a_ffn))
        return x, h_ffn, a_ffn


class ClinicalTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.sequence_length, cfg.d_model))
        self.layers = nn.ModuleList([ClinicalTransformerLayer(cfg) for _ in range(cfg.layers)])
        self.head = nn.Linear(cfg.d_model, 1)
        nn.init.normal_(self.pos, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
        replacements: Dict[int, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        h = self.input_proj(x) + self.pos[:, : x.shape[1]]
        h_ffn: List[torch.Tensor] = []
        a_ffn: List[torch.Tensor] = []
        replacements = replacements or {}
        for layer_id, layer in enumerate(self.layers):
            h, h_layer, a_layer = layer(h, replacements.get(layer_id))
            if return_activations:
                h_ffn.append(h_layer)
                a_ffn.append(a_layer)
        logit = self.head(h.mean(dim=1)).squeeze(-1)
        out: Dict[str, torch.Tensor | List[torch.Tensor]] = {"logit": logit, "probability": torch.sigmoid(logit)}
        if return_activations:
            out["h_ffn"] = h_ffn
            out["a_ffn"] = a_ffn
        return out

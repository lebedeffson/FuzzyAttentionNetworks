from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .fuzzy_attention import FuzzyTemporalAttention


class FuzzyTemporalAttentionLayer(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, d_ffn: int = 512, dropout: float = 0.1):
        super().__init__()
        self.attn = FuzzyTemporalAttention(d_model=d_model, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ffn, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        a, weights, _ = self.attn(x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(a))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1).bool(), 0.0)
        return x, weights


class FuzzyTemporalAttentionEncoder(nn.Module):
    def __init__(self, input_dim: int, sequence_length: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 4, d_ffn: int = 512, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        self.layers = nn.ModuleList([FuzzyTemporalAttentionLayer(d_model, n_heads, d_ffn, dropout) for _ in range(n_layers)])
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None, return_attention: bool = False):
        h = self.input_proj(x) + self.pos[:, : x.shape[1]]
        weights = []
        for layer in self.layers:
            h, w = layer(h, key_padding_mask)
            weights.append(w)
        return (h, weights) if return_attention else h

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .memberships import GaussianMembership
from .tnorms import product_tnorm, softmin_tnorm


class FuzzyTemporalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_memberships: int = 3, tnorm: str = "product", mode: str = "hybrid"):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if tnorm not in {"product", "softmin"}:
            raise ValueError("tnorm must be product or softmin")
        if mode not in {"hybrid", "fuzzy_only"}:
            raise ValueError("mode must be hybrid or fuzzy_only")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.n_memberships = int(n_memberships)
        self.tnorm = tnorm
        self.mode = mode
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.membership = GaussianMembership(n_heads, n_memberships, self.head_dim)
        self.rule_logits = nn.Parameter(torch.zeros(n_heads, n_memberships))
        self.raw_gate = nn.Parameter(torch.full((n_heads,), torch.logit(torch.tensor(0.8))))

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = x.shape
        return x.view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        bsz, seq, _ = x.shape
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(x))
        v = self._split(self.v_proj(x))
        dot = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mu_q = self.membership(q)
        mu_k = self.membership(k)
        if self.tnorm == "product":
            rules = product_tnorm(mu_q.unsqueeze(3), mu_k.unsqueeze(2)).mean(dim=-1)
        else:
            rules = softmin_tnorm(mu_q.unsqueeze(3), mu_k.unsqueeze(2)).mean(dim=-1)
        # rules: [B,H,Tq,Tk,M]
        rule_w = torch.softmax(self.rule_logits, dim=-1).view(1, self.n_heads, 1, 1, self.n_memberships)
        fuzzy = (rules * rule_w).sum(dim=-1).clamp(1e-6, 1 - 1e-6)
        fuzzy_score = torch.logit(fuzzy)
        gate = torch.sigmoid(self.raw_gate).view(1, self.n_heads, 1, 1)
        scores = fuzzy_score if self.mode == "fuzzy_only" else gate * dot + (1.0 - gate) * fuzzy_score
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.view(1, 1, seq, seq).bool(), -torch.inf)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.view(bsz, 1, 1, seq).bool(), -torch.inf)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = attn @ v
        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask.view(bsz, 1, seq, 1).bool(), 0.0)
            attn = attn.masked_fill(key_padding_mask.view(bsz, 1, seq, 1).bool(), 0.0)
        out = out.transpose(1, 2).contiguous().view(bsz, seq, self.d_model)
        out = self.out_proj(out)
        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask.unsqueeze(-1).bool(), 0.0)
        return out, attn, {"mu_q": mu_q, "mu_k": mu_k, "gate": torch.sigmoid(self.raw_gate), "fuzzy_score": fuzzy_score}

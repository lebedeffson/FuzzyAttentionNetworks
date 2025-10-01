"""
Fuzzy Attention Networks (FAN) - Core implementation
Differentiable neuro-fuzzy architectures for interpretable transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any

# Импортируем напрямую из src
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from utils import FuzzyOperators, gaussian_membership
from config import config

class FuzzyMembership(nn.Module):
    """Learnable fuzzy membership functions"""

    def __init__(self, input_dim: int, n_functions: int = 3, 
                 function_type: str = 'gaussian'):
        super().__init__()
        self.input_dim = input_dim
        self.n_functions = n_functions
        self.function_type = function_type

        if function_type == 'gaussian':
            # Learnable centers and sigmas for Gaussian membership
            self.centers = nn.Parameter(torch.randn(n_functions, input_dim))
            self.sigmas = nn.Parameter(torch.ones(n_functions, input_dim) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute fuzzy membership values
        """
        if self.function_type == 'gaussian':
            return gaussian_membership(x, self.centers, torch.abs(self.sigmas))
        return torch.ones(*x.shape[:-1], self.n_functions, device=x.device)

class FuzzyAttentionHead(nn.Module):
    """Single fuzzy attention head with learnable fuzzy logic"""

    def __init__(self, d_model: int, d_k: int, fuzzy_type: str = 'product',
                 n_membership_functions: int = 3):
        super().__init__()
        self.d_k = d_k
        self.fuzzy_type = fuzzy_type

        # Standard Q, K, V projections
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)

        # Fuzzy membership functions for Q and K
        self.fuzzy_q = FuzzyMembership(d_k, n_membership_functions)
        self.fuzzy_k = FuzzyMembership(d_k, n_membership_functions)

        # Learnable fuzzy parameters
        self.fuzzy_weight = nn.Parameter(torch.tensor(0.5))
        self.fuzzy_temperature = nn.Parameter(torch.tensor(1.0))

        # T-norm operator
        self.tnorm = FuzzyOperators.get_tnorm(fuzzy_type)
        self.scale = 1.0 / math.sqrt(d_k)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               mask: Optional[torch.Tensor] = None, 
               return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fuzzy attention forward pass"""

        # Project to Q, K, V
        Q = self.W_q(query)
        K = self.W_k(key) 
        V = self.W_v(value)

        # Standard attention scores
        standard_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Fuzzy logic enhancement
        fuzzy_scores = self._compute_fuzzy_scores(Q, K)

        # Combine standard and fuzzy scores
        fuzzy_weight = torch.sigmoid(self.fuzzy_weight)
        combined_scores = (1 - fuzzy_weight) * standard_scores + fuzzy_weight * fuzzy_scores

        # Apply temperature
        combined_scores = combined_scores / torch.abs(self.fuzzy_temperature)

        # Apply mask if provided
        if mask is not None:
            combined_scores.masked_fill_(mask == 0, -1e9)

        # Compute attention weights
        attention_weights = F.softmax(combined_scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)

        if return_attention:
            return output, attention_weights
        return output, None

    def _compute_fuzzy_scores(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Compute fuzzy attention scores using t-norms"""
        batch_size, seq_len_q, d_k = Q.shape
        seq_len_k = K.shape[1]

        # Compute fuzzy memberships
        membership_q = self.fuzzy_q(Q)
        membership_k = self.fuzzy_k(K)

        # Compute pairwise fuzzy interactions
        fuzzy_scores = torch.zeros(batch_size, seq_len_q, seq_len_k, 
                                 device=Q.device, dtype=Q.dtype)

        for i in range(membership_q.shape[-1]):
            mu_q_i = membership_q[..., i:i+1]
            mu_k_i = membership_k[..., i].unsqueeze(1)

            tnorm_result = self.tnorm(mu_q_i, mu_k_i)
            fuzzy_scores += tnorm_result

        # Normalize by number of membership functions
        fuzzy_scores = fuzzy_scores / membership_q.shape[-1]
        return fuzzy_scores

class MultiHeadFuzzyAttention(nn.Module):
    """Multi-head fuzzy attention mechanism"""

    def __init__(self, d_model: int, n_heads: int, fuzzy_type: str = 'product'):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Multiple fuzzy attention heads
        self.heads = nn.ModuleList([
            FuzzyAttentionHead(d_model, self.d_k, fuzzy_type)
            for _ in range(n_heads)
        ])

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)  # Using default value

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               mask: Optional[torch.Tensor] = None,
               return_attention: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Multi-head fuzzy attention forward pass"""

        head_outputs = []
        head_attentions = [] if return_attention else None

        # Process each head
        for head in self.heads:
            head_out, head_attn = head(query, key, value, mask, return_attention)
            head_outputs.append(head_out)
            if return_attention:
                head_attentions.append(head_attn)

        # Concatenate heads
        concat_output = torch.cat(head_outputs, dim=-1)

        # Final projection
        output = self.W_o(concat_output)
        output = self.dropout(output)

        if return_attention:
            attention_info = {
                'head_attentions': head_attentions,
                'avg_attention': torch.stack(head_attentions).mean(dim=0)
            }
            return output, attention_info

        return output, None

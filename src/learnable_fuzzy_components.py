#!/usr/bin/env python3
"""
Learnable Fuzzy Components for Fuzzy Attention Networks
Implements learnable membership functions and differentiable t-norms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple

class LearnableGaussianMembership(nn.Module):
    """Learnable Gaussian membership function"""
    
    def __init__(self, input_dim: int, num_functions: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.num_functions = num_functions
        
        # Learnable parameters for each membership function
        self.centers = nn.Parameter(torch.randn(num_functions, input_dim))
        self.sigmas = nn.Parameter(torch.ones(num_functions, input_dim))
        self.weights = nn.Parameter(torch.ones(num_functions))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute membership values for input x
        
        Args:
            x: [batch_size, seq_len, input_dim] input tensor
            
        Returns:
            [batch_size, seq_len, num_functions] membership values
        """
        batch_size, seq_len, _ = x.shape
        
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(-2)  # [batch, seq, 1, input_dim]
        centers_expanded = self.centers.unsqueeze(0).unsqueeze(0)  # [1, 1, num_func, input_dim]
        sigmas_expanded = self.sigmas.unsqueeze(0).unsqueeze(0)  # [1, 1, num_func, input_dim]
        
        # Compute Gaussian membership: exp(-0.5 * ((x - center) / sigma)^2)
        diff = x_expanded - centers_expanded
        normalized_diff = diff / (sigmas_expanded + 1e-8)
        gaussian = torch.exp(-0.5 * (normalized_diff ** 2))
        
        # Product over input dimensions
        membership = torch.prod(gaussian, dim=-1)  # [batch, seq, num_functions]
        
        # Apply learnable weights
        weighted_membership = membership * self.weights.unsqueeze(0).unsqueeze(0)
        
        return weighted_membership

class LearnableTriangularMembership(nn.Module):
    """Learnable Triangular membership function"""
    
    def __init__(self, input_dim: int, num_functions: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.num_functions = num_functions
        
        # Learnable parameters
        self.centers = nn.Parameter(torch.randn(num_functions, input_dim))
        self.widths = nn.Parameter(torch.ones(num_functions, input_dim))
        self.weights = nn.Parameter(torch.ones(num_functions))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute triangular membership values"""
        batch_size, seq_len, _ = x.shape
        
        x_expanded = x.unsqueeze(-2)  # [batch, seq, 1, input_dim]
        centers_expanded = self.centers.unsqueeze(0).unsqueeze(0)
        widths_expanded = self.widths.unsqueeze(0).unsqueeze(0)
        
        # Triangular membership: max(0, 1 - |x - center| / width)
        diff = torch.abs(x_expanded - centers_expanded)
        triangular = torch.clamp(1 - diff / (widths_expanded + 1e-8), min=0)
        
        # Product over input dimensions
        membership = torch.prod(triangular, dim=-1)
        
        # Apply learnable weights
        weighted_membership = membership * self.weights.unsqueeze(0).unsqueeze(0)
        
        return weighted_membership

class LearnableTNorm(nn.Module):
    """Learnable differentiable t-norm"""
    
    def __init__(self, input_dim: int, num_tnorms: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.num_tnorms = num_tnorms
        
        # Learnable weights for different t-norms
        self.tnorm_weights = nn.Parameter(torch.ones(num_tnorms))
        
        # T-norm types: product, minimum, Lukasiewicz
        self.tnorm_types = ['product', 'minimum', 'lukasiewicz']
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable t-norm to inputs
        
        Args:
            x, y: [batch_size, seq_len, input_dim] input tensors
            
        Returns:
            [batch_size, seq_len, input_dim] t-norm result
        """
        # Normalize weights
        weights = F.softmax(self.tnorm_weights, dim=0)
        
        # Compute different t-norms
        product_tnorm = x * y
        min_tnorm = torch.min(x, y)
        lukasiewicz_tnorm = torch.clamp(x + y - 1, min=0)
        
        # Weighted combination
        result = (weights[0] * product_tnorm + 
                 weights[1] * min_tnorm + 
                 weights[2] * lukasiewicz_tnorm)
        
        return result

class LearnableFuzzyAttention(nn.Module):
    """Learnable Fuzzy Attention with membership functions and t-norms"""
    
    def __init__(self, 
                 input_dim: int, 
                 num_heads: int = 8,
                 num_membership_functions: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        # Learnable membership functions
        self.gaussian_membership = LearnableGaussianMembership(
            self.head_dim, num_membership_functions
        )
        self.triangular_membership = LearnableTriangularMembership(
            self.head_dim, num_membership_functions
        )
        
        # Learnable t-norms
        self.tnorm = LearnableTNorm(self.head_dim)
        
        # Linear projections
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.output_proj = nn.Linear(input_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply learnable fuzzy attention
        
        Args:
            x: [batch_size, seq_len, input_dim] input tensor
            
        Returns:
            output: [batch_size, seq_len, input_dim] output tensor
            attention_weights: [batch_size, num_heads, seq_len, seq_len] attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.query_proj(x)  # [batch, seq, input_dim]
        K = self.key_proj(x)
        V = self.value_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch, num_heads, seq, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute fuzzy attention weights
        attention_weights = self._compute_fuzzy_attention(Q, K)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Reshape back
        attended_values = attended_values.transpose(1, 2).contiguous()
        attended_values = attended_values.view(batch_size, seq_len, self.input_dim)
        
        # Output projection
        output = self.output_proj(attended_values)
        output = self.dropout(output)
        
        return output, attention_weights
    
    def _compute_fuzzy_attention(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Compute fuzzy attention weights using learnable membership functions"""
        batch_size, num_heads, seq_len, head_dim = Q.shape
        
        # Compute membership values for queries and keys
        Q_membership = self.gaussian_membership(Q)  # [batch, num_heads, seq, num_functions]
        K_membership = self.gaussian_membership(K)
        
        # Compute attention scores using t-norm
        attention_scores = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=Q.device)
        
        for i in range(seq_len):
            for j in range(seq_len):
                # Get membership values for position i (query) and j (key)
                q_mem = Q_membership[:, :, i, :]  # [batch, num_heads, num_functions]
                k_mem = K_membership[:, :, j, :]  # [batch, num_heads, num_functions]
                
                # Apply t-norm to compute attention strength
                attention_strength = self.tnorm(q_mem, k_mem)  # [batch, num_heads, num_functions]
                
                # Sum over membership functions to get final attention score
                attention_scores[:, :, i, j] = attention_strength.sum(dim=-1)
        
        # Apply softmax for normalization
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        return attention_weights

class MembershipFunctionVisualizer:
    """Visualizer for membership functions"""
    
    def __init__(self, membership_functions: Dict[str, nn.Module]):
        self.membership_functions = membership_functions
    
    def visualize_membership_functions(self, x_range: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Visualize membership functions over a range of values
        
        Args:
            x_range: [num_points] tensor of input values
            
        Returns:
            Dict mapping function names to membership values
        """
        visualizations = {}
        
        for name, func in self.membership_functions.items():
            # Expand x_range to match expected input shape
            x_input = x_range.unsqueeze(0).unsqueeze(-1)  # [1, num_points, 1]
            
            with torch.no_grad():
                membership_values = func(x_input)  # [1, num_points, num_functions]
                visualizations[name] = membership_values.squeeze(0)  # [num_points, num_functions]
        
        return visualizations

class CompositionalRuleDeriver:
    """Derives compositional rules from fuzzy attention weights"""
    
    def __init__(self, rule_extractor):
        self.rule_extractor = rule_extractor
    
    def derive_compositional_rules(self, 
                                 attention_weights: torch.Tensor,
                                 tokens: List[str]) -> List[Dict[str, any]]:
        """
        Derive compositional rules showing how individual rules combine
        
        Args:
            attention_weights: [batch_size, seq_len, seq_len] attention matrix
            tokens: List of tokens
            
        Returns:
            List of compositional rule derivations
        """
        # Extract individual rules
        individual_rules = self.rule_extractor.extract_rules(attention_weights, tokens)
        
        # Group rules by source position
        rules_by_position = {}
        for rule in individual_rules:
            pos = rule.from_position
            if pos not in rules_by_position:
                rules_by_position[pos] = []
            rules_by_position[pos].append(rule)
        
        # Create compositional derivations
        compositional_rules = []
        
        for pos, rules in rules_by_position.items():
            if len(rules) > 1:  # Only create compositions for positions with multiple rules
                # Sort rules by strength
                sorted_rules = sorted(rules, key=lambda r: r.strength, reverse=True)
                
                # Create compositional rule
                composition = {
                    'source_position': pos,
                    'source_token': tokens[pos] if pos < len(tokens) else f"pos_{pos}",
                    'individual_rules': sorted_rules,
                    'composition_strength': sum(r.strength for r in sorted_rules),
                    'composition_type': 'conjunctive',  # Rules are combined with AND
                    'derivation': self._create_derivation_text(sorted_rules, tokens)
                }
                
                compositional_rules.append(composition)
        
        return compositional_rules
    
    def _create_derivation_text(self, rules: List, tokens: List[str]) -> str:
        """Create human-readable derivation text"""
        if not rules:
            return "No rules to derive"
        
        # Create derivation showing how rules combine
        rule_texts = []
        for i, rule in enumerate(rules):
            target_token = tokens[rule.to_position] if rule.to_position < len(tokens) else f"pos_{rule.to_position}"
            rule_texts.append(f"Rule {i+1}: {rule.linguistic_description} (strength: {rule.strength:.3f})")
        
        derivation = f"Compositional derivation for position {rules[0].from_position}:\n"
        derivation += f"Source token: {tokens[rules[0].from_position] if rules[0].from_position < len(tokens) else f'pos_{rules[0].from_position}'}\n"
        derivation += f"Combined rules:\n" + "\n".join(rule_texts)
        derivation += f"\nTotal composition strength: {sum(r.strength for r in rules):.3f}"
        
        return derivation

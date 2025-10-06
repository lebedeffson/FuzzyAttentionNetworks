"""
Simplified Fuzzy Attention Model that actually learns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Tuple


class SimpleFuzzyAttention(nn.Module):
    """Simplified fuzzy attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Fuzzy membership functions (simplified)
        self.fuzzy_centers = nn.Parameter(torch.randn(n_heads, 3))  # 3 membership functions per head
        self.fuzzy_widths = nn.Parameter(torch.ones(n_heads, 3))
        
        # Temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, query, key, value, return_attention=False):
        batch_size, seq_len, d_model = query.size()
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # Apply fuzzy membership functions
        fuzzy_scores = self._apply_fuzzy_membership(scores)
        
        # Apply temperature scaling
        fuzzy_scores = fuzzy_scores / self.temperature
        
        # Softmax
        attention_weights = F.softmax(fuzzy_scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Reshape and project
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_o(attended)
        
        if return_attention:
            avg_attention = attention_weights.mean(dim=1)  # Average across heads
            return output, {'avg_attention': avg_attention}
        
        return output, None
    
    def _apply_fuzzy_membership(self, scores):
        """Apply simplified fuzzy membership functions"""
        # Reshape scores for fuzzy processing
        batch_size, n_heads, seq_len, _ = scores.size()
        
        # Create fuzzy membership values
        fuzzy_values = torch.zeros_like(scores)
        
        for head in range(n_heads):
            for func in range(3):  # 3 membership functions
                center = self.fuzzy_centers[head, func]
                width = self.fuzzy_widths[head, func]
                
                # Gaussian membership function
                membership = torch.exp(-((scores - center) ** 2) / (2 * width ** 2))
                fuzzy_values += membership
        
        # Normalize
        fuzzy_values = fuzzy_values / 3.0
        
        return fuzzy_values


class SimpleMultimodalFuzzyModel(nn.Module):
    """Simplified multimodal fuzzy model that learns"""
    
    def __init__(self, vocab_size=10000, text_dim=256, image_dim=2048, 
                 hidden_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        
        # Text processing
        self.text_embedding = nn.Embedding(vocab_size, text_dim)
        self.text_pos_embedding = nn.Embedding(128, text_dim)
        
        # Image processing
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        self.image_pos_embedding = nn.Embedding(196, hidden_dim)
        
        # Fuzzy attention layers
        self.text_attention_layers = nn.ModuleList([
            SimpleFuzzyAttention(text_dim, n_heads) for _ in range(n_layers)
        ])
        
        self.image_attention_layers = nn.ModuleList([
            SimpleFuzzyAttention(hidden_dim, n_heads) for _ in range(n_layers)
        ])
        
        # Cross-modal attention
        self.cross_attention = SimpleFuzzyAttention(text_dim + hidden_dim, n_heads)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(text_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, text_tokens, image_features, return_explanations=False):
        batch_size = text_tokens.size(0)
        device = text_tokens.device
        
        # Text processing
        text_emb = self.text_embedding(text_tokens)
        text_pos = self.text_pos_embedding(torch.arange(text_tokens.size(1), device=device))
        text_features = text_emb + text_pos.unsqueeze(0)
        
        # Image processing
        image_proj = self.image_projection(image_features)
        image_pos = self.image_pos_embedding(torch.arange(image_features.size(1), device=device))
        image_features = image_proj + image_pos.unsqueeze(0)
        
        # Self-attention layers
        for text_attn, img_attn in zip(self.text_attention_layers, self.image_attention_layers):
            text_features, _ = text_attn(text_features, text_features, text_features)
            image_features, _ = img_attn(image_features, image_features, image_features)
        
        # Cross-modal attention
        # Concatenate text and image features
        text_pooled = text_features.mean(dim=1)  # [batch, text_dim]
        image_pooled = image_features.mean(dim=1)  # [batch, hidden_dim]
        
        # Create cross-modal features
        cross_features = torch.cat([text_pooled, image_pooled], dim=-1)
        cross_features = cross_features.unsqueeze(1)  # [batch, 1, text_dim + hidden_dim]
        
        # Apply cross-modal attention
        cross_attended, _ = self.cross_attention(cross_features, cross_features, cross_features)
        cross_attended = cross_attended.squeeze(1)  # [batch, text_dim + hidden_dim]
        
        # Classification
        logits = self.classifier(cross_attended)
        probs = F.softmax(logits, dim=-1)
        
        result = {
            'binary_logits': logits,
            'binary_probs': probs,
            'predictions': torch.argmax(logits, dim=-1),
            'confidence': torch.max(probs, dim=-1)[0]
        }
        
        if return_explanations:
            result['explanations'] = {
                'prediction': result['predictions'].item() if result['predictions'].dim() == 0 else result['predictions'][0].item(),
                'confidence': result['confidence'].item() if result['confidence'].dim() == 0 else result['confidence'][0].item(),
                'prediction_text': 'Hateful' if result['predictions'].item() == 1 else 'Non-hateful'
            }
        
        return result


def demo_simple_fuzzy_model():
    """Demo the simple fuzzy model"""
    print("🚀 Simple Fuzzy Model Demo")
    print("=" * 40)
    
    # Create model
    model = SimpleMultimodalFuzzyModel(
        vocab_size=10000,
        text_dim=256,
        image_dim=2048,
        hidden_dim=128,
        n_heads=4,
        n_layers=2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size, question_len, image_patches = 2, 10, 49
    question_tokens = torch.randint(0, 10000, (batch_size, question_len))
    image_features = torch.randn(batch_size, image_patches, 2048)
    
    with torch.no_grad():
        result = model(question_tokens, image_features, return_explanations=True)
    
    print(f"Output shapes:")
    print(f"  Logits: {result['binary_logits'].shape}")
    print(f"  Probs: {result['binary_probs'].shape}")
    print(f"  Predictions: {result['predictions'].shape}")
    print(f"  Confidence: {result['confidence'].shape}")
    
    print(f"\nSample predictions:")
    for i in range(batch_size):
        pred = "Hateful" if result['predictions'][i].item() == 1 else "Non-hateful"
        conf = result['confidence'][i].item()
        print(f"  Sample {i+1}: {pred} (confidence: {conf:.3f})")
    
    print(f"\n✅ Simple fuzzy model working!")
    return model


if __name__ == "__main__":
    demo_simple_fuzzy_model()

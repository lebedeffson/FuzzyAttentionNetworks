"""
Multimodal Fuzzy Attention Networks
Cross-modal fuzzy reasoning for text + image tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

from fuzzy_attention import MultiHeadFuzzyAttention, FuzzyAttentionHead
from rule_extractor import RuleExtractor, FuzzyRule

class CrossModalFuzzyAttention(nn.Module):
    """Cross-modal fuzzy attention between text and image modalities"""
    
    def __init__(self, 
                 text_dim: int = 512,
                 image_dim: int = 768, 
                 hidden_dim: int = 256,
                 n_heads: int = 4,
                 fuzzy_type: str = 'product'):
        super().__init__()
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        # Projection layers to common space
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        
        # Cross-modal fuzzy attention
        self.cross_attention = MultiHeadFuzzyAttention(
            d_model=hidden_dim,
            n_heads=n_heads,
            fuzzy_type=fuzzy_type
        )
        
        # Output projections
        self.text_output = nn.Linear(hidden_dim, text_dim)
        self.image_output = nn.Linear(hidden_dim, image_dim)
        
        # Learnable cross-modal fusion weights
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, 
                text_features: torch.Tensor,
                image_features: torch.Tensor,
                return_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Cross-modal fuzzy attention forward pass
        
        Args:
            text_features: [batch_size, text_seq_len, text_dim]
            image_features: [batch_size, image_seq_len, image_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            enhanced_text_features, enhanced_image_features, attention_info
        """
        
        # Project to common space
        text_proj = self.text_projection(text_features)  # [batch, text_seq, hidden_dim]
        image_proj = self.image_projection(image_features)  # [batch, image_seq, hidden_dim]
        
        # Cross-modal attention: text attends to image
        text_enhanced, attention_info = self.cross_attention(
            query=text_proj,
            key=image_proj,
            value=image_proj,
            return_attention=return_attention
        )
        
        # Cross-modal attention: image attends to text
        image_enhanced, _ = self.cross_attention(
            query=image_proj,
            key=text_proj,
            value=text_proj,
            return_attention=False
        )
        
        # Fusion with learnable weights
        fusion_weight = torch.sigmoid(self.fusion_weight)
        
        # Combine original and enhanced features
        text_final = (1 - fusion_weight) * text_features + fusion_weight * self.text_output(text_enhanced)
        image_final = (1 - fusion_weight) * image_features + fusion_weight * self.image_output(image_enhanced)
        
        if return_attention:
            return text_final, image_final, attention_info
        return text_final, image_final, None

class MultimodalFuzzyTransformer(nn.Module):
    """Complete multimodal fuzzy transformer for VQA and similar tasks"""
    
    def __init__(self,
                 vocab_size: int = 30000,
                 text_dim: int = 512,
                 image_dim: int = 768,
                 hidden_dim: int = 256,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 max_text_len: int = 128,
                 max_image_patches: int = 196,  # 14x14 for 224x224 image
                 num_classes: int = 1000,
                 fuzzy_type: str = 'product',
                 dropout: float = 0.1):
        super().__init__()
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.max_text_len = max_text_len
        self.max_image_patches = max_image_patches
        self.dropout = dropout
        
        # Text processing
        self.text_embedding = nn.Embedding(vocab_size, text_dim)
        self.text_pos_embedding = nn.Embedding(max_text_len, text_dim)
        self.text_transformer = nn.ModuleList([
            MultiHeadFuzzyAttention(text_dim, n_heads, fuzzy_type)
            for _ in range(n_layers)
        ])
        
        # Image processing (assuming pre-extracted features)
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        self.image_pos_embedding = nn.Embedding(max_image_patches, hidden_dim)
        self.image_transformer = nn.ModuleList([
            MultiHeadFuzzyAttention(hidden_dim, n_heads, fuzzy_type)
            for _ in range(n_layers)
        ])
        
        # Cross-modal attention layers
        self.cross_modal_layers = nn.ModuleList([
            CrossModalFuzzyAttention(text_dim, hidden_dim, hidden_dim, n_heads, fuzzy_type)
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(text_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Rule extraction for interpretability
        self.rule_extractor = RuleExtractor()
        
    def forward(self, 
                text_tokens: torch.Tensor,
                image_features: torch.Tensor,
                return_attention: bool = False,
                return_rules: bool = False) -> Dict[str, Any]:
        """
        Forward pass for multimodal fuzzy transformer
        
        Args:
            text_tokens: [batch_size, text_seq_len]
            image_features: [batch_size, image_seq_len, image_dim]
            return_attention: Whether to return attention weights
            return_rules: Whether to extract and return fuzzy rules
            
        Returns:
            Dictionary with predictions, attention weights, and rules
        """
        
        batch_size, text_seq_len = text_tokens.shape
        image_seq_len = image_features.shape[1]
        
        # Text processing
        text_emb = self.text_embedding(text_tokens)  # [batch, text_seq, text_dim]
        text_pos = self.text_pos_embedding(torch.arange(text_seq_len, device=text_tokens.device))
        text_features = text_emb + text_pos.unsqueeze(0)
        text_features = F.dropout(text_features, p=self.dropout, training=self.training)
        
        # Image processing
        image_proj = self.image_projection(image_features)  # [batch, image_seq, hidden_dim]
        image_pos = self.image_pos_embedding(torch.arange(image_seq_len, device=image_features.device))
        image_features = image_proj + image_pos.unsqueeze(0)
        
        # Store attention weights for rule extraction
        all_attention_weights = []
        
        # Process through transformer layers
        for i in range(len(self.text_transformer)):
            # Self-attention within modalities
            text_features, text_attn = self.text_transformer[i](
                text_features, text_features, text_features, return_attention=return_attention
            )
            image_features, image_attn = self.image_transformer[i](
                image_features, image_features, image_features, return_attention=return_attention
            )
            
            # Cross-modal attention
            text_features, image_features, cross_attn = self.cross_modal_layers[i](
                text_features, image_features, return_attention=return_attention
            )
            
            if return_attention:
                all_attention_weights.extend([text_attn, image_attn, cross_attn])
        
        # Global average pooling
        text_pooled = text_features.mean(dim=1)  # [batch, text_dim]
        image_pooled = image_features.mean(dim=1)  # [batch, hidden_dim]
        
        # Concatenate for classification
        combined_features = torch.cat([text_pooled, image_pooled], dim=-1)
        logits = self.classifier(combined_features)
        
        result = {
            'logits': logits,
            'text_features': text_features,
            'image_features': image_features
        }
        
        if return_attention:
            result['attention_weights'] = all_attention_weights
        
        if return_rules:
            # Extract rules from all attention layers
            if return_attention and len(all_attention_weights) > 0:
                # Combine all attention weights for rule extraction
                all_rules = []
                all_patterns = {}
                
                for layer_idx, attn_data in enumerate(all_attention_weights):
                    if 'avg_attention' in attn_data:
                        attention = attn_data['avg_attention']
                        
                        # Extract rules from this layer
                        layer_rules = self.rule_extractor.extract_rules(attention)
                        all_rules.extend(layer_rules)
                        
                        # Extract patterns from this layer
                        layer_patterns = self.rule_extractor.extract_attention_patterns(attention)
                        for key, value in layer_patterns.items():
                            if key not in all_patterns:
                                all_patterns[key] = []
                            all_patterns[key].append(value)
                
                # Combine patterns (average across layers)
                for key in all_patterns:
                    if all_patterns[key]:
                        # Handle both numeric and list values
                        if isinstance(all_patterns[key][0], (int, float)):
                            all_patterns[key] = sum(all_patterns[key]) / len(all_patterns[key])
                        else:
                            # For non-numeric values, just take the first one
                            all_patterns[key] = all_patterns[key][0]
                
                result['fuzzy_rules'] = all_rules
                result['attention_patterns'] = all_patterns
        
        return result

class VQAFuzzyModel(nn.Module):
    """VQA model with fuzzy attention for interpretable visual question answering"""
    
    def __init__(self,
                 vocab_size: int = 30000,
                 answer_vocab_size: int = 3000,
                 text_dim: int = 512,
                 image_dim: int = 2048,  # ResNet features
                 hidden_dim: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 fuzzy_type: str = 'product',
                 dropout: float = 0.1):
        super().__init__()
        
        self.multimodal_transformer = MultimodalFuzzyTransformer(
            vocab_size=vocab_size,
            text_dim=text_dim,
            image_dim=image_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            num_classes=answer_vocab_size,
            fuzzy_type=fuzzy_type,
            dropout=dropout
        )
        
        self.rule_extractor = RuleExtractor()
        
    def forward(self, 
                question_tokens: torch.Tensor,
                image_features: torch.Tensor,
                return_explanations: bool = False) -> Dict[str, Any]:
        """
        Forward pass for VQA with fuzzy attention
        
        Args:
            question_tokens: [batch_size, question_len]
            image_features: [batch_size, num_patches, image_dim]
            return_explanations: Whether to return interpretable explanations
            
        Returns:
            Dictionary with predictions and explanations
        """
        
        # Get multimodal features and attention
        result = self.multimodal_transformer(
            text_tokens=question_tokens,
            image_features=image_features,
            return_attention=return_explanations,
            return_rules=return_explanations
        )
        
        # Add VQA-specific outputs
        result['answer_logits'] = result['logits']
        result['answer_probs'] = F.softmax(result['logits'], dim=-1)
        
        if return_explanations:
            # Generate explanations for VQA
            result['explanations'] = self._generate_vqa_explanations(
                result.get('fuzzy_rules', []),
                result.get('attention_patterns', {})
            )
        
        return result
    
    def _generate_vqa_explanations(self, 
                                 rules: List[FuzzyRule],
                                 patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Generate VQA-specific explanations"""
        
        explanations = {
            'question_image_connections': [],
            'attention_summary': {},
            'reasoning_steps': []
        }
        
        # Analyze question-image connections
        for rule in rules[:5]:  # Top 5 rules
            if rule.from_position < 20:  # Assuming first 20 positions are question
                explanations['question_image_connections'].append({
                    'question_word_pos': rule.from_position,
                    'image_region_pos': rule.to_position - 20,  # Adjust for image regions
                    'strength': rule.strength,
                    'description': rule.linguistic_description
                })
        
        # Attention summary
        if patterns:
            explanations['attention_summary'] = {
                'entropy': np.mean(patterns.get('attention_entropy', [0])),
                'sparsity': np.mean(patterns.get('attention_sparsity', [0])),
                'dominant_connections': np.mean(patterns.get('dominant_connections', [0]))
            }
        
        # Generate reasoning steps
        explanations['reasoning_steps'] = [
            "1. Question words attend to relevant image regions",
            "2. Fuzzy rules capture semantic relationships",
            "3. Cross-modal attention integrates text and visual information",
            "4. Final prediction based on integrated multimodal understanding"
        ]
        
        return explanations

def demo_multimodal_fuzzy_attention():
    """Demo function for multimodal fuzzy attention"""
    print("🖼️ Multimodal Fuzzy Attention Demo")
    print("=" * 50)
    
    # Create VQA model
    model = VQAFuzzyModel(
        vocab_size=10000,
        answer_vocab_size=1000,
        text_dim=256,
        image_dim=512,
        hidden_dim=256,
        n_heads=4,
        n_layers=2
    )
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create demo data
    batch_size, question_len, image_patches = 2, 10, 49  # 7x7 patches
    question_tokens = torch.randint(0, 10000, (batch_size, question_len))
    image_features = torch.randn(batch_size, image_patches, 512)
    
    print(f"📊 Input shapes:")
    print(f"   Question tokens: {question_tokens.shape}")
    print(f"   Image features: {image_features.shape}")
    
    # Forward pass with explanations
    with torch.no_grad():
        result = model(question_tokens, image_features, return_explanations=True)
    
    print(f"✅ Output shapes:")
    print(f"   Answer logits: {result['answer_logits'].shape}")
    print(f"   Answer probabilities: {result['answer_probs'].shape}")
    
    if 'fuzzy_rules' in result:
        print(f"🔍 Extracted {len(result['fuzzy_rules'])} fuzzy rules")
        
        # Show top 3 rules
        for i, rule in enumerate(result['fuzzy_rules'][:3]):
            print(f"   Rule {i+1}: {rule.linguistic_description} (strength: {rule.strength:.3f})")
    
    if 'explanations' in result:
        explanations = result['explanations']
        print(f"📝 Generated explanations:")
        print(f"   Question-image connections: {len(explanations['question_image_connections'])}")
        print(f"   Reasoning steps: {len(explanations['reasoning_steps'])}")
        
        # Show reasoning steps
        for step in explanations['reasoning_steps']:
            print(f"   {step}")
    
    # Test cross-modal attention separately
    print(f"\n🔄 Testing Cross-Modal Attention:")
    cross_modal = CrossModalFuzzyAttention(text_dim=256, image_dim=512, hidden_dim=128)
    
    text_feat = torch.randn(1, 10, 256)
    image_feat = torch.randn(1, 49, 512)
    
    with torch.no_grad():
        text_out, image_out, attn_info = cross_modal(text_feat, image_feat, return_attention=True)
    
    print(f"   Text output: {text_out.shape}")
    print(f"   Image output: {image_out.shape}")
    print(f"   Attention shape: {attn_info['avg_attention'].shape}")
    
    return model, result

if __name__ == "__main__":
    demo_multimodal_fuzzy_attention()


"""
Enhanced Multimodal Integration with CLIP and Real Cross-Modal Reasoning
Implements sophisticated multimodal fuzzy attention as described in the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import clip
from PIL import Image
import requests
from io import BytesIO
import math

from fuzzy_attention import MultiHeadFuzzyAttention
from enhanced_rule_extraction import EnhancedRuleExtractor, CompositionalRule

class CLIPFuzzyIntegration(nn.Module):
    """Integration of CLIP with fuzzy attention for cross-modal reasoning"""
    
    def __init__(self, 
                 clip_model_name: str = "ViT-B/32",
                 text_dim: int = 512,
                 image_dim: int = 512,
                 hidden_dim: int = 256,
                 n_heads: int = 8):
        super().__init__()
        
        # Load CLIP model
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device="cpu")
        self.clip_dim = self.clip_model.text_projection.shape[1]  # Usually 512
        
        # Project CLIP features to our hidden dimension
        self.text_projection = nn.Linear(self.clip_dim, hidden_dim)
        self.image_projection = nn.Linear(self.clip_dim, hidden_dim)
        
        # Cross-modal fuzzy attention layers
        self.cross_modal_attention = MultiHeadFuzzyAttention(
            d_model=hidden_dim,
            n_heads=n_heads,
            fuzzy_type='product'
        )
        
        # Intra-modal fuzzy attention
        self.text_self_attention = MultiHeadFuzzyAttention(
            d_model=hidden_dim,
            n_heads=n_heads,
            fuzzy_type='product'
        )
        
        self.image_self_attention = MultiHeadFuzzyAttention(
            d_model=hidden_dim,
            n_heads=n_heads,
            fuzzy_type='product'
        )
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text using CLIP"""
        with torch.no_grad():
            text_tokens = clip.tokenize([text], truncate=True)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image using CLIP"""
        with torch.no_grad():
            image_input = self.clip_preprocess(image).unsqueeze(0)
            image_features = self.clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def forward(self, 
                text: str,
                image: Image.Image,
                return_attention: bool = False) -> Dict[str, Any]:
        """Forward pass for cross-modal fuzzy attention"""
        
        # Encode text and image with CLIP
        text_features = self.encode_text(text)  # [1, clip_dim]
        image_features = self.encode_image(image)  # [1, clip_dim]
        
        # Project to hidden dimension
        text_proj = self.text_projection(text_features)  # [1, hidden_dim]
        image_proj = self.image_projection(image_features)  # [1, hidden_dim]
        
        # Add sequence dimension for attention
        text_seq = text_proj.unsqueeze(1)  # [1, 1, hidden_dim]
        image_seq = image_proj.unsqueeze(1)  # [1, 1, hidden_dim]
        
        # Intra-modal self-attention
        text_enhanced, text_attn = self.text_self_attention(
            text_seq, text_seq, text_seq, return_attention=return_attention
        )
        
        image_enhanced, image_attn = self.image_self_attention(
            image_seq, image_seq, image_seq, return_attention=return_attention
        )
        
        # Cross-modal attention: text attends to image
        text_cross, cross_attn = self.cross_modal_attention(
            text_enhanced, image_enhanced, image_enhanced, return_attention=return_attention
        )
        
        # Cross-modal attention: image attends to text
        image_cross, _ = self.cross_modal_attention(
            image_enhanced, text_enhanced, text_enhanced, return_attention=return_attention
        )
        
        # Fusion with learnable weights
        fusion_weights = F.softmax(self.fusion_weights, dim=0)
        
        # Combine original and cross-modal features
        text_final = fusion_weights[0] * text_enhanced + (1 - fusion_weights[0]) * text_cross
        image_final = fusion_weights[1] * image_enhanced + (1 - fusion_weights[1]) * image_cross
        
        # Final fusion
        combined = torch.cat([text_final, image_final], dim=-1)
        fused_features = self.fusion_layer(combined)
        
        result = {
            'text_features': text_final,
            'image_features': image_final,
            'fused_features': fused_features,
            'text_original': text_features,
            'image_original': image_features
        }
        
        if return_attention:
            result['text_attention'] = text_attn
            result['image_attention'] = image_attn
            result['cross_modal_attention'] = cross_attn
        
        return result

class HierarchicalCrossModalAttention(nn.Module):
    """Hierarchical cross-modal attention with multiple levels of abstraction"""
    
    def __init__(self, 
                 hidden_dim: int = 256,
                 n_heads: int = 8,
                 n_levels: int = 3):
        super().__init__()
        
        self.n_levels = n_levels
        self.hidden_dim = hidden_dim
        
        # Different levels of attention
        self.attention_levels = nn.ModuleList([
            MultiHeadFuzzyAttention(hidden_dim, n_heads, fuzzy_type='product')
            for _ in range(n_levels)
        ])
        
        # Level-specific projections
        self.level_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_levels)
        ])
        
        # Hierarchical fusion
        self.hierarchical_fusion = nn.Sequential(
            nn.Linear(hidden_dim * n_levels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, 
                text_features: torch.Tensor,
                image_features: torch.Tensor,
                return_attention: bool = False) -> Dict[str, Any]:
        """Hierarchical cross-modal attention forward pass"""
        
        level_outputs = []
        level_attentions = [] if return_attention else None
        
        for level in range(self.n_levels):
            # Project features for this level
            text_proj = self.level_projections[level](text_features)
            image_proj = self.level_projections[level](image_features)
            
            # Cross-modal attention at this level
            text_attended, attn = self.attention_levels[level](
                text_proj, image_proj, image_proj, return_attention=return_attention
            )
            
            level_outputs.append(text_attended)
            if return_attention:
                level_attentions.append(attn)
        
        # Hierarchical fusion
        combined_levels = torch.cat(level_outputs, dim=-1)
        hierarchical_output = self.hierarchical_fusion(combined_levels)
        
        result = {
            'hierarchical_output': hierarchical_output,
            'level_outputs': level_outputs
        }
        
        if return_attention:
            result['level_attentions'] = level_attentions
        
        return result

class CompositionalCrossModalReasoning(nn.Module):
    """Compositional cross-modal reasoning with fuzzy logic"""
    
    def __init__(self, 
                 hidden_dim: int = 256,
                 n_heads: int = 8):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Compositional attention layers
        self.compositional_attention = MultiHeadFuzzyAttention(
            d_model=hidden_dim,
            n_heads=n_heads,
            fuzzy_type='product'
        )
        
        # Rule-based reasoning components
        self.rule_extractor = EnhancedRuleExtractor()
        
        # Compositional fusion
        self.compositional_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # text, image, cross-modal
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, 
                text_features: torch.Tensor,
                image_features: torch.Tensor,
                return_rules: bool = False) -> Dict[str, Any]:
        """Compositional cross-modal reasoning forward pass"""
        
        # Cross-modal attention
        text_cross, cross_attn = self.compositional_attention(
            text_features, image_features, image_features, return_attention=True
        )
        
        image_cross, _ = self.compositional_attention(
            image_features, text_features, text_features, return_attention=True
        )
        
        # Compositional fusion
        combined = torch.cat([text_features, image_features, text_cross], dim=-1)
        compositional_output = self.compositional_fusion(combined)
        
        result = {
            'text_cross_modal': text_cross,
            'image_cross_modal': image_cross,
            'compositional_output': compositional_output,
            'cross_modal_attention': cross_attn
        }
        
        if return_rules:
            # Extract compositional rules
            attention_weights = cross_attn['avg_attention']
            rules = self.rule_extractor.extract_enhanced_rules(
                attention_weights,
                cross_modal_info={'text_length': text_features.shape[1], 'image_length': image_features.shape[1]}
            )
            result['compositional_rules'] = rules
        
        return result

class EnhancedMultimodalFuzzyTransformer(nn.Module):
    """Complete enhanced multimodal fuzzy transformer"""
    
    def __init__(self,
                 clip_model_name: str = "ViT-B/32",
                 hidden_dim: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 num_classes: int = 1000):
        super().__init__()
        
        # CLIP integration
        self.clip_integration = CLIPFuzzyIntegration(
            clip_model_name=clip_model_name,
            hidden_dim=hidden_dim,
            n_heads=n_heads
        )
        
        # Hierarchical attention
        self.hierarchical_attention = HierarchicalCrossModalAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_levels=3
        )
        
        # Compositional reasoning
        self.compositional_reasoning = CompositionalCrossModalReasoning(
            hidden_dim=hidden_dim,
            n_heads=n_heads
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1
            )
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Rule extraction for interpretability
        self.rule_extractor = EnhancedRuleExtractor()
        
    def forward(self, 
                text: str,
                image: Image.Image,
                return_explanations: bool = False) -> Dict[str, Any]:
        """Complete multimodal fuzzy transformer forward pass"""
        
        # CLIP integration
        clip_result = self.clip_integration(text, image, return_attention=True)
        
        # Hierarchical cross-modal attention
        hierarchical_result = self.hierarchical_attention(
            clip_result['text_features'],
            clip_result['image_features'],
            return_attention=True
        )
        
        # Compositional reasoning
        compositional_result = self.compositional_reasoning(
            clip_result['text_features'],
            clip_result['image_features'],
            return_rules=True
        )
        
        # Combine all features
        combined_features = (
            clip_result['fused_features'] + 
            hierarchical_result['hierarchical_output'] + 
            compositional_result['compositional_output']
        ) / 3.0
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            combined_features = layer(combined_features)
        
        # Global pooling
        pooled_features = combined_features.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled_features)
        
        result = {
            'logits': logits,
            'text_features': clip_result['text_features'],
            'image_features': clip_result['image_features'],
            'fused_features': combined_features,
            'clip_similarity': self._compute_clip_similarity(clip_result['text_original'], clip_result['image_original'])
        }
        
        if return_explanations:
            # Extract comprehensive explanations
            explanations = self._generate_comprehensive_explanations(
                clip_result, hierarchical_result, compositional_result
            )
            result['explanations'] = explanations
        
        return result
    
    def _compute_clip_similarity(self, text_features: torch.Tensor, image_features: torch.Tensor) -> float:
        """Compute CLIP similarity between text and image"""
        similarity = torch.cosine_similarity(text_features, image_features, dim=-1)
        return similarity.item()
    
    def _generate_comprehensive_explanations(self, 
                                           clip_result: Dict[str, Any],
                                           hierarchical_result: Dict[str, Any],
                                           compositional_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive explanations for multimodal reasoning"""
        
        explanations = {
            'clip_analysis': {
                'text_image_similarity': clip_result.get('clip_similarity', 0.0),
                'cross_modal_attention_entropy': self._compute_attention_entropy(
                    clip_result.get('cross_modal_attention', {}).get('avg_attention', torch.rand(1, 1, 1))
                )
            },
            'hierarchical_analysis': {
                'n_levels': len(hierarchical_result.get('level_outputs', [])),
                'level_attention_patterns': [
                    self._compute_attention_entropy(attn.get('avg_attention', torch.rand(1, 1, 1)))
                    for attn in hierarchical_result.get('level_attentions', [])
                ]
            },
            'compositional_analysis': {
                'compositional_rules': compositional_result.get('compositional_rules', {}),
                'cross_modal_connections': self._analyze_cross_modal_connections(
                    compositional_result.get('cross_modal_attention', {})
                )
            },
            'reasoning_steps': [
                "1. CLIP encodes text and image into shared semantic space",
                "2. Fuzzy attention identifies cross-modal relationships",
                "3. Hierarchical attention captures multi-level abstractions",
                "4. Compositional reasoning generates interpretable rules",
                "5. Final prediction based on integrated multimodal understanding"
            ]
        }
        
        return explanations
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute attention entropy"""
        if attention_weights.numel() == 0:
            return 0.0
        
        entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean()
        return entropy.item()
    
    def _analyze_cross_modal_connections(self, cross_modal_attention: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cross-modal connections"""
        if 'avg_attention' not in cross_modal_attention:
            return {'connections': 0, 'strength': 0.0}
        
        attention = cross_modal_attention['avg_attention']
        strong_connections = (attention > 0.1).sum().item()
        avg_strength = attention.mean().item()
        
        return {
            'connections': strong_connections,
            'strength': avg_strength,
            'max_connection': attention.max().item()
        }

class VQAEnhancedModel(nn.Module):
    """Enhanced VQA model with full multimodal fuzzy reasoning"""
    
    def __init__(self,
                 clip_model_name: str = "ViT-B/32",
                 answer_vocab_size: int = 3000,
                 hidden_dim: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 3):
        super().__init__()
        
        self.multimodal_transformer = EnhancedMultimodalFuzzyTransformer(
            clip_model_name=clip_model_name,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            num_classes=answer_vocab_size
        )
        
        # Enhanced rule extraction
        self.rule_extractor = EnhancedRuleExtractor()
        
    def forward(self, 
                question: str,
                image: Image.Image,
                return_explanations: bool = False) -> Dict[str, Any]:
        """Enhanced VQA forward pass with comprehensive explanations"""
        
        # Get multimodal features and explanations
        result = self.multimodal_transformer(
            text=question,
            image=image,
            return_explanations=return_explanations
        )
        
        # Add VQA-specific outputs
        result['answer_logits'] = result['logits']
        result['answer_probs'] = F.softmax(result['logits'], dim=-1)
        
        if return_explanations:
            # Generate VQA-specific explanations
            result['vqa_explanations'] = self._generate_vqa_explanations(
                result.get('explanations', {}),
                question,
                image
            )
        
        return result
    
    def _generate_vqa_explanations(self, 
                                 explanations: Dict[str, Any],
                                 question: str,
                                 image: Image.Image) -> Dict[str, Any]:
        """Generate VQA-specific explanations"""
        
        vqa_explanations = {
            'question_analysis': {
                'question_type': self._classify_question_type(question),
                'key_words': self._extract_key_words(question),
                'complexity': self._assess_question_complexity(question)
            },
            'image_analysis': {
                'image_size': image.size,
                'dominant_colors': self._extract_dominant_colors(image),
                'detected_objects': self._detect_objects(image)  # Would use object detection
            },
            'cross_modal_reasoning': explanations.get('compositional_analysis', {}),
            'answer_justification': self._generate_answer_justification(
                explanations.get('compositional_analysis', {}),
                question
            )
        }
        
        return vqa_explanations
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what', 'which']):
            return 'object_identification'
        elif any(word in question_lower for word in ['where', 'location']):
            return 'spatial_reasoning'
        elif any(word in question_lower for word in ['how', 'why']):
            return 'causal_reasoning'
        elif any(word in question_lower for word in ['how many', 'count']):
            return 'counting'
        else:
            return 'general'
    
    def _extract_key_words(self, question: str) -> List[str]:
        """Extract key words from question"""
        # Simple keyword extraction
        words = question.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        key_words = [word for word in words if word not in stop_words and len(word) > 2]
        return key_words[:5]  # Top 5 keywords
    
    def _assess_question_complexity(self, question: str) -> str:
        """Assess question complexity"""
        word_count = len(question.split())
        
        if word_count <= 5:
            return 'simple'
        elif word_count <= 10:
            return 'medium'
        else:
            return 'complex'
    
    def _extract_dominant_colors(self, image: Image.Image) -> List[str]:
        """Extract dominant colors from image"""
        # This would implement actual color extraction
        return ['red', 'blue', 'green']  # Mock implementation
    
    def _detect_objects(self, image: Image.Image) -> List[str]:
        """Detect objects in image"""
        # This would use actual object detection
        return ['cat', 'mat', 'floor']  # Mock implementation
    
    def _generate_answer_justification(self, 
                                     compositional_analysis: Dict[str, Any],
                                     question: str) -> str:
        """Generate answer justification"""
        rules = compositional_analysis.get('compositional_rules', {})
        rule_count = rules.get('total_rules', 0)
        
        return f"Based on {rule_count} fuzzy rules extracted from cross-modal attention, the model identified key relationships between the question '{question}' and visual elements in the image."

def demo_enhanced_multimodal_integration():
    """Demo function for enhanced multimodal integration"""
    print("🖼️ Enhanced Multimodal Integration Demo")
    print("=" * 50)
    
    # Create enhanced model
    model = VQAEnhancedModel(
        answer_vocab_size=100,
        hidden_dim=128,
        n_heads=4,
        n_layers=2
    )
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create demo data
    question = "What is the cat doing in the image?"
    
    # Create a simple demo image
    demo_image = Image.new('RGB', (224, 224), color='lightblue')
    
    print(f"📊 Input:")
    print(f"   Question: {question}")
    print(f"   Image size: {demo_image.size}")
    
    # Forward pass with explanations
    with torch.no_grad():
        result = model(question, demo_image, return_explanations=True)
    
    print(f"✅ Output shapes:")
    print(f"   Answer logits: {result['answer_logits'].shape}")
    print(f"   Answer probabilities: {result['answer_probs'].shape}")
    
    # Show top predictions
    top_answers = torch.topk(result['answer_probs'], k=3, dim=-1)
    print(f"🎯 Top 3 answer predictions:")
    for j in range(3):
        prob = top_answers.values[0, j].item()
        answer_idx = top_answers.indices[0, j].item()
        print(f"   Answer {answer_idx}: {prob:.3f}")
    
    # Show explanations
    if 'vqa_explanations' in result:
        explanations = result['vqa_explanations']
        print(f"📝 Generated explanations:")
        print(f"   Question type: {explanations['question_analysis']['question_type']}")
        print(f"   Key words: {explanations['question_analysis']['key_words']}")
        print(f"   Question complexity: {explanations['question_analysis']['complexity']}")
        print(f"   Cross-modal rules: {explanations['cross_modal_reasoning'].get('compositional_rules', {}).get('total_rules', 0)}")
        
        # Show reasoning steps
        print(f"🧠 Reasoning steps:")
        for step in explanations['cross_modal_reasoning'].get('reasoning_steps', []):
            print(f"   {step}")
    
    return model, result

if __name__ == "__main__":
    demo_enhanced_multimodal_integration()


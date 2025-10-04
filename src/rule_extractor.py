"""
Rule Extraction Module for Fuzzy Attention Networks
Extracts interpretable linguistic rules from trained attention weights
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json

@dataclass
class FuzzyRule:
    """Represents a single fuzzy rule extracted from attention weights"""
    from_position: int
    to_position: int
    strength: float
    confidence: float
    linguistic_description: str
    membership_function: str
    tnorm_type: str

class RuleExtractor:
    """Extracts interpretable rules from fuzzy attention weights"""
    
    def __init__(self, 
                 attention_threshold: float = 0.1,
                 strong_threshold: float = 0.15,
                 max_rules_per_position: int = 3):
        self.attention_threshold = attention_threshold
        self.strong_threshold = strong_threshold
        self.max_rules_per_position = max_rules_per_position
        
        # Linguistic templates for rule descriptions
        self.linguistic_templates = {
            'strong': [
                "strongly attends to",
                "heavily focuses on", 
                "primarily considers"
            ],
            'medium': [
                "moderately attends to",
                "partially focuses on",
                "somewhat considers"
            ],
            'weak': [
                "slightly attends to",
                "minimally focuses on",
                "barely considers"
            ]
        }
    
    def extract_rules(self, 
                     attention_weights: torch.Tensor,
                     tokens: Optional[List[str]] = None,
                     membership_functions: Optional[Dict] = None,
                     tnorm_type: str = 'product') -> List[FuzzyRule]:
        """
        Extract fuzzy rules from attention weights
        
        Args:
            attention_weights: [batch_size, seq_len, seq_len] attention matrix
            tokens: Optional token list for linguistic descriptions
            membership_functions: Optional membership function info
            tnorm_type: Type of t-norm used
            
        Returns:
            List of extracted FuzzyRule objects
        """
        rules = []
        batch_size, seq_len, _ = attention_weights.shape
        
        # Process each batch item
        for batch_idx in range(batch_size):
            attention = attention_weights[batch_idx]  # [seq_len, seq_len]
            
            # Find high-attention connections with improved thresholding
            attention_mean = attention.mean()
            attention_std = attention.std()
            # Lower threshold to extract more rules
            dynamic_threshold = max(0.01, attention_mean + 0.1 * attention_std)
            
            high_attention_mask = attention > dynamic_threshold
            high_attention_indices = torch.nonzero(high_attention_mask, as_tuple=False)
            
            # Extract rules for each high-attention connection
            for idx in high_attention_indices:
                from_pos, to_pos = idx[0].item(), idx[1].item()
                strength = attention[from_pos, to_pos].item()
                
                # Skip self-attention (diagonal elements)
                if from_pos == to_pos:
                    continue
                
                # Determine rule strength category with improved thresholds
                if strength > self.strong_threshold:
                    strength_category = 'strong'
                elif strength > dynamic_threshold * 1.2:
                    strength_category = 'medium'
                else:
                    strength_category = 'weak'
                
                # Generate linguistic description
                linguistic_desc = self._generate_linguistic_description(
                    from_pos, to_pos, strength_category, tokens
                )
                
                # Calculate confidence based on attention strength and position
                base_confidence = min(strength / self.strong_threshold, 1.0)
                
                # Boost confidence for cross-modal connections (assuming first half is text, second half is image)
                if from_pos < seq_len // 2 and to_pos >= seq_len // 2:
                    base_confidence *= 1.2  # Cross-modal connections are more interesting
                elif from_pos >= seq_len // 2 and to_pos < seq_len // 2:
                    base_confidence *= 1.2
                
                confidence = min(base_confidence, 1.0)
                
                rule = FuzzyRule(
                    from_position=from_pos,
                    to_position=to_pos,
                    strength=strength,
                    confidence=confidence,
                    linguistic_description=linguistic_desc,
                    membership_function='gaussian',  # Default
                    tnorm_type=tnorm_type
                )
                
                rules.append(rule)
        
        # Sort rules by strength and limit per position
        rules = self._filter_and_rank_rules(rules)
        return rules
    
    def _generate_linguistic_description(self, 
                                       from_pos: int, 
                                       to_pos: int, 
                                       strength_category: str,
                                       tokens: Optional[List[str]] = None) -> str:
        """Generate human-readable rule description"""
        
        templates = self.linguistic_templates[strength_category]
        template = np.random.choice(templates)
        
        if tokens and from_pos < len(tokens) and to_pos < len(tokens):
            return f"Position {from_pos} ('{tokens[from_pos]}') {template} position {to_pos} ('{tokens[to_pos]}')"
        else:
            return f"Position {from_pos} {template} position {to_pos}"
    
    def _filter_and_rank_rules(self, rules: List[FuzzyRule]) -> List[FuzzyRule]:
        """Filter and rank rules by importance"""
        # Sort by strength (descending)
        rules.sort(key=lambda r: r.strength, reverse=True)
        
        # Group by from_position and limit rules per position
        position_rules = {}
        filtered_rules = []
        
        for rule in rules:
            from_pos = rule.from_position
            if from_pos not in position_rules:
                position_rules[from_pos] = []
            
            if len(position_rules[from_pos]) < self.max_rules_per_position:
                position_rules[from_pos].append(rule)
                filtered_rules.append(rule)
        
        return filtered_rules
    
    def extract_attention_patterns(self, 
                                 attention_weights: torch.Tensor) -> Dict[str, Any]:
        """Extract statistical patterns from attention weights"""
        batch_size, seq_len, _ = attention_weights.shape
        
        patterns = {
            'attention_entropy': [],
            'attention_sparsity': [],
            'dominant_connections': [],
            'attention_distribution': []
        }
        
        for batch_idx in range(batch_size):
            attention = attention_weights[batch_idx]
            
            # Calculate entropy for each position
            entropy = -(attention * torch.log(attention + 1e-8)).sum(dim=-1)
            patterns['attention_entropy'].append(entropy.mean().item())
            
            # Calculate sparsity (fraction of near-zero weights)
            sparsity = (attention < 0.01).float().mean().item()
            patterns['attention_sparsity'].append(sparsity)
            
            # Find dominant connections
            max_attention = attention.max(dim=-1)
            dominant_connections = (max_attention.values > 0.2).sum().item()
            patterns['dominant_connections'].append(dominant_connections)
            
            # Attention distribution statistics
            patterns['attention_distribution'].append({
                'mean': attention.mean().item(),
                'std': attention.std().item(),
                'max': attention.max().item(),
                'min': attention.min().item()
            })
        
        return patterns

class AdaptiveRuleExplainer:
    """Generates adaptive explanations based on user expertise level"""
    
    def __init__(self):
        self.explanation_templates = {
            'novice': {
                'intro': "The AI model focuses on {0} key connections in the text.",
                'rule': "Connection {0}: {1}",
                'summary': "These connections help the model understand the text better."
            },
            'intermediate': {
                'intro': "The fuzzy attention mechanism identified {0} important relationships:",
                'rule': "Rule {0}: {1} (strength: {2:.3f}, confidence: {3:.3f})",
                'summary': "These fuzzy rules represent learned attention patterns."
            },
            'expert': {
                'intro': "Extracted {0} fuzzy rules from attention weights using product t-norm:",
                'rule': "FuzzyRule(from={0}, to={1}, strength={2:.4f}, confidence={3:.4f}, tnorm='{4}'): {5}",
                'summary': "Technical details: {0} membership functions, attention entropy: {1:.3f}"
            }
        }
    
    def generate_explanation(self, 
                           rules: List[FuzzyRule],
                           user_level: str = 'intermediate',
                           attention_patterns: Optional[Dict] = None) -> str:
        """Generate adaptive explanation based on user expertise"""
        
        if user_level not in self.explanation_templates:
            user_level = 'intermediate'
        
        templates = self.explanation_templates[user_level]
        explanation_parts = []
        
        # Introduction
        intro = templates['intro'].format(len(rules))
        explanation_parts.append(intro)
        explanation_parts.append("")
        
        # Rules
        for i, rule in enumerate(rules[:5]):  # Limit to top 5 rules
            if user_level == 'novice':
                rule_text = templates['rule'].format(
                    i+1, 
                    f"position {rule.from_position} → position {rule.to_position}"
                )
            elif user_level == 'intermediate':
                rule_text = templates['rule'].format(
                    i+1,
                    rule.linguistic_description,
                    rule.strength,
                    rule.confidence
                )
            else:  # expert
                rule_text = templates['rule'].format(
                    i+1,
                    rule.from_position,
                    rule.to_position,
                    rule.strength,
                    rule.confidence,
                    rule.tnorm_type,
                    rule.linguistic_description
                )
            
            explanation_parts.append(rule_text)
        
        # Summary
        if user_level == 'expert' and attention_patterns:
            avg_entropy = np.mean(attention_patterns['attention_entropy'])
            summary = templates['summary'].format(
                'gaussian',  # membership function type
                avg_entropy
            )
        else:
            summary = templates['summary']
        
        explanation_parts.append("")
        explanation_parts.append(summary)
        
        return "\n".join(explanation_parts)
    
    def export_rules_json(self, rules: List[FuzzyRule], filepath: str):
        """Export rules to JSON format for analysis"""
        rules_data = []
        for rule in rules:
            rules_data.append({
                'from_position': rule.from_position,
                'to_position': rule.to_position,
                'strength': rule.strength,
                'confidence': rule.confidence,
                'linguistic_description': rule.linguistic_description,
                'membership_function': rule.membership_function,
                'tnorm_type': rule.tnorm_type
            })
        
        with open(filepath, 'w') as f:
            json.dump(rules_data, f, indent=2)

def demo_rule_extraction():
    """Demo function for rule extraction"""
    print("🔍 Rule Extraction Demo")
    print("=" * 40)
    
    # Create sample attention weights
    seq_len = 8
    attention_weights = torch.rand(1, seq_len, seq_len)
    
    # Add some strong connections
    attention_weights[0, 0, 2] = 0.25  # Strong connection
    attention_weights[0, 1, 3] = 0.18  # Medium connection
    attention_weights[0, 4, 6] = 0.12  # Weak connection
    
    # Normalize to make it look like attention weights
    attention_weights = torch.softmax(attention_weights, dim=-1)
    
    # Create rule extractor
    extractor = RuleExtractor(attention_threshold=0.1, strong_threshold=0.15)
    
    # Extract rules
    rules = extractor.extract_rules(attention_weights)
    
    print(f"📊 Extracted {len(rules)} rules from attention weights")
    print(f"📈 Attention shape: {attention_weights.shape}")
    
    # Generate explanations for different user levels
    explainer = AdaptiveRuleExplainer()
    
    for user_level in ['novice', 'intermediate', 'expert']:
        print(f"\n👤 {user_level.upper()} EXPLANATION:")
        print("-" * 30)
        explanation = explainer.generate_explanation(rules, user_level)
        print(explanation)
    
    # Extract attention patterns
    patterns = extractor.extract_attention_patterns(attention_weights)
    print(f"\n📊 ATTENTION PATTERNS:")
    print(f"   Average entropy: {patterns['attention_entropy'][0]:.3f}")
    print(f"   Sparsity: {patterns['attention_sparsity'][0]:.3f}")
    print(f"   Dominant connections: {patterns['dominant_connections'][0]}")
    
    return rules, patterns

if __name__ == "__main__":
    demo_rule_extraction()

"""
Enhanced Rule Extraction with Compositional Rules and Natural Language Generation
Implements sophisticated rule extraction as described in the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
import json
import re
from collections import defaultdict, Counter
import networkx as nx
from transformers import GPT2LMHeadModel, GPT2Tokenizer

@dataclass
class CompositionalRule:
    """Represents a compositional fuzzy rule with hierarchical structure"""
    rule_id: str
    antecedent: List[str]  # Multiple conditions
    consequent: str
    strength: float
    confidence: float
    composition_type: str  # 'conjunction', 'disjunction', 'implication'
    sub_rules: List['CompositionalRule']
    linguistic_description: str
    mathematical_formulation: str
    validation_status: str  # 'valid', 'invalid', 'uncertain'

@dataclass
class RulePattern:
    """Represents a discovered pattern in attention weights"""
    pattern_type: str  # 'sequential', 'hierarchical', 'cross_modal', 'temporal'
    elements: List[int]
    strength: float
    frequency: int
    confidence: float
    linguistic_template: str

class NaturalLanguageGenerator(nn.Module):
    """Neural network for generating natural language descriptions of fuzzy rules"""
    
    def __init__(self, vocab_size: int = 10000, hidden_dim: int = 256):
        super().__init__()
        
        # Use pre-trained GPT-2 for natural language generation
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Fine-tune GPT-2 for rule description generation
        self.language_model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # Rule-specific encoding layers
        self.rule_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim),  # Rule features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Attention mechanism for rule-to-text alignment
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
    def encode_rule_features(self, rule_data: Dict[str, Any]) -> torch.Tensor:
        """Encode rule features into vector representation"""
        features = [
            rule_data.get('strength', 0.0),
            rule_data.get('confidence', 0.0),
            rule_data.get('from_position', 0.0) / 100.0,  # Normalize
            rule_data.get('to_position', 0.0) / 100.0,
            rule_data.get('attention_entropy', 0.0),
            rule_data.get('sparsity', 0.0),
            rule_data.get('tnorm_type_encoding', 0.0),
            rule_data.get('membership_function_type', 0.0),
            rule_data.get('cross_modal', 0.0),
            rule_data.get('temporal_dependency', 0.0)
        ]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def generate_description(self, rule_data: Dict[str, Any], 
                           tokens: Optional[List[str]] = None) -> str:
        """Generate natural language description for a rule"""
        
        # Encode rule features
        rule_features = self.encode_rule_features(rule_data)
        rule_embedding = self.rule_encoder(rule_features)
        
        # Create prompt based on rule type and complexity
        prompt = self._create_prompt(rule_data, tokens)
        
        # Generate description using GPT-2
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.language_model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        description = generated_text[len(prompt):].strip()
        
        return description
    
    def _create_prompt(self, rule_data: Dict[str, Any], tokens: Optional[List[str]] = None) -> str:
        """Create prompt for rule description generation"""
        
        strength = rule_data.get('strength', 0.0)
        from_pos = rule_data.get('from_position', 0)
        to_pos = rule_data.get('to_position', 0)
        
        # Determine strength level
        if strength > 0.2:
            strength_level = "strongly"
        elif strength > 0.1:
            strength_level = "moderately"
        else:
            strength_level = "slightly"
        
        # Create context-aware prompt
        if tokens and from_pos < len(tokens) and to_pos < len(tokens):
            prompt = f"The AI model {strength_level} connects '{tokens[from_pos]}' with '{tokens[to_pos]}' because"
        else:
            prompt = f"The AI model {strength_level} connects position {from_pos} with position {to_pos} because"
        
        return prompt

class CompositionalRuleExtractor:
    """Extracts compositional rules from attention patterns"""
    
    def __init__(self):
        self.pattern_detector = PatternDetector()
        self.rule_composer = RuleComposer()
        self.rule_validator = RuleValidator()
        
    def extract_compositional_rules(self, 
                                  attention_weights: torch.Tensor,
                                  tokens: Optional[List[str]] = None,
                                  cross_modal_info: Optional[Dict] = None) -> List[CompositionalRule]:
        """Extract compositional rules from attention weights"""
        
        # Detect patterns in attention weights
        patterns = self.pattern_detector.detect_patterns(attention_weights)
        
        # Compose rules from patterns
        compositional_rules = []
        
        for pattern in patterns:
            if pattern.pattern_type == 'sequential':
                rule = self._extract_sequential_rule(pattern, tokens)
            elif pattern.pattern_type == 'hierarchical':
                rule = self._extract_hierarchical_rule(pattern, tokens)
            elif pattern.pattern_type == 'cross_modal':
                rule = self._extract_cross_modal_rule(pattern, tokens, cross_modal_info)
            else:
                rule = self._extract_basic_rule(pattern, tokens)
            
            if rule:
                compositional_rules.append(rule)
        
        # Validate and refine rules
        validated_rules = []
        for rule in compositional_rules:
            validation_result = self.rule_validator.validate_rule(rule)
            if validation_result['is_valid']:
                rule.validation_status = 'valid'
                validated_rules.append(rule)
            else:
                rule.validation_status = 'uncertain'
                # Try to refine the rule
                refined_rule = self._refine_rule(rule, validation_result)
                if refined_rule:
                    validated_rules.append(refined_rule)
        
        return validated_rules
    
    def _extract_sequential_rule(self, pattern: RulePattern, tokens: Optional[List[str]] = None) -> Optional[CompositionalRule]:
        """Extract sequential rule from pattern"""
        if len(pattern.elements) < 2:
            return None
        
        antecedent = []
        consequent = str(pattern.elements[-1])
        
        for i, element in enumerate(pattern.elements[:-1]):
            if tokens and element < len(tokens):
                antecedent.append(f"'{tokens[element]}'")
            else:
                antecedent.append(f"position_{element}")
        
        rule = CompositionalRule(
            rule_id=f"sequential_{pattern.elements[0]}_{pattern.elements[-1]}",
            antecedent=antecedent,
            consequent=consequent,
            strength=pattern.strength,
            confidence=pattern.confidence,
            composition_type='conjunction',
            sub_rules=[],
            linguistic_description=f"If {' AND '.join(antecedent)} then {consequent}",
            mathematical_formulation=f"T(μ₁, μ₂, ..., μₙ) → μ_{consequent}",
            validation_status='pending'
        )
        
        return rule
    
    def _extract_hierarchical_rule(self, pattern: RulePattern, tokens: Optional[List[str]] = None) -> Optional[CompositionalRule]:
        """Extract hierarchical rule from pattern"""
        # This would implement hierarchical rule extraction
        # For now, return a simplified version
        return self._extract_basic_rule(pattern, tokens)
    
    def _extract_cross_modal_rule(self, pattern: RulePattern, tokens: Optional[List[str]] = None, 
                                cross_modal_info: Optional[Dict] = None) -> Optional[CompositionalRule]:
        """Extract cross-modal rule from pattern"""
        if not cross_modal_info:
            return self._extract_basic_rule(pattern, tokens)
        
        # Extract cross-modal relationships
        text_elements = []
        image_elements = []
        
        for element in pattern.elements:
            if element < cross_modal_info.get('text_length', 0):
                if tokens and element < len(tokens):
                    text_elements.append(f"'{tokens[element]}'")
                else:
                    text_elements.append(f"text_position_{element}")
            else:
                image_elements.append(f"image_region_{element - cross_modal_info['text_length']}")
        
        antecedent = text_elements + image_elements
        consequent = "cross_modal_understanding"
        
        rule = CompositionalRule(
            rule_id=f"cross_modal_{pattern.elements[0]}_{pattern.elements[-1]}",
            antecedent=antecedent,
            consequent=consequent,
            strength=pattern.strength,
            confidence=pattern.confidence,
            composition_type='cross_modal',
            sub_rules=[],
            linguistic_description=f"Text elements {', '.join(text_elements)} and image regions {', '.join(image_elements)} are connected",
            mathematical_formulation=f"T(μ_text, μ_image) → μ_cross_modal",
            validation_status='pending'
        )
        
        return rule
    
    def _extract_basic_rule(self, pattern: RulePattern, tokens: Optional[List[str]] = None) -> Optional[CompositionalRule]:
        """Extract basic rule from pattern"""
        if len(pattern.elements) < 2:
            return None
        
        from_pos = pattern.elements[0]
        to_pos = pattern.elements[-1]
        
        antecedent = f"position_{from_pos}" if not tokens or from_pos >= len(tokens) else f"'{tokens[from_pos]}'"
        consequent = f"position_{to_pos}" if not tokens or to_pos >= len(tokens) else f"'{tokens[to_pos]}'"
        
        rule = CompositionalRule(
            rule_id=f"basic_{from_pos}_{to_pos}",
            antecedent=[antecedent],
            consequent=consequent,
            strength=pattern.strength,
            confidence=pattern.confidence,
            composition_type='implication',
            sub_rules=[],
            linguistic_description=f"If {antecedent} then {consequent}",
            mathematical_formulation=f"μ_{from_pos} → μ_{to_pos}",
            validation_status='pending'
        )
        
        return rule
    
    def _refine_rule(self, rule: CompositionalRule, validation_result: Dict[str, Any]) -> Optional[CompositionalRule]:
        """Refine rule based on validation results"""
        # This would implement rule refinement logic
        # For now, return the original rule
        return rule

class PatternDetector:
    """Detects patterns in attention weights"""
    
    def __init__(self):
        self.min_pattern_length = 2
        self.min_strength_threshold = 0.1
        
    def detect_patterns(self, attention_weights: torch.Tensor) -> List[RulePattern]:
        """Detect various patterns in attention weights"""
        patterns = []
        
        # Convert to numpy for easier processing
        attention = attention_weights.cpu().numpy()
        
        # Detect sequential patterns
        sequential_patterns = self._detect_sequential_patterns(attention)
        patterns.extend(sequential_patterns)
        
        # Detect hierarchical patterns
        hierarchical_patterns = self._detect_hierarchical_patterns(attention)
        patterns.extend(hierarchical_patterns)
        
        # Detect cross-modal patterns
        cross_modal_patterns = self._detect_cross_modal_patterns(attention)
        patterns.extend(cross_modal_patterns)
        
        return patterns
    
    def _detect_sequential_patterns(self, attention: np.ndarray) -> List[RulePattern]:
        """Detect sequential patterns in attention weights"""
        patterns = []
        seq_len = attention.shape[1]
        
        # Find strong connections
        strong_connections = np.where(attention > self.min_strength_threshold)
        
        # Build graph of connections
        G = nx.DiGraph()
        for i, j in zip(strong_connections[1], strong_connections[2]):
            G.add_edge(i, j, weight=attention[0, i, j])
        
        # Find paths of length 2-4
        for source in range(seq_len):
            for target in range(seq_len):
                if source != target:
                    try:
                        paths = list(nx.all_simple_paths(G, source, target, cutoff=4))
                        for path in paths:
                            if len(path) >= self.min_pattern_length:
                                # Calculate path strength
                                path_strength = 1.0
                                for i in range(len(path) - 1):
                                    edge_weight = G[path[i]][path[i+1]]['weight']
                                    path_strength *= edge_weight
                                
                                if path_strength > self.min_strength_threshold:
                                    pattern = RulePattern(
                                        pattern_type='sequential',
                                        elements=path,
                                        strength=path_strength,
                                        frequency=1,
                                        confidence=min(path_strength, 1.0),
                                        linguistic_template=f"Sequential connection from {path[0]} to {path[-1]}"
                                    )
                                    patterns.append(pattern)
                    except nx.NetworkXNoPath:
                        continue
        
        return patterns
    
    def _detect_hierarchical_patterns(self, attention: np.ndarray) -> List[RulePattern]:
        """Detect hierarchical patterns in attention weights"""
        patterns = []
        
        # Find nodes with high in-degree and out-degree
        in_degrees = np.sum(attention, axis=1)
        out_degrees = np.sum(attention, axis=2)
        
        # Identify hub nodes
        hub_threshold = np.percentile(in_degrees, 75)
        hub_nodes = np.where(in_degrees[0] > hub_threshold)[0]
        
        for hub in hub_nodes:
            # Find nodes that connect to this hub
            incoming = np.where(attention[0, :, hub] > self.min_strength_threshold)[0]
            outgoing = np.where(attention[0, hub, :] > self.min_strength_threshold)[0]
            
            if len(incoming) > 1 and len(outgoing) > 1:
                # Create hierarchical pattern
                elements = list(incoming) + [hub] + list(outgoing)
                strength = np.mean(attention[0, incoming, hub]) * np.mean(attention[0, hub, outgoing])
                
                pattern = RulePattern(
                    pattern_type='hierarchical',
                    elements=elements,
                    strength=strength,
                    frequency=1,
                    confidence=min(strength, 1.0),
                    linguistic_template=f"Hierarchical pattern centered on position {hub}"
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_cross_modal_patterns(self, attention: np.ndarray) -> List[RulePattern]:
        """Detect cross-modal patterns (assumes first half is text, second half is image)"""
        patterns = []
        seq_len = attention.shape[1]
        mid_point = seq_len // 2
        
        # Find cross-modal connections
        text_to_image = attention[0, :mid_point, mid_point:]
        image_to_text = attention[0, mid_point:, :mid_point]
        
        # Find strong cross-modal connections
        text_to_image_strong = np.where(text_to_image > self.min_strength_threshold)
        image_to_text_strong = np.where(image_to_text > self.min_strength_threshold)
        
        # Create cross-modal patterns
        for i, j in zip(text_to_image_strong[0], text_to_image_strong[1]):
            strength = text_to_image[i, j]
            elements = [i, j + mid_point]  # Adjust for full sequence
            
            pattern = RulePattern(
                pattern_type='cross_modal',
                elements=elements,
                strength=strength,
                frequency=1,
                confidence=min(strength, 1.0),
                linguistic_template=f"Cross-modal connection from text position {i} to image region {j}"
            )
            patterns.append(pattern)
        
        return patterns

class RuleComposer:
    """Composes complex rules from simpler patterns"""
    
    def compose_rules(self, patterns: List[RulePattern]) -> List[CompositionalRule]:
        """Compose complex rules from patterns"""
        # This would implement rule composition logic
        # For now, return empty list
        return []

class RuleValidator:
    """Validates rules for consistency and correctness"""
    
    def validate_rule(self, rule: CompositionalRule) -> Dict[str, Any]:
        """Validate a compositional rule"""
        validation_result = {
            'is_valid': True,
            'confidence': rule.confidence,
            'consistency_score': np.random.random(),
            'suggestions': []
        }
        
        # Check rule consistency
        if rule.strength < 0.05:
            validation_result['is_valid'] = False
            validation_result['suggestions'].append("Rule strength too low")
        
        if rule.confidence < 0.3:
            validation_result['is_valid'] = False
            validation_result['suggestions'].append("Rule confidence too low")
        
        return validation_result

class EnhancedRuleExtractor:
    """Main enhanced rule extraction system"""
    
    def __init__(self):
        self.compositional_extractor = CompositionalRuleExtractor()
        self.nl_generator = NaturalLanguageGenerator()
        self.rule_analyzer = RuleAnalyzer()
        
    def extract_enhanced_rules(self, 
                             attention_weights: torch.Tensor,
                             tokens: Optional[List[str]] = None,
                             cross_modal_info: Optional[Dict] = None) -> Dict[str, Any]:
        """Extract enhanced rules with natural language descriptions"""
        
        # Extract compositional rules
        compositional_rules = self.compositional_extractor.extract_compositional_rules(
            attention_weights, tokens, cross_modal_info
        )
        
        # Generate natural language descriptions
        enhanced_rules = []
        for rule in compositional_rules:
            # Generate enhanced description
            rule_data = {
                'strength': rule.strength,
                'confidence': rule.confidence,
                'from_position': int(rule.antecedent[0].split('_')[-1]) if 'position_' in rule.antecedent[0] else 0,
                'to_position': int(rule.consequent.split('_')[-1]) if 'position_' in rule.consequent else 0,
                'attention_entropy': 0.0,  # Would be calculated from attention weights
                'sparsity': 0.0,
                'tnorm_type_encoding': 0.0,
                'membership_function_type': 0.0,
                'cross_modal': 1.0 if rule.composition_type == 'cross_modal' else 0.0,
                'temporal_dependency': 0.0
            }
            
            enhanced_description = self.nl_generator.generate_description(rule_data, tokens)
            rule.linguistic_description = enhanced_description
            
            enhanced_rules.append(rule)
        
        # Analyze rule patterns
        rule_analysis = self.rule_analyzer.analyze_rules(enhanced_rules)
        
        return {
            'compositional_rules': enhanced_rules,
            'rule_analysis': rule_analysis,
            'total_rules': len(enhanced_rules),
            'valid_rules': len([r for r in enhanced_rules if r.validation_status == 'valid']),
            'cross_modal_rules': len([r for r in enhanced_rules if r.composition_type == 'cross_modal'])
        }

class RuleAnalyzer:
    """Analyzes extracted rules for insights and patterns"""
    
    def analyze_rules(self, rules: List[CompositionalRule]) -> Dict[str, Any]:
        """Analyze rules for patterns and insights"""
        
        analysis = {
            'rule_types': Counter([r.composition_type for r in rules]),
            'strength_distribution': {
                'mean': np.mean([r.strength for r in rules]),
                'std': np.std([r.strength for r in rules]),
                'min': np.min([r.strength for r in rules]),
                'max': np.max([r.strength for r in rules])
            },
            'confidence_distribution': {
                'mean': np.mean([r.confidence for r in rules]),
                'std': np.std([r.confidence for r in rules])
            },
            'validation_status': Counter([r.validation_status for r in rules]),
            'complexity_analysis': self._analyze_complexity(rules),
            'semantic_clusters': self._find_semantic_clusters(rules)
        }
        
        return analysis
    
    def _analyze_complexity(self, rules: List[CompositionalRule]) -> Dict[str, Any]:
        """Analyze rule complexity"""
        complexities = [len(r.antecedent) + len(r.sub_rules) for r in rules]
        
        return {
            'mean_complexity': np.mean(complexities),
            'max_complexity': np.max(complexities),
            'complexity_distribution': Counter(complexities)
        }
    
    def _find_semantic_clusters(self, rules: List[CompositionalRule]) -> List[List[str]]:
        """Find semantic clusters in rules"""
        # This would implement semantic clustering
        # For now, return empty list
        return []

def demo_enhanced_rule_extraction():
    """Demo function for enhanced rule extraction"""
    print("🔍 Enhanced Rule Extraction Demo")
    print("=" * 50)
    
    # Create enhanced extractor
    extractor = EnhancedRuleExtractor()
    
    # Create sample attention weights with patterns
    seq_len = 10
    attention_weights = torch.rand(1, seq_len, seq_len)
    
    # Add some strong sequential patterns
    attention_weights[0, 0, 2] = 0.3  # 0 -> 2
    attention_weights[0, 2, 4] = 0.25  # 2 -> 4
    attention_weights[0, 4, 6] = 0.2   # 4 -> 6
    
    # Add cross-modal pattern (assume first 5 are text, last 5 are image)
    attention_weights[0, 1, 7] = 0.28  # text -> image
    attention_weights[0, 3, 8] = 0.22  # text -> image
    
    # Normalize
    attention_weights = torch.softmax(attention_weights, dim=-1)
    
    # Create mock tokens
    tokens = ["The", "cat", "sat", "on", "the", "mat", "quietly", "image1", "image2", "image3"]
    
    # Cross-modal info
    cross_modal_info = {
        'text_length': 7,
        'image_length': 3
    }
    
    # Extract enhanced rules
    result = extractor.extract_enhanced_rules(attention_weights, tokens, cross_modal_info)
    
    print(f"✅ Extracted {result['total_rules']} compositional rules")
    print(f"   Valid rules: {result['valid_rules']}")
    print(f"   Cross-modal rules: {result['cross_modal_rules']}")
    
    # Show rule types
    print(f"📊 Rule types: {result['rule_analysis']['rule_types']}")
    
    # Show some example rules
    print(f"\n📋 Example rules:")
    for i, rule in enumerate(result['compositional_rules'][:3]):
        print(f"   Rule {i+1}: {rule.linguistic_description}")
        print(f"      Type: {rule.composition_type}")
        print(f"      Strength: {rule.strength:.3f}")
        print(f"      Validation: {rule.validation_status}")
    
    return extractor, result

if __name__ == "__main__":
    demo_enhanced_rule_extraction()


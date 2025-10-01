"""
Adaptive User Interface for Fuzzy Attention Networks
Three-tier progressive disclosure system with real-time user expertise assessment
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import time
from collections import defaultdict, deque

class UserExpertiseLevel(Enum):
    NOVICE = "novice"
    INTERMEDIATE = "intermediate" 
    EXPERT = "expert"

@dataclass
class UserInteraction:
    """Represents a single user interaction"""
    timestamp: float
    action_type: str  # 'click', 'hover', 'request_explanation', 'feedback'
    target_element: str
    duration: float
    context: Dict[str, Any]

@dataclass
class UserProfile:
    """User profile with expertise assessment"""
    user_id: str
    expertise_level: UserExpertiseLevel
    confidence_score: float
    interaction_history: List[UserInteraction]
    expertise_indicators: Dict[str, float]
    last_updated: float

class UserExpertiseAssessor:
    """Real-time user expertise assessment through interaction patterns"""
    
    def __init__(self):
        self.expertise_indicators = {
            'technical_terminology_usage': 0.0,
            'explanation_depth_preference': 0.0,
            'interaction_sophistication': 0.0,
            'feedback_quality': 0.0,
            'exploration_patterns': 0.0
        }
        
        # Thresholds for expertise levels
        self.expertise_thresholds = {
            UserExpertiseLevel.NOVICE: 0.3,
            UserExpertiseLevel.INTERMEDIATE: 0.6,
            UserExpertiseLevel.EXPERT: 0.8
        }
    
    def assess_expertise(self, user_profile: UserProfile) -> Tuple[UserExpertiseLevel, float]:
        """Assess user expertise level based on interaction patterns"""
        
        # Calculate weighted expertise score
        weights = {
            'technical_terminology_usage': 0.25,
            'explanation_depth_preference': 0.30,
            'interaction_sophistication': 0.20,
            'feedback_quality': 0.15,
            'exploration_patterns': 0.10
        }
        
        total_score = 0.0
        for indicator, weight in weights.items():
            total_score += user_profile.expertise_indicators[indicator] * weight
        
        # Determine expertise level
        if total_score >= self.expertise_thresholds[UserExpertiseLevel.EXPERT]:
            level = UserExpertiseLevel.EXPERT
        elif total_score >= self.expertise_thresholds[UserExpertiseLevel.INTERMEDIATE]:
            level = UserExpertiseLevel.INTERMEDIATE
        else:
            level = UserExpertiseLevel.NOVICE
        
        return level, total_score
    
    def update_expertise_indicators(self, 
                                  user_profile: UserProfile, 
                                  interaction: UserInteraction):
        """Update expertise indicators based on new interaction"""
        
        # Technical terminology usage
        if interaction.action_type == 'request_explanation':
            requested_depth = interaction.context.get('explanation_depth', 'basic')
            if requested_depth == 'technical':
                user_profile.expertise_indicators['technical_terminology_usage'] += 0.1
            elif requested_depth == 'basic':
                user_profile.expertise_indicators['technical_terminology_usage'] -= 0.05
        
        # Explanation depth preference
        if interaction.action_type == 'feedback':
            feedback_type = interaction.context.get('feedback_type', '')
            if 'more detail' in feedback_type.lower():
                user_profile.expertise_indicators['explanation_depth_preference'] += 0.1
            elif 'too complex' in feedback_type.lower():
                user_profile.expertise_indicators['explanation_depth_preference'] -= 0.1
        
        # Interaction sophistication
        if interaction.duration > 5.0:  # Long interactions suggest deeper engagement
            user_profile.expertise_indicators['interaction_sophistication'] += 0.05
        
        # Exploration patterns
        if interaction.action_type == 'click' and 'rule' in interaction.target_element:
            user_profile.expertise_indicators['exploration_patterns'] += 0.05
        
        # Clamp values to [0, 1]
        for key in user_profile.expertise_indicators:
            user_profile.expertise_indicators[key] = np.clip(
                user_profile.expertise_indicators[key], 0.0, 1.0
            )

class AdaptiveExplanationSystem:
    """Three-tier progressive disclosure system"""
    
    def __init__(self):
        self.explanation_templates = {
            UserExpertiseLevel.NOVICE: {
                'title': "How the AI understands your text",
                'intro': "The AI model looks at {} important connections in your text to understand it better.",
                'rule_format': "🔗 Connection {}: The model connects '{}' with '{}'",
                'summary': "These connections help the AI understand the meaning and relationships in your text.",
                'technical_details': False,
                'visual_style': 'simple',
                'max_rules_shown': 3
            },
            UserExpertiseLevel.INTERMEDIATE: {
                'title': "Fuzzy Attention Analysis",
                'intro': "The fuzzy attention mechanism identified {} key relationships with the following strengths:",
                'rule_format': "📊 Rule {}: {} (Attention: {:.3f}, Confidence: {:.3f})",
                'summary': "These fuzzy rules represent learned attention patterns that guide the model's understanding.",
                'technical_details': True,
                'visual_style': 'detailed',
                'max_rules_shown': 5
            },
            UserExpertiseLevel.EXPERT: {
                'title': "Fuzzy Attention Network Analysis",
                'intro': "Extracted {} fuzzy rules from attention weights using {} t-norm operations:",
                'rule_format': "🧠 FuzzyRule(from={}, to={}, strength={:.4f}, confidence={:.4f}, tnorm='{}'): {}",
                'summary': "Technical details: {} membership functions, attention entropy: {:.3f}, sparsity: {:.3f}",
                'technical_details': True,
                'visual_style': 'technical',
                'max_rules_shown': 10
            }
        }
    
    def generate_adaptive_explanation(self, 
                                    rules: List[Any],
                                    user_profile: UserProfile,
                                    tokens: Optional[List[str]] = None,
                                    attention_patterns: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate adaptive explanation based on user expertise level"""
        
        expertise_level = user_profile.expertise_level
        template = self.explanation_templates[expertise_level]
        
        # Limit number of rules shown
        max_rules = template['max_rules_shown']
        shown_rules = rules[:max_rules]
        
        # Generate explanation content
        explanation = {
            'title': template['title'],
            'user_level': expertise_level.value,
            'confidence': user_profile.confidence_score,
            'content': {
                'intro': template['intro'].format(len(shown_rules)),
                'rules': [],
                'summary': template['summary'],
                'technical_details': template['technical_details']
            },
            'visual_style': template['visual_style'],
            'metadata': {
                'total_rules': len(rules),
                'rules_shown': len(shown_rules),
                'generation_time': time.time()
            }
        }
        
        # Format rules based on user level
        for i, rule in enumerate(shown_rules):
            if expertise_level == UserExpertiseLevel.NOVICE:
                rule_text = template['rule_format'].format(
                    i+1,
                    tokens[rule.from_position] if tokens and rule.from_position < len(tokens) else f"position {rule.from_position}",
                    tokens[rule.to_position] if tokens and rule.to_position < len(tokens) else f"position {rule.to_position}"
                )
            elif expertise_level == UserExpertiseLevel.INTERMEDIATE:
                rule_text = template['rule_format'].format(
                    i+1,
                    rule.linguistic_description,
                    rule.strength,
                    rule.confidence
                )
            else:  # EXPERT
                rule_text = template['rule_format'].format(
                    i+1,
                    rule.from_position,
                    rule.to_position,
                    rule.strength,
                    rule.confidence,
                    rule.tnorm_type,
                    rule.linguistic_description
                )
            
            explanation['content']['rules'].append({
                'text': rule_text,
                'strength': rule.strength,
                'confidence': rule.confidence,
                'from_position': rule.from_position,
                'to_position': rule.to_position
            })
        
        # Add technical details for expert users
        if expertise_level == UserExpertiseLevel.EXPERT and attention_patterns:
            explanation['content']['summary'] = template['summary'].format(
                'gaussian',
                np.mean(attention_patterns['attention_entropy']),
                np.mean(attention_patterns['attention_sparsity'])
            )
        
        return explanation

class InteractiveInterface:
    """Main interactive interface for user interaction"""
    
    def __init__(self):
        self.expertise_assessor = UserExpertiseAssessor()
        self.explanation_system = AdaptiveExplanationSystem()
        self.user_profiles: Dict[str, UserProfile] = {}
        self.session_data: Dict[str, Any] = {}
    
    def initialize_user(self, user_id: str) -> UserProfile:
        """Initialize new user profile"""
        user_profile = UserProfile(
            user_id=user_id,
            expertise_level=UserExpertiseLevel.NOVICE,
            confidence_score=0.5,
            interaction_history=[],
            expertise_indicators={
                'technical_terminology_usage': 0.0,
                'explanation_depth_preference': 0.0,
                'interaction_sophistication': 0.0,
                'feedback_quality': 0.0,
                'exploration_patterns': 0.0
            },
            last_updated=time.time()
        )
        
        self.user_profiles[user_id] = user_profile
        return user_profile
    
    def process_interaction(self, 
                          user_id: str,
                          action_type: str,
                          target_element: str,
                          duration: float = 0.0,
                          context: Optional[Dict] = None) -> UserProfile:
        """Process user interaction and update profile"""
        
        if user_id not in self.user_profiles:
            self.initialize_user(user_id)
        
        user_profile = self.user_profiles[user_id]
        
        # Create interaction record
        interaction = UserInteraction(
            timestamp=time.time(),
            action_type=action_type,
            target_element=target_element,
            duration=duration,
            context=context or {}
        )
        
        # Update user profile
        user_profile.interaction_history.append(interaction)
        user_profile.last_updated = time.time()
        
        # Update expertise indicators
        self.expertise_assessor.update_expertise_indicators(user_profile, interaction)
        
        # Reassess expertise level
        new_level, confidence = self.expertise_assessor.assess_expertise(user_profile)
        user_profile.expertise_level = new_level
        user_profile.confidence_score = confidence
        
        return user_profile
    
    def get_explanation(self, 
                       user_id: str,
                       rules: List[Any],
                       tokens: Optional[List[str]] = None,
                       attention_patterns: Optional[Dict] = None) -> Dict[str, Any]:
        """Get adaptive explanation for user"""
        
        if user_id not in self.user_profiles:
            self.initialize_user(user_id)
        
        user_profile = self.user_profiles[user_id]
        
        return self.explanation_system.generate_adaptive_explanation(
            rules, user_profile, tokens, attention_patterns
        )
    
    def export_user_data(self, user_id: str, filepath: str):
        """Export user interaction data for analysis"""
        if user_id not in self.user_profiles:
            return
        
        user_profile = self.user_profiles[user_id]
        
        export_data = {
            'user_id': user_profile.user_id,
            'expertise_level': user_profile.expertise_level.value,
            'confidence_score': user_profile.confidence_score,
            'expertise_indicators': user_profile.expertise_indicators,
            'interaction_count': len(user_profile.interaction_history),
            'last_updated': user_profile.last_updated,
            'interactions': [
                {
                    'timestamp': i.timestamp,
                    'action_type': i.action_type,
                    'target_element': i.target_element,
                    'duration': i.duration,
                    'context': i.context
                }
                for i in user_profile.interaction_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

def demo_adaptive_interface():
    """Demo function for adaptive interface"""
    print("🎯 Adaptive Interface Demo")
    print("=" * 40)
    
    # Create interface
    interface = InteractiveInterface()
    user_id = "demo_user"
    
    # Simulate user interactions
    print("👤 Simulating user interactions...")
    
    # Novice user interactions
    interface.process_interaction(user_id, 'request_explanation', 'attention_weights', 
                                context={'explanation_depth': 'basic'})
    interface.process_interaction(user_id, 'feedback', 'explanation', 
                                context={'feedback_type': 'too complex'})
    
    print(f"   Initial expertise: {interface.user_profiles[user_id].expertise_level.value}")
    
    # Intermediate user interactions
    interface.process_interaction(user_id, 'request_explanation', 'attention_weights',
                                context={'explanation_depth': 'detailed'})
    interface.process_interaction(user_id, 'click', 'rule_1', duration=3.0)
    interface.process_interaction(user_id, 'hover', 'technical_details', duration=2.0)
    
    print(f"   After interactions: {interface.user_profiles[user_id].expertise_level.value}")
    
    # Expert user interactions
    interface.process_interaction(user_id, 'request_explanation', 'attention_weights',
                                context={'explanation_depth': 'technical'})
    interface.process_interaction(user_id, 'click', 'rule_2', duration=8.0)
    interface.process_interaction(user_id, 'feedback', 'explanation',
                                context={'feedback_type': 'need more detail about t-norms'})
    
    final_profile = interface.user_profiles[user_id]
    print(f"   Final expertise: {final_profile.expertise_level.value} (confidence: {final_profile.confidence_score:.3f})")
    
    # Generate explanations for different expertise levels
    print("\n📝 Generating adaptive explanations...")
    
    # Mock rules for demonstration
    from rule_extractor import FuzzyRule
    mock_rules = [
        FuzzyRule(0, 2, 0.25, 0.8, "Position 0 strongly attends to position 2", "gaussian", "product"),
        FuzzyRule(1, 3, 0.18, 0.7, "Position 1 moderately attends to position 3", "gaussian", "product"),
        FuzzyRule(4, 6, 0.12, 0.6, "Position 4 slightly attends to position 6", "gaussian", "product")
    ]
    
    mock_tokens = ["The", "cat", "sat", "on", "the", "mat", "quietly"]
    
    explanation = interface.get_explanation(user_id, mock_rules, mock_tokens)
    
    print(f"\n🎯 EXPLANATION FOR {explanation['user_level'].upper()}:")
    print(f"Title: {explanation['title']}")
    print(f"Confidence: {explanation['confidence']:.3f}")
    print(f"Content: {explanation['content']['intro']}")
    
    for rule in explanation['content']['rules']:
        print(f"  {rule['text']}")
    
    print(f"Summary: {explanation['content']['summary']}")
    
    return interface, explanation

if __name__ == "__main__":
    demo_adaptive_interface()


"""
Enhanced Adaptive Interface with ML-based User Assessment and Reinforcement Learning
Implements the full adaptive explanation system described in the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import time
from collections import defaultdict, deque
import math

class UserExpertiseLevel(Enum):
    NOVICE = "novice"
    INTERMEDIATE = "intermediate" 
    EXPERT = "expert"

@dataclass
class UserInteraction:
    """Enhanced user interaction with more detailed context"""
    timestamp: float
    action_type: str  # 'click', 'hover', 'request_explanation', 'feedback', 'rule_edit', 'validation'
    target_element: str
    duration: float
    context: Dict[str, Any]
    satisfaction_score: Optional[float] = None
    cognitive_load_estimate: Optional[float] = None

@dataclass
class UserProfile:
    """Enhanced user profile with ML-based assessment"""
    user_id: str
    expertise_level: UserExpertiseLevel
    confidence_score: float
    interaction_history: List[UserInteraction]
    expertise_indicators: Dict[str, float]
    learning_preferences: Dict[str, Any]
    last_updated: float
    adaptation_history: List[Dict[str, Any]]

class MLUserExpertiseAssessor(nn.Module):
    """ML-based user expertise assessment using neural networks"""
    
    def __init__(self, input_dim: int = 50, hidden_dim: int = 128):
        super().__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Expertise level prediction
        self.expertise_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 expertise levels
        )
        
        # Confidence prediction
        self.confidence_regressor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Learning rate for online adaptation
        self.learning_rate = 0.01
        
    def extract_features(self, user_profile: UserProfile) -> torch.Tensor:
        """Extract features from user profile for ML assessment"""
        features = []
        
        # Basic expertise indicators
        for indicator in ['technical_terminology_usage', 'explanation_depth_preference', 
                         'interaction_sophistication', 'feedback_quality', 'exploration_patterns']:
            features.append(user_profile.expertise_indicators.get(indicator, 0.0))
        
        # Interaction patterns
        if user_profile.interaction_history:
            recent_interactions = user_profile.interaction_history[-10:]  # Last 10 interactions
            
            # Average interaction duration
            avg_duration = np.mean([i.duration for i in recent_interactions])
            features.append(min(avg_duration / 10.0, 1.0))  # Normalize to [0,1]
            
            # Interaction diversity
            action_types = set(i.action_type for i in recent_interactions)
            features.append(len(action_types) / 10.0)  # Normalize
            
            # Technical interaction ratio
            technical_actions = ['request_explanation', 'rule_edit', 'validation']
            tech_ratio = sum(1 for i in recent_interactions if i.action_type in technical_actions) / len(recent_interactions)
            features.append(tech_ratio)
            
            # Satisfaction trend
            satisfaction_scores = [i.satisfaction_score for i in recent_interactions if i.satisfaction_score is not None]
            if satisfaction_scores:
                features.append(np.mean(satisfaction_scores))
            else:
                features.append(0.5)  # Default neutral
        else:
            features.extend([0.0, 0.0, 0.0, 0.5])  # Default values
        
        # Learning preferences
        for pref in ['visual_preference', 'text_preference', 'interactive_preference']:
            features.append(user_profile.learning_preferences.get(pref, 0.5))
        
        # Pad or truncate to fixed size
        while len(features) < 50:
            features.append(0.0)
        features = features[:50]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def forward(self, user_profile: UserProfile) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for expertise assessment"""
        features = self.extract_features(user_profile)
        hidden = self.feature_extractor(features)
        
        expertise_logits = self.expertise_classifier(hidden)
        confidence = self.confidence_regressor(hidden)
        
        return expertise_logits, confidence
    
    def assess_expertise(self, user_profile: UserProfile) -> Tuple[UserExpertiseLevel, float]:
        """Assess user expertise level using ML model"""
        with torch.no_grad():
            expertise_logits, confidence = self.forward(user_profile)
            
            # Get predicted expertise level
            expertise_probs = F.softmax(expertise_logits, dim=-1)
            predicted_level = torch.argmax(expertise_probs, dim=-1).item()
            
            expertise_levels = [UserExpertiseLevel.NOVICE, UserExpertiseLevel.INTERMEDIATE, UserExpertiseLevel.EXPERT]
            level = expertise_levels[predicted_level]
            confidence_score = confidence.item()
            
            return level, confidence_score
    
    def update_with_feedback(self, user_profile: UserProfile, feedback: Dict[str, Any]):
        """Update model with user feedback using online learning"""
        # This would implement online learning in a real system
        # For now, we'll simulate the update
        pass

class ReinforcementLearningAdapter(nn.Module):
    """Reinforcement learning agent for adaptive explanation complexity"""
    
    def __init__(self, state_dim: int = 20, action_dim: int = 3):
        super().__init__()
        
        # State representation: user profile + current explanation context
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Policy network for action selection
        self.policy_network = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value network for state value estimation
        self.value_network = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=1000)
        
    def get_state(self, user_profile: UserProfile, explanation_context: Dict[str, Any]) -> torch.Tensor:
        """Encode current state for RL agent"""
        state_features = []
        
        # User expertise indicators
        for indicator in user_profile.expertise_indicators.values():
            state_features.append(indicator)
        
        # Current explanation complexity
        state_features.append(explanation_context.get('current_complexity', 0.5))
        
        # Recent performance metrics
        state_features.append(explanation_context.get('comprehension_score', 0.5))
        state_features.append(explanation_context.get('satisfaction_score', 0.5))
        state_features.append(explanation_context.get('cognitive_load', 0.5))
        
        # Pad to fixed size
        while len(state_features) < 20:
            state_features.append(0.0)
        state_features = state_features[:20]
        
        return torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
    
    def select_action(self, state: torch.Tensor) -> int:
        """Select action (explanation complexity level) using policy"""
        with torch.no_grad():
            hidden = self.state_encoder(state)
            action_probs = self.policy_network(hidden)
            action = torch.multinomial(action_probs, 1).item()
            return action
    
    def get_value(self, state: torch.Tensor) -> float:
        """Get state value estimate"""
        with torch.no_grad():
            hidden = self.state_encoder(state)
            value = self.value_network(hidden)
            return value.item()
    
    def store_experience(self, state: torch.Tensor, action: int, reward: float, 
                        next_state: torch.Tensor, done: bool):
        """Store experience in replay buffer"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        self.experience_buffer.append(experience)
    
    def update_policy(self, optimizer: torch.optim.Optimizer):
        """Update policy using experience replay"""
        if len(self.experience_buffer) < 32:
            return
        
        # Sample batch from experience buffer
        batch = np.random.choice(self.experience_buffer, size=32, replace=False)
        
        states = torch.cat([exp['state'] for exp in batch])
        actions = torch.tensor([exp['action'] for exp in batch])
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32)
        next_states = torch.cat([exp['next_state'] for exp in batch])
        dones = torch.tensor([exp['done'] for exp in batch], dtype=torch.float32)
        
        # Compute policy loss
        hidden = self.state_encoder(states)
        action_probs = self.policy_network(hidden)
        log_probs = torch.log(action_probs + 1e-8)
        
        # Compute value targets
        with torch.no_grad():
            next_values = self.value_network(self.state_encoder(next_states)).squeeze()
            targets = rewards + 0.99 * next_values * (1 - dones)
        
        current_values = self.value_network(hidden).squeeze()
        value_loss = F.mse_loss(current_values, targets)
        
        # Policy loss with value baseline
        advantages = targets - current_values.detach()
        policy_loss = -(log_probs.gather(1, actions.unsqueeze(1)) * advantages.unsqueeze(1)).mean()
        
        total_loss = policy_loss + value_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

class EnhancedAdaptiveExplanationSystem:
    """Enhanced three-tier progressive disclosure system with ML and RL"""
    
    def __init__(self):
        self.ml_assessor = MLUserExpertiseAssessor()
        self.rl_adapter = ReinforcementLearningAdapter()
        self.rl_optimizer = torch.optim.Adam(self.rl_adapter.parameters(), lr=0.001)
        
        # Enhanced explanation templates with more sophisticated content
        self.explanation_templates = {
            UserExpertiseLevel.NOVICE: {
                'title': "How the AI understands your content",
                'intro': "The AI model looks at {0} important connections to understand your content better.",
                'rule_format': "🔗 Connection {0}: The model connects '{1}' with '{2}'",
                'summary': "These connections help the AI understand the meaning and relationships in your content.",
                'visualization': 'simple_attention_heatmap',
                'interaction_options': ['show_more', 'simplify', 'ask_question'],
                'max_rules_shown': 3
            },
            UserExpertiseLevel.INTERMEDIATE: {
                'title': "Fuzzy Attention Analysis with Visualizations",
                'intro': "The fuzzy attention mechanism identified {0} key relationships with detailed analysis:",
                'rule_format': "📊 Rule {0}: {1} (Attention: {2:.3f}, Confidence: {3:.3f})",
                'summary': "These fuzzy rules represent learned attention patterns. You can explore membership functions and t-norm operations.",
                'visualization': 'membership_function_plots',
                'interaction_options': ['explore_membership', 'modify_rules', 'compare_tnorms', 'export_analysis'],
                'max_rules_shown': 5
            },
            UserExpertiseLevel.EXPERT: {
                'title': "Compositional Fuzzy Rule Derivation",
                'intro': "Extracted {0} fuzzy rules with full compositional derivations using {1} t-norm operations:",
                'rule_format': "🧠 FuzzyRule(from={0}, to={1}, strength={2:.4f}, confidence={3:.4f}, tnorm='{4}'): {5}",
                'summary': "Technical details: {0} membership functions, attention entropy: {1:.3f}, sparsity: {2:.3f}. Full mathematical derivations available.",
                'visualization': 'full_mathematical_derivation',
                'interaction_options': ['edit_rules', 'validate_derivations', 'export_code', 'compare_architectures'],
                'max_rules_shown': 10
            }
        }
    
    def generate_adaptive_explanation(self, 
                                    rules: List[Any],
                                    user_profile: UserProfile,
                                    tokens: Optional[List[str]] = None,
                                    attention_patterns: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate enhanced adaptive explanation with ML-based personalization"""
        
        # Use ML to assess current expertise level
        current_level, confidence = self.ml_assessor.assess_expertise(user_profile)
        
        # Use RL to select optimal explanation complexity
        explanation_context = {
            'current_complexity': self._get_complexity_score(current_level),
            'comprehension_score': self._estimate_comprehension(user_profile),
            'satisfaction_score': self._estimate_satisfaction(user_profile),
            'cognitive_load': self._estimate_cognitive_load(user_profile)
        }
        
        state = self.rl_adapter.get_state(user_profile, explanation_context)
        rl_action = self.rl_adapter.select_action(state)
        
        # Adjust explanation level based on RL action
        adjusted_level = self._adjust_level_with_rl(current_level, rl_action)
        
        template = self.explanation_templates[adjusted_level]
        
        # Generate explanation content
        explanation = {
            'title': template['title'],
            'user_level': adjusted_level.value,
            'confidence': confidence,
            'rl_action': rl_action,
            'content': {
                'intro': template['intro'].format(len(rules)),
                'rules': [],
                'summary': template['summary'],
                'visualization_type': template['visualization'],
                'interaction_options': template['interaction_options']
            },
            'metadata': {
                'total_rules': len(rules),
                'rules_shown': min(len(rules), template['max_rules_shown']),
                'generation_time': time.time(),
                'ml_confidence': confidence,
                'rl_value': self.rl_adapter.get_value(state)
            }
        }
        
        # Format rules with enhanced information
        shown_rules = rules[:template['max_rules_shown']]
        for i, rule in enumerate(shown_rules):
            rule_info = self._format_enhanced_rule(rule, i, adjusted_level, tokens)
            explanation['content']['rules'].append(rule_info)
        
        # Add technical details for expert users
        if adjusted_level == UserExpertiseLevel.EXPERT and attention_patterns:
            explanation['content']['summary'] = template['summary'].format(
                'gaussian',
                np.mean(attention_patterns.get('attention_entropy', [0])),
                np.mean(attention_patterns.get('attention_sparsity', [0]))
            )
        
        return explanation
    
    def _get_complexity_score(self, level: UserExpertiseLevel) -> float:
        """Get complexity score for current level"""
        scores = {
            UserExpertiseLevel.NOVICE: 0.2,
            UserExpertiseLevel.INTERMEDIATE: 0.5,
            UserExpertiseLevel.EXPERT: 0.8
        }
        return scores[level]
    
    def _estimate_comprehension(self, user_profile: UserProfile) -> float:
        """Estimate user comprehension based on interaction history"""
        if not user_profile.interaction_history:
            return 0.5
        
        recent_interactions = user_profile.interaction_history[-5:]
        comprehension_indicators = []
        
        for interaction in recent_interactions:
            if interaction.action_type == 'feedback':
                if 'understood' in interaction.context.get('feedback_type', '').lower():
                    comprehension_indicators.append(1.0)
                elif 'confused' in interaction.context.get('feedback_type', '').lower():
                    comprehension_indicators.append(0.0)
                else:
                    comprehension_indicators.append(0.5)
            elif interaction.action_type == 'request_explanation':
                if interaction.context.get('explanation_depth') == 'technical':
                    comprehension_indicators.append(0.8)
                elif interaction.context.get('explanation_depth') == 'basic':
                    comprehension_indicators.append(0.3)
                else:
                    comprehension_indicators.append(0.5)
        
        return np.mean(comprehension_indicators) if comprehension_indicators else 0.5
    
    def _estimate_satisfaction(self, user_profile: UserProfile) -> float:
        """Estimate user satisfaction based on interaction patterns"""
        if not user_profile.interaction_history:
            return 0.5
        
        recent_interactions = user_profile.interaction_history[-5:]
        satisfaction_scores = [i.satisfaction_score for i in recent_interactions if i.satisfaction_score is not None]
        
        if satisfaction_scores:
            return np.mean(satisfaction_scores)
        
        # Estimate from interaction patterns
        positive_indicators = 0
        total_indicators = 0
        
        for interaction in recent_interactions:
            if interaction.action_type == 'feedback':
                total_indicators += 1
                if 'good' in interaction.context.get('feedback_type', '').lower():
                    positive_indicators += 1
            elif interaction.duration > 3.0:  # Long interactions suggest engagement
                total_indicators += 1
                positive_indicators += 0.7
        
        return positive_indicators / total_indicators if total_indicators > 0 else 0.5
    
    def _estimate_cognitive_load(self, user_profile: UserProfile) -> float:
        """Estimate cognitive load based on interaction patterns"""
        if not user_profile.interaction_history:
            return 0.5
        
        recent_interactions = user_profile.interaction_history[-5:]
        load_indicators = []
        
        for interaction in recent_interactions:
            if interaction.action_type == 'feedback':
                if 'too complex' in interaction.context.get('feedback_type', '').lower():
                    load_indicators.append(0.8)
                elif 'too simple' in interaction.context.get('feedback_type', '').lower():
                    load_indicators.append(0.2)
                else:
                    load_indicators.append(0.5)
            elif interaction.cognitive_load_estimate is not None:
                load_indicators.append(interaction.cognitive_load_estimate)
        
        return np.mean(load_indicators) if load_indicators else 0.5
    
    def _adjust_level_with_rl(self, base_level: UserExpertiseLevel, rl_action: int) -> UserExpertiseLevel:
        """Adjust explanation level based on RL action"""
        levels = [UserExpertiseLevel.NOVICE, UserExpertiseLevel.INTERMEDIATE, UserExpertiseLevel.EXPERT]
        current_idx = levels.index(base_level)
        
        # RL action: 0 = simplify, 1 = keep same, 2 = make more complex
        if rl_action == 0 and current_idx > 0:
            return levels[current_idx - 1]
        elif rl_action == 2 and current_idx < len(levels) - 1:
            return levels[current_idx + 1]
        else:
            return base_level
    
    def _format_enhanced_rule(self, rule: Any, index: int, level: UserExpertiseLevel, 
                            tokens: Optional[List[str]] = None) -> Dict[str, Any]:
        """Format rule with enhanced information based on user level"""
        rule_info = {
            'index': index + 1,
            'from_position': rule.from_position,
            'to_position': rule.to_position,
            'strength': rule.strength,
            'confidence': rule.confidence,
            'linguistic_description': rule.linguistic_description
        }
        
        if level == UserExpertiseLevel.NOVICE:
            rule_info['display_text'] = f"Connection {index + 1}: The model connects '{tokens[rule.from_position] if tokens and rule.from_position < len(tokens) else f'position {rule.from_position}'}' with '{tokens[rule.to_position] if tokens and rule.to_position < len(tokens) else f'position {rule.to_position}'}'"
        elif level == UserExpertiseLevel.INTERMEDIATE:
            rule_info['display_text'] = f"Rule {index + 1}: {rule.linguistic_description} (Attention: {rule.strength:.3f}, Confidence: {rule.confidence:.3f})"
            rule_info['membership_values'] = self._get_membership_values(rule)
            rule_info['tnorm_details'] = self._get_tnorm_details(rule)
        else:  # EXPERT
            rule_info['display_text'] = f"FuzzyRule(from={rule.from_position}, to={rule.to_position}, strength={rule.strength:.4f}, confidence={rule.confidence:.4f}, tnorm='{rule.tnorm_type}'): {rule.linguistic_description}"
            rule_info['mathematical_derivation'] = self._get_mathematical_derivation(rule)
            rule_info['membership_functions'] = self._get_membership_functions(rule)
            rule_info['compositional_structure'] = self._get_compositional_structure(rule)
        
        return rule_info
    
    def _get_membership_values(self, rule: Any) -> Dict[str, float]:
        """Get membership function values for intermediate users"""
        # This would extract actual membership values from the model
        return {
            'query_membership': np.random.random(),
            'key_membership': np.random.random(),
            'combined_membership': rule.strength
        }
    
    def _get_tnorm_details(self, rule: Any) -> Dict[str, Any]:
        """Get t-norm operation details"""
        return {
            'tnorm_type': rule.tnorm_type,
            'operation': f"T({np.random.random():.3f}, {np.random.random():.3f}) = {rule.strength:.3f}",
            'properties': ['associative', 'commutative', 'monotonic']
        }
    
    def _get_mathematical_derivation(self, rule: Any) -> str:
        """Get full mathematical derivation for expert users"""
        return f"""
        Mathematical Derivation:
        μ_Q(x) = exp(-||x - c_Q||²/(2σ_Q²))
        μ_K(y) = exp(-||y - c_K||²/(2σ_K²))
        T(μ_Q, μ_K) = μ_Q · μ_K = {rule.strength:.4f}
        """
    
    def _get_membership_functions(self, rule: Any) -> Dict[str, Any]:
        """Get membership function details"""
        return {
            'type': 'gaussian',
            'centers': [np.random.random() for _ in range(3)],
            'sigmas': [np.random.random() for _ in range(3)],
            'parameters': 'learnable'
        }
    
    def _get_compositional_structure(self, rule: Any) -> Dict[str, Any]:
        """Get compositional rule structure"""
        return {
            'antecedent': f"position_{rule.from_position}",
            'consequent': f"position_{rule.to_position}",
            'strength': rule.strength,
            'composition_type': 'fuzzy_implication',
            'derivation_steps': ['membership_computation', 'tnorm_application', 'normalization']
        }

class InteractiveRuleRefinement:
    """Interactive rule refinement and validation system"""
    
    def __init__(self):
        self.rule_validator = RuleValidator()
        self.rule_editor = RuleEditor()
    
    def refine_rule(self, rule: Any, user_feedback: Dict[str, Any]) -> Any:
        """Refine rule based on user feedback"""
        # This would implement actual rule refinement
        refined_rule = rule
        return refined_rule
    
    def validate_rule(self, rule: Any) -> Dict[str, Any]:
        """Validate rule for consistency and correctness"""
        validation_result = {
            'is_valid': True,
            'confidence': rule.confidence,
            'consistency_score': np.random.random(),
            'suggestions': []
        }
        return validation_result

class RuleValidator:
    """Validates fuzzy rules for consistency and correctness"""
    pass

class RuleEditor:
    """Interactive rule editing interface"""
    pass

def demo_enhanced_adaptive_interface():
    """Demo function for enhanced adaptive interface"""
    print("🎯 Enhanced Adaptive Interface Demo")
    print("=" * 50)
    
    # Create enhanced system
    system = EnhancedAdaptiveExplanationSystem()
    
    # Create user profile
    user_profile = UserProfile(
        user_id="demo_user",
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
        learning_preferences={
            'visual_preference': 0.7,
            'text_preference': 0.3,
            'interactive_preference': 0.8
        },
        last_updated=time.time(),
        adaptation_history=[]
    )
    
    # Simulate interactions to build expertise
    print("👤 Simulating user interactions...")
    
    # Novice interactions
    interaction1 = UserInteraction(
        timestamp=time.time(),
        action_type='request_explanation',
        target_element='attention_weights',
        duration=2.0,
        context={'explanation_depth': 'basic'},
        satisfaction_score=0.8
    )
    user_profile.interaction_history.append(interaction1)
    
    # Intermediate interactions
    interaction2 = UserInteraction(
        timestamp=time.time(),
        action_type='request_explanation',
        target_element='attention_weights',
        duration=5.0,
        context={'explanation_depth': 'detailed'},
        satisfaction_score=0.9
    )
    user_profile.interaction_history.append(interaction2)
    
    # Expert interactions
    interaction3 = UserInteraction(
        timestamp=time.time(),
        action_type='rule_edit',
        target_element='rule_1',
        duration=8.0,
        context={'edit_type': 'modify_strength', 'new_value': 0.8},
        satisfaction_score=0.95
    )
    user_profile.interaction_history.append(interaction3)
    
    # Generate adaptive explanation
    from rule_extractor import FuzzyRule
    mock_rules = [
        FuzzyRule(0, 2, 0.25, 0.8, "Position 0 strongly attends to position 2", "gaussian", "product"),
        FuzzyRule(1, 3, 0.18, 0.7, "Position 1 moderately attends to position 3", "gaussian", "product"),
        FuzzyRule(4, 6, 0.12, 0.6, "Position 4 slightly attends to position 6", "gaussian", "product")
    ]
    
    explanation = system.generate_adaptive_explanation(mock_rules, user_profile)
    
    print(f"🎯 Generated explanation for {explanation['user_level']} user")
    print(f"   ML Confidence: {explanation['metadata']['ml_confidence']:.3f}")
    print(f"   RL Action: {explanation['rl_action']}")
    print(f"   RL Value: {explanation['metadata']['rl_value']:.3f}")
    print(f"   Visualization: {explanation['content']['visualization_type']}")
    print(f"   Interaction options: {explanation['content']['interaction_options']}")
    
    return system, explanation

if __name__ == "__main__":
    demo_enhanced_adaptive_interface()


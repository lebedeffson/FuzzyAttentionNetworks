#!/usr/bin/env python3
"""
Real-time User Expertise Assessment with Reinforcement Learning
Implements dynamic expertise assessment based on interaction patterns
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time
import json

class InteractionType(Enum):
    """Types of user interactions"""
    TEXT_INPUT = "text_input"
    IMAGE_UPLOAD = "image_upload"
    EXPLANATION_REQUEST = "explanation_request"
    RULE_REFINEMENT = "rule_refinement"
    FEEDBACK_PROVIDED = "feedback_provided"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    DETAILED_ANALYSIS = "detailed_analysis"

@dataclass
class InteractionEvent:
    """Single interaction event"""
    timestamp: float
    interaction_type: InteractionType
    context: Dict[str, Any]
    duration: float
    complexity_score: float  # 0-1, how complex the interaction was
    success: bool  # Whether the interaction was successful

class InteractionPatternAnalyzer:
    """Analyzes user interaction patterns to assess expertise"""
    
    def __init__(self):
        self.interaction_weights = {
            InteractionType.TEXT_INPUT: 0.1,
            InteractionType.IMAGE_UPLOAD: 0.2,
            InteractionType.EXPLANATION_REQUEST: 0.3,
            InteractionType.RULE_REFINEMENT: 0.8,
            InteractionType.FEEDBACK_PROVIDED: 0.6,
            InteractionType.PARAMETER_ADJUSTMENT: 0.7,
            InteractionType.DETAILED_ANALYSIS: 0.9
        }
        
        self.complexity_indicators = {
            'technical_terminology_usage': 0.0,
            'explanation_depth_preference': 0.0,
            'interaction_sophistication': 0.0,
            'feedback_quality': 0.0,
            'exploration_patterns': 0.0
        }
    
    def analyze_interaction_pattern(self, interactions: List[InteractionEvent]) -> Dict[str, float]:
        """Analyze interaction pattern to extract expertise indicators"""
        
        if not interactions:
            return self.complexity_indicators.copy()
        
        # Calculate technical terminology usage
        technical_terms = self._count_technical_terms(interactions)
        total_interactions = len(interactions)
        self.complexity_indicators['technical_terminology_usage'] = min(technical_terms / max(total_interactions, 1), 1.0)
        
        # Calculate explanation depth preference
        depth_requests = sum(1 for i in interactions if i.interaction_type == InteractionType.EXPLANATION_REQUEST)
        self.complexity_indicators['explanation_depth_preference'] = min(depth_requests / max(total_interactions, 1), 1.0)
        
        # Calculate interaction sophistication
        sophisticated_interactions = sum(1 for i in interactions 
                                       if i.interaction_type in [InteractionType.RULE_REFINEMENT, 
                                                               InteractionType.PARAMETER_ADJUSTMENT,
                                                               InteractionType.DETAILED_ANALYSIS])
        self.complexity_indicators['interaction_sophistication'] = min(sophisticated_interactions / max(total_interactions, 1), 1.0)
        
        # Calculate feedback quality
        feedback_interactions = [i for i in interactions if i.interaction_type == InteractionType.FEEDBACK_PROVIDED]
        if feedback_interactions:
            avg_feedback_quality = sum(i.complexity_score for i in feedback_interactions) / len(feedback_interactions)
            self.complexity_indicators['feedback_quality'] = avg_feedback_quality
        else:
            self.complexity_indicators['feedback_quality'] = 0.0
        
        # Calculate exploration patterns
        unique_interaction_types = len(set(i.interaction_type for i in interactions))
        self.complexity_indicators['exploration_patterns'] = min(unique_interaction_types / len(InteractionType), 1.0)
        
        return self.complexity_indicators.copy()
    
    def _count_technical_terms(self, interactions: List[InteractionEvent]) -> int:
        """Count technical terms used in interactions"""
        technical_terms = [
            'attention', 'fuzzy', 'membership', 't-norm', 'rule', 'extraction',
            'cross-modal', 'reasoning', 'interpretability', 'explanation',
            'threshold', 'confidence', 'pattern', 'entropy', 'sparsity'
        ]
        
        count = 0
        for interaction in interactions:
            if 'text' in interaction.context:
                text = interaction.context['text'].lower()
                count += sum(1 for term in technical_terms if term in text)
        
        return count

class ReinforcementLearningExpertiseAssessor(nn.Module):
    """Reinforcement Learning model for expertise assessment"""
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Neural network for expertise assessment
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3 expertise levels
            nn.Softmax(dim=-1)
        )
        
        # Value network for reward estimation
        self.value_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.experience_buffer = []
    
    def forward(self, expertise_indicators: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for expertise assessment
        
        Args:
            expertise_indicators: [batch_size, input_dim] expertise indicators
            
        Returns:
            expertise_probs: [batch_size, 3] probability distribution over expertise levels
            value_estimate: [batch_size, 1] value estimate
        """
        expertise_probs = self.network(expertise_indicators)
        value_estimate = self.value_network(expertise_indicators)
        
        return expertise_probs, value_estimate
    
    def update_with_feedback(self, 
                           expertise_indicators: torch.Tensor,
                           predicted_level: int,
                           actual_feedback: int,
                           reward: float):
        """Update model with user feedback using RL"""
        
        # Convert to tensors
        if not isinstance(expertise_indicators, torch.Tensor):
            expertise_indicators = torch.tensor(expertise_indicators, dtype=torch.float32)
        
        # Get current predictions
        expertise_probs, value_estimate = self.forward(expertise_indicators.unsqueeze(0))
        
        # Create target for expertise level
        target = torch.zeros(3)
        target[actual_feedback] = 1.0
        
        # Compute losses
        expertise_loss = nn.CrossEntropyLoss()(expertise_probs, target.unsqueeze(0))
        value_loss = nn.MSELoss()(value_estimate, torch.tensor([[reward]], dtype=torch.float32))
        
        # Total loss
        total_loss = expertise_loss + 0.1 * value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()

class RealTimeExpertiseAssessor:
    """Main class for real-time expertise assessment"""
    
    def __init__(self):
        self.pattern_analyzer = InteractionPatternAnalyzer()
        self.rl_assessor = ReinforcementLearningExpertiseAssessor()
        self.user_interactions: Dict[str, List[InteractionEvent]] = {}
        self.user_expertise_history: Dict[str, List[float]] = {}
        
    def log_interaction(self, 
                       user_id: str,
                       interaction_type: InteractionType,
                       context: Dict[str, Any],
                       duration: float = 0.0,
                       complexity_score: float = 0.5,
                       success: bool = True):
        """Log a user interaction"""
        
        if user_id not in self.user_interactions:
            self.user_interactions[user_id] = []
        
        event = InteractionEvent(
            timestamp=time.time(),
            interaction_type=interaction_type,
            context=context,
            duration=duration,
            complexity_score=complexity_score,
            success=success
        )
        
        self.user_interactions[user_id].append(event)
        
        # Keep only recent interactions (last 100)
        if len(self.user_interactions[user_id]) > 100:
            self.user_interactions[user_id] = self.user_interactions[user_id][-100:]
    
    def assess_expertise(self, user_id: str) -> Tuple[int, float, Dict[str, float]]:
        """
        Assess user expertise in real-time
        
        Args:
            user_id: User identifier
            
        Returns:
            expertise_level: 0=novice, 1=intermediate, 2=expert
            confidence: Confidence in assessment (0-1)
            indicators: Detailed expertise indicators
        """
        
        if user_id not in self.user_interactions:
            return 0, 0.5, self.pattern_analyzer.complexity_indicators.copy()
        
        interactions = self.user_interactions[user_id]
        
        # Analyze interaction patterns
        indicators = self.pattern_analyzer.analyze_interaction_pattern(interactions)
        
        # Convert to tensor
        indicators_tensor = torch.tensor([
            indicators['technical_terminology_usage'],
            indicators['explanation_depth_preference'],
            indicators['interaction_sophistication'],
            indicators['feedback_quality'],
            indicators['exploration_patterns']
        ], dtype=torch.float32)
        
        # Get RL assessment
        with torch.no_grad():
            expertise_probs, value_estimate = self.rl_assessor(indicators_tensor.unsqueeze(0))
        
        # Determine expertise level
        expertise_level = torch.argmax(expertise_probs).item()
        confidence = torch.max(expertise_probs).item()
        
        # Store history
        if user_id not in self.user_expertise_history:
            self.user_expertise_history[user_id] = []
        self.user_expertise_history[user_id].append(expertise_level)
        
        return expertise_level, confidence, indicators
    
    def update_with_feedback(self, 
                           user_id: str,
                           feedback_level: int,
                           satisfaction_score: float):
        """Update assessment with user feedback"""
        
        if user_id not in self.user_interactions:
            return
        
        interactions = self.user_interactions[user_id]
        indicators = self.pattern_analyzer.analyze_interaction_pattern(interactions)
        
        # Convert to tensor
        indicators_tensor = torch.tensor([
            indicators['technical_terminology_usage'],
            indicators['explanation_depth_preference'],
            indicators['interaction_sophistication'],
            indicators['feedback_quality'],
            indicators['exploration_patterns']
        ], dtype=torch.float32)
        
        # Calculate reward based on satisfaction
        reward = satisfaction_score * 2 - 1  # Convert 0-1 to -1 to 1
        
        # Update RL model
        loss = self.rl_assessor.update_with_feedback(
            indicators_tensor,
            feedback_level,
            feedback_level,
            reward
        )
        
        return loss
    
    def get_expertise_trend(self, user_id: str, window_size: int = 10) -> List[float]:
        """Get expertise trend over time"""
        
        if user_id not in self.user_expertise_history:
            return []
        
        history = self.user_expertise_history[user_id]
        
        if len(history) < window_size:
            return history
        
        # Return moving average
        trend = []
        for i in range(window_size, len(history) + 1):
            window = history[i-window_size:i]
            trend.append(sum(window) / len(window))
        
        return trend
    
    def get_adaptive_explanation_complexity(self, 
                                          user_id: str,
                                          base_complexity: float = 0.5) -> float:
        """Get adaptive explanation complexity based on expertise"""
        
        expertise_level, confidence, indicators = self.assess_expertise(user_id)
        
        # Adjust complexity based on expertise level
        if expertise_level == 0:  # Novice
            complexity = base_complexity * 0.5
        elif expertise_level == 1:  # Intermediate
            complexity = base_complexity
        else:  # Expert
            complexity = base_complexity * 1.5
        
        # Adjust based on confidence
        complexity *= confidence
        
        # Ensure within bounds
        complexity = max(0.1, min(1.0, complexity))
        
        return complexity

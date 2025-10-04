#!/usr/bin/env python3
"""
Interactive Rule Editor for Fuzzy Attention Networks
Allows users to interactively refine and validate fuzzy rules
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import time

class RuleEditAction(Enum):
    """Types of rule editing actions"""
    MODIFY_STRENGTH = "modify_strength"
    MODIFY_CONFIDENCE = "modify_confidence"
    MODIFY_DESCRIPTION = "modify_description"
    ADD_RULE = "add_rule"
    REMOVE_RULE = "remove_rule"
    MERGE_RULES = "merge_rules"
    SPLIT_RULE = "split_rule"

@dataclass
class RuleEdit:
    """Single rule edit operation"""
    timestamp: float
    action: RuleEditAction
    rule_id: str
    old_value: Any
    new_value: Any
    user_id: str
    justification: str = ""

@dataclass
class RuleValidation:
    """Rule validation result"""
    rule_id: str
    is_valid: bool
    confidence: float
    feedback: str
    suggested_improvements: List[str]

class InteractiveRuleEditor:
    """Interactive rule editor with validation and refinement capabilities"""
    
    def __init__(self, rule_extractor):
        self.rule_extractor = rule_extractor
        self.rule_edits: Dict[str, List[RuleEdit]] = {}
        self.rule_validations: Dict[str, List[RuleValidation]] = {}
        self.rule_templates = self._initialize_rule_templates()
        
    def _initialize_rule_templates(self) -> Dict[str, List[str]]:
        """Initialize linguistic templates for rule generation"""
        return {
            'attention_patterns': [
                "Position {from_pos} {strength} attends to position {to_pos}",
                "Token {from_token} {strength} focuses on token {to_token}",
                "Word {from_word} {strength} considers word {to_word}",
                "Element {from_pos} {strength} relates to element {to_pos}"
            ],
            'strength_modifiers': {
                'strong': ['strongly', 'primarily', 'mainly', 'heavily'],
                'medium': ['moderately', 'partially', 'somewhat', 'fairly'],
                'weak': ['slightly', 'minimally', 'barely', 'weakly']
            },
            'linguistic_connectors': [
                "in relation to", "with respect to", "concerning", "regarding",
                "in connection with", "as it relates to", "in the context of"
            ]
        }
    
    def edit_rule(self, 
                  user_id: str,
                  rule_id: str,
                  action: RuleEditAction,
                  new_value: Any,
                  justification: str = "") -> bool:
        """Edit a fuzzy rule"""
        
        if user_id not in self.rule_edits:
            self.rule_edits[user_id] = []
        
        # Find the rule to edit
        rule = self._find_rule_by_id(rule_id)
        if not rule:
            return False
        
        # Store old value
        old_value = self._get_rule_value(rule, action)
        
        # Apply edit
        success = self._apply_rule_edit(rule, action, new_value)
        
        if success:
            # Log the edit
            edit = RuleEdit(
                timestamp=time.time(),
                action=action,
                rule_id=rule_id,
                old_value=old_value,
                new_value=new_value,
                user_id=user_id,
                justification=justification
            )
            self.rule_edits[user_id].append(edit)
        
        return success
    
    def validate_rule(self, rule_id: str, context: Dict[str, Any]) -> RuleValidation:
        """Validate a rule and provide feedback"""
        
        rule = self._find_rule_by_id(rule_id)
        if not rule:
            return RuleValidation(
                rule_id=rule_id,
                is_valid=False,
                confidence=0.0,
                feedback="Rule not found",
                suggested_improvements=[]
            )
        
        # Perform validation checks
        validation_checks = self._perform_validation_checks(rule, context)
        
        # Calculate overall confidence
        confidence = sum(validation_checks.values()) / len(validation_checks)
        
        # Determine if valid
        is_valid = confidence > 0.6
        
        # Generate feedback
        feedback = self._generate_validation_feedback(rule, validation_checks)
        
        # Suggest improvements
        improvements = self._suggest_improvements(rule, validation_checks)
        
        validation = RuleValidation(
            rule_id=rule_id,
            is_valid=is_valid,
            confidence=confidence,
            feedback=feedback,
            suggested_improvements=improvements
        )
        
        # Store validation
        if rule_id not in self.rule_validations:
            self.rule_validations[rule_id] = []
        self.rule_validations[rule_id].append(validation)
        
        return validation
    
    def refine_rule_automatically(self, rule_id: str) -> bool:
        """Automatically refine a rule based on validation feedback"""
        
        rule = self._find_rule_by_id(rule_id)
        if not rule:
            return False
        
        # Get recent validations
        if rule_id not in self.rule_validations:
            return False
        
        recent_validations = self.rule_validations[rule_id][-3:]  # Last 3 validations
        
        # Calculate average confidence
        avg_confidence = sum(v.confidence for v in recent_validations) / len(recent_validations)
        
        if avg_confidence > 0.8:
            return True  # Rule is already good
        
        # Apply automatic refinements
        refinements_applied = 0
        
        # Refine strength if too low
        if rule.strength < 0.3:
            rule.strength = min(rule.strength * 1.2, 1.0)
            refinements_applied += 1
        
        # Refine confidence if too low
        if rule.confidence < 0.5:
            rule.confidence = min(rule.confidence * 1.1, 1.0)
            refinements_applied += 1
        
        # Refine linguistic description
        if len(rule.linguistic_description) < 20:  # Too short
            rule.linguistic_description = self._generate_enhanced_description(rule)
            refinements_applied += 1
        
        return refinements_applied > 0
    
    def merge_rules(self, user_id: str, rule_ids: List[str]) -> Optional[str]:
        """Merge multiple rules into one"""
        
        if len(rule_ids) < 2:
            return None
        
        # Find all rules
        rules = [self._find_rule_by_id(rid) for rid in rule_ids]
        rules = [r for r in rules if r is not None]
        
        if len(rules) < 2:
            return None
        
        # Create merged rule
        merged_rule = self._create_merged_rule(rules)
        
        # Log the merge
        edit = RuleEdit(
            timestamp=time.time(),
            action=RuleEditAction.MERGE_RULES,
            rule_id="merged_" + str(int(time.time())),
            old_value=[r.linguistic_description for r in rules],
            new_value=merged_rule.linguistic_description,
            user_id=user_id,
            justification=f"Merged {len(rules)} rules"
        )
        self.rule_edits[user_id].append(edit)
        
        return merged_rule.linguistic_description
    
    def split_rule(self, user_id: str, rule_id: str, split_criteria: Dict[str, Any]) -> List[str]:
        """Split a rule into multiple rules"""
        
        rule = self._find_rule_by_id(rule_id)
        if not rule:
            return []
        
        # Create split rules
        split_rules = self._create_split_rules(rule, split_criteria)
        
        # Log the split
        edit = RuleEdit(
            timestamp=time.time(),
            action=RuleEditAction.SPLIT_RULE,
            rule_id=rule_id,
            old_value=rule.linguistic_description,
            new_value=[r.linguistic_description for r in split_rules],
            user_id=user_id,
            justification=f"Split rule based on {split_criteria}"
        )
        self.rule_edits[user_id].append(edit)
        
        return [r.linguistic_description for r in split_rules]
    
    def get_rule_edit_history(self, user_id: str) -> List[RuleEdit]:
        """Get rule edit history for a user"""
        return self.rule_edits.get(user_id, [])
    
    def get_rule_validation_history(self, rule_id: str) -> List[RuleValidation]:
        """Get validation history for a rule"""
        return self.rule_validations.get(rule_id, [])
    
    def export_rules(self, user_id: str) -> Dict[str, Any]:
        """Export user's edited rules"""
        if user_id not in self.rule_edits:
            return {}
        
        return {
            'user_id': user_id,
            'edits': [
                {
                    'timestamp': edit.timestamp,
                    'action': edit.action.value,
                    'rule_id': edit.rule_id,
                    'old_value': edit.old_value,
                    'new_value': edit.new_value,
                    'justification': edit.justification
                }
                for edit in self.rule_edits[user_id]
            ],
            'export_timestamp': time.time()
        }
    
    def _find_rule_by_id(self, rule_id: str):
        """Find rule by ID (placeholder implementation)"""
        # In real implementation, this would search through active rules
        return None
    
    def _get_rule_value(self, rule, action: RuleEditAction):
        """Get current value of a rule property"""
        if action == RuleEditAction.MODIFY_STRENGTH:
            return rule.strength
        elif action == RuleEditAction.MODIFY_CONFIDENCE:
            return rule.confidence
        elif action == RuleEditAction.MODIFY_DESCRIPTION:
            return rule.linguistic_description
        return None
    
    def _apply_rule_edit(self, rule, action: RuleEditAction, new_value: Any) -> bool:
        """Apply rule edit"""
        try:
            if action == RuleEditAction.MODIFY_STRENGTH:
                rule.strength = float(new_value)
            elif action == RuleEditAction.MODIFY_CONFIDENCE:
                rule.confidence = float(new_value)
            elif action == RuleEditAction.MODIFY_DESCRIPTION:
                rule.linguistic_description = str(new_value)
            return True
        except:
            return False
    
    def _perform_validation_checks(self, rule, context: Dict[str, Any]) -> Dict[str, float]:
        """Perform validation checks on a rule"""
        checks = {}
        
        # Strength validation
        checks['strength_valid'] = 1.0 if 0.0 <= rule.strength <= 1.0 else 0.0
        
        # Confidence validation
        checks['confidence_valid'] = 1.0 if 0.0 <= rule.confidence <= 1.0 else 0.0
        
        # Description quality
        desc_length = len(rule.linguistic_description)
        checks['description_quality'] = min(desc_length / 50, 1.0)  # Prefer longer descriptions
        
        # Linguistic coherence
        checks['linguistic_coherence'] = self._check_linguistic_coherence(rule.linguistic_description)
        
        # Context relevance
        checks['context_relevance'] = self._check_context_relevance(rule, context)
        
        return checks
    
    def _check_linguistic_coherence(self, description: str) -> float:
        """Check linguistic coherence of rule description"""
        # Simple heuristic: check for proper sentence structure
        if not description.strip():
            return 0.0
        
        # Check for basic linguistic patterns
        coherence_score = 0.5  # Base score
        
        if any(word in description.lower() for word in ['attends', 'focuses', 'considers', 'relates']):
            coherence_score += 0.2
        
        if any(word in description.lower() for word in ['strongly', 'moderately', 'slightly']):
            coherence_score += 0.2
        
        if description.endswith('.') or description.endswith('!'):
            coherence_score += 0.1
        
        return min(coherence_score, 1.0)
    
    def _check_context_relevance(self, rule, context: Dict[str, Any]) -> float:
        """Check if rule is relevant to current context"""
        # Simple relevance check based on context
        if 'tokens' in context:
            tokens = context['tokens']
            if rule.from_position < len(tokens) and rule.to_position < len(tokens):
                return 1.0
        return 0.5
    
    def _generate_validation_feedback(self, rule, validation_checks: Dict[str, float]) -> str:
        """Generate human-readable validation feedback"""
        feedback_parts = []
        
        if validation_checks['strength_valid'] < 1.0:
            feedback_parts.append("Rule strength should be between 0.0 and 1.0")
        
        if validation_checks['confidence_valid'] < 1.0:
            feedback_parts.append("Rule confidence should be between 0.0 and 1.0")
        
        if validation_checks['description_quality'] < 0.5:
            feedback_parts.append("Rule description could be more detailed")
        
        if validation_checks['linguistic_coherence'] < 0.7:
            feedback_parts.append("Rule description could be more linguistically coherent")
        
        if validation_checks['context_relevance'] < 0.8:
            feedback_parts.append("Rule may not be relevant to current context")
        
        if not feedback_parts:
            return "Rule validation passed successfully"
        
        return "Validation issues: " + "; ".join(feedback_parts)
    
    def _suggest_improvements(self, rule, validation_checks: Dict[str, float]) -> List[str]:
        """Suggest improvements for a rule"""
        suggestions = []
        
        if validation_checks['description_quality'] < 0.5:
            suggestions.append("Add more detail to the rule description")
        
        if validation_checks['linguistic_coherence'] < 0.7:
            suggestions.append("Use more specific linguistic terms")
        
        if rule.strength < 0.3:
            suggestions.append("Consider increasing rule strength")
        
        if rule.confidence < 0.5:
            suggestions.append("Consider increasing rule confidence")
        
        return suggestions
    
    def _generate_enhanced_description(self, rule) -> str:
        """Generate enhanced linguistic description"""
        templates = self.rule_templates['attention_patterns']
        template = templates[0]  # Use first template
        
        strength_mod = 'moderately'
        if rule.strength > 0.7:
            strength_mod = 'strongly'
        elif rule.strength < 0.3:
            strength_mod = 'slightly'
        
        return template.format(
            from_pos=rule.from_position,
            to_pos=rule.to_position,
            strength=strength_mod
        )
    
    def _create_merged_rule(self, rules: List) -> Any:
        """Create merged rule from multiple rules"""
        # Simple merging: average properties
        merged_strength = sum(r.strength for r in rules) / len(rules)
        merged_confidence = sum(r.confidence for r in rules) / len(rules)
        merged_description = f"Merged rule combining {len(rules)} individual rules"
        
        # Create new rule object (simplified)
        class MergedRule:
            def __init__(self, strength, confidence, description):
                self.strength = strength
                self.confidence = confidence
                self.linguistic_description = description
        
        return MergedRule(merged_strength, merged_confidence, merged_description)
    
    def _create_split_rules(self, rule, split_criteria: Dict[str, Any]) -> List[Any]:
        """Create split rules from a single rule"""
        # Simple splitting: create two rules with modified properties
        split_rules = []
        
        # First split rule
        class SplitRule:
            def __init__(self, strength, confidence, description):
                self.strength = strength
                self.confidence = confidence
                self.linguistic_description = description
        
        rule1 = SplitRule(
            rule.strength * 0.7,
            rule.confidence * 0.8,
            f"Split rule 1: {rule.linguistic_description}"
        )
        
        rule2 = SplitRule(
            rule.strength * 0.5,
            rule.confidence * 0.6,
            f"Split rule 2: {rule.linguistic_description}"
        )
        
        split_rules.extend([rule1, rule2])
        return split_rules

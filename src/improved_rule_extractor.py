#!/usr/bin/env python3
"""
Улучшенный извлекатель правил для Fuzzy Attention Networks
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SemanticFuzzyRule:
    """Семантическое нечеткое правило"""
    rule_id: str
    semantic_type: str
    confidence: float
    attention_strength: float
    text_tokens: List[str]
    image_tokens: List[str]
    description: str
    conditions: Dict[str, Any]
    conclusion: str


class ImprovedRuleExtractor:
    """Улучшенный извлекатель семантических правил"""
    
    def __init__(self, 
                 attention_threshold: float = 0.1,
                 strong_threshold: float = 0.15,
                 max_rules_per_head: int = 10):
        """
        Инициализация извлекателя правил
        
        Args:
            attention_threshold: Минимальный порог внимания
            strong_threshold: Порог для сильных связей
            max_rules_per_head: Максимальное количество правил на голову
        """
        self.attention_threshold = attention_threshold
        self.strong_threshold = strong_threshold
        self.max_rules_per_head = max_rules_per_head
        
        # Семантические типы правил
        self.semantic_types = {
            'text_to_image': 'Текст-Изображение',
            'image_to_text': 'Изображение-Текст', 
            'text_to_text': 'Текст-Текст',
            'image_to_image': 'Изображение-Изображение',
            'cross_modal': 'Межмодальное',
            'attention_focus': 'Фокус внимания'
        }
    
    def extract_semantic_rules(self, 
                             attention_weights: torch.Tensor,
                             text_tokens: List[str],
                             image_tokens: Optional[List[str]] = None,
                             class_names: Optional[List[str]] = None,
                             head_idx: int = 0) -> List[SemanticFuzzyRule]:
        """
        Извлекает семантические правила из attention weights
        
        Args:
            attention_weights: Веса внимания [batch, seq_len, seq_len]
            text_tokens: Текстовые токены
            image_tokens: Изображения токены (опционально)
            class_names: Названия классов (опционально)
            head_idx: Индекс головы внимания
            
        Returns:
            Список семантических правил
        """
        rules = []
        
        # Получаем attention weights для выбранной головы
        if attention_weights.dim() == 3:
            attn = attention_weights[head_idx]
        else:
            attn = attention_weights
            
        seq_len = attn.shape[0]
        
        # Если image_tokens не предоставлены, создаем их
        if image_tokens is None:
            image_tokens = [f"img_{i}" for i in range(seq_len - len(text_tokens))]
        
        # Определяем границы текста и изображения
        text_len = len(text_tokens)
        image_start = text_len
        
        # Извлекаем правила для разных типов связей
        rules.extend(self._extract_text_to_image_rules(attn, text_tokens, image_tokens, text_len, image_start))
        rules.extend(self._extract_image_to_text_rules(attn, text_tokens, image_tokens, text_len, image_start))
        rules.extend(self._extract_text_to_text_rules(attn, text_tokens, text_len))
        rules.extend(self._extract_image_to_image_rules(attn, image_tokens, text_len, image_start))
        rules.extend(self._extract_cross_modal_rules(attn, text_tokens, image_tokens, text_len, image_start))
        rules.extend(self._extract_attention_focus_rules(attn, text_tokens, image_tokens, text_len, image_start))
        
        # Сортируем по уверенности
        rules.sort(key=lambda x: x.confidence, reverse=True)
        
        # Ограничиваем количество правил
        return rules[:self.max_rules_per_head]
    
    def _extract_text_to_image_rules(self, attn, text_tokens, image_tokens, text_len, image_start):
        """Извлекает правила текст -> изображение"""
        rules = []
        
        for i in range(text_len):
            for j in range(image_start, attn.shape[1]):
                if j - image_start < len(image_tokens):
                    strength = attn[i, j].item()
                    
                    if strength >= self.attention_threshold:
                        rule_type = 'text_to_image'
                        confidence = min(strength * 2, 1.0)
                        
                        rule = SemanticFuzzyRule(
                            rule_id=f"T2I_{i}_{j}",
                            semantic_type=rule_type,
                            confidence=confidence,
                            attention_strength=strength,
                            text_tokens=[text_tokens[i]],
                            image_tokens=[image_tokens[j - image_start]],
                            description=f"Токен '{text_tokens[i]}' влияет на изображение '{image_tokens[j - image_start]}'",
                            conditions={
                                'text_token': text_tokens[i],
                                'image_token': image_tokens[j - image_start],
                                'attention_strength': strength
                            },
                            conclusion=f"IF text='{text_tokens[i]}' THEN focus_on_image='{image_tokens[j - image_start]}'"
                        )
                        rules.append(rule)
        
        return rules
    
    def _extract_image_to_text_rules(self, attn, text_tokens, image_tokens, text_len, image_start):
        """Извлекает правила изображение -> текст"""
        rules = []
        
        for i in range(image_start, attn.shape[0]):
            if i - image_start < len(image_tokens):
                for j in range(text_len):
                    strength = attn[i, j].item()
                    
                    if strength >= self.attention_threshold:
                        rule_type = 'image_to_text'
                        confidence = min(strength * 2, 1.0)
                        
                        rule = SemanticFuzzyRule(
                            rule_id=f"I2T_{i}_{j}",
                            semantic_type=rule_type,
                            confidence=confidence,
                            attention_strength=strength,
                            text_tokens=[text_tokens[j]],
                            image_tokens=[image_tokens[i - image_start]],
                            description=f"Изображение '{image_tokens[i - image_start]}' влияет на токен '{text_tokens[j]}'",
                            conditions={
                                'image_token': image_tokens[i - image_start],
                                'text_token': text_tokens[j],
                                'attention_strength': strength
                            },
                            conclusion=f"IF image='{image_tokens[i - image_start]}' THEN focus_on_text='{text_tokens[j]}'"
                        )
                        rules.append(rule)
        
        return rules
    
    def _extract_text_to_text_rules(self, attn, text_tokens, text_len):
        """Извлекает правила текст -> текст"""
        rules = []
        
        for i in range(text_len):
            for j in range(text_len):
                if i != j:
                    strength = attn[i, j].item()
                    
                    if strength >= self.attention_threshold:
                        rule_type = 'text_to_text'
                        confidence = min(strength * 1.5, 1.0)
                        
                        rule = SemanticFuzzyRule(
                            rule_id=f"T2T_{i}_{j}",
                            semantic_type=rule_type,
                            confidence=confidence,
                            attention_strength=strength,
                            text_tokens=[text_tokens[i], text_tokens[j]],
                            image_tokens=[],
                            description=f"Токен '{text_tokens[i]}' связан с токеном '{text_tokens[j]}'",
                            conditions={
                                'source_token': text_tokens[i],
                                'target_token': text_tokens[j],
                                'attention_strength': strength
                            },
                            conclusion=f"IF text='{text_tokens[i]}' THEN also_consider='{text_tokens[j]}'"
                        )
                        rules.append(rule)
        
        return rules
    
    def _extract_image_to_image_rules(self, attn, image_tokens, text_len, image_start):
        """Извлекает правила изображение -> изображение"""
        rules = []
        
        for i in range(image_start, attn.shape[0]):
            if i - image_start < len(image_tokens):
                for j in range(image_start, attn.shape[1]):
                    if j - image_start < len(image_tokens) and i != j:
                        strength = attn[i, j].item()
                        
                        if strength >= self.attention_threshold:
                            rule_type = 'image_to_image'
                            confidence = min(strength * 1.5, 1.0)
                            
                            rule = SemanticFuzzyRule(
                                rule_id=f"I2I_{i}_{j}",
                                semantic_type=rule_type,
                                confidence=confidence,
                                attention_strength=strength,
                                text_tokens=[],
                                image_tokens=[image_tokens[i - image_start], image_tokens[j - image_start]],
                                description=f"Изображение '{image_tokens[i - image_start]}' связано с '{image_tokens[j - image_start]}'",
                                conditions={
                                    'source_image': image_tokens[i - image_start],
                                    'target_image': image_tokens[j - image_start],
                                    'attention_strength': strength
                                },
                                conclusion=f"IF image='{image_tokens[i - image_start]}' THEN also_consider='{image_tokens[j - image_start]}'"
                            )
                            rules.append(rule)
        
        return rules
    
    def _extract_cross_modal_rules(self, attn, text_tokens, image_tokens, text_len, image_start):
        """Извлекает межмодальные правила"""
        rules = []
        
        # Ищем сильные связи между текстом и изображением
        for i in range(text_len):
            for j in range(image_start, attn.shape[1]):
                if j - image_start < len(image_tokens):
                    strength = attn[i, j].item()
                    
                    if strength >= self.strong_threshold:
                        rule_type = 'cross_modal'
                        confidence = min(strength * 2.5, 1.0)
                        
                        rule = SemanticFuzzyRule(
                            rule_id=f"CM_{i}_{j}",
                            semantic_type=rule_type,
                            confidence=confidence,
                            attention_strength=strength,
                            text_tokens=[text_tokens[i]],
                            image_tokens=[image_tokens[j - image_start]],
                            description=f"Сильная межмодальная связь: '{text_tokens[i]}' ↔ '{image_tokens[j - image_start]}'",
                            conditions={
                                'text_token': text_tokens[i],
                                'image_token': image_tokens[j - image_start],
                                'attention_strength': strength,
                                'is_strong': True
                            },
                            conclusion=f"IF text='{text_tokens[i]}' AND image='{image_tokens[j - image_start]}' THEN high_confidence"
                        )
                        rules.append(rule)
        
        return rules
    
    def _extract_attention_focus_rules(self, attn, text_tokens, image_tokens, text_len, image_start):
        """Извлекает правила фокуса внимания"""
        rules = []
        
        # Находим токены с максимальным вниманием
        for i in range(attn.shape[0]):
            max_attn = attn[i].max().item()
            max_idx = attn[i].argmax().item()
            
            if max_attn >= self.strong_threshold:
                if i < text_len:
                    # Текстовый токен с максимальным вниманием
                    rule_type = 'attention_focus'
                    confidence = min(max_attn * 2, 1.0)
                    
                    if max_idx < text_len:
                        target_token = text_tokens[max_idx]
                        target_type = "text"
                    else:
                        target_token = image_tokens[max_idx - image_start] if max_idx - image_start < len(image_tokens) else f"img_{max_idx - image_start}"
                        target_type = "image"
                    
                    rule = SemanticFuzzyRule(
                        rule_id=f"AF_{i}_{max_idx}",
                        semantic_type=rule_type,
                        confidence=confidence,
                        attention_strength=max_attn,
                        text_tokens=[text_tokens[i]] if i < text_len else [],
                        image_tokens=[image_tokens[i - image_start]] if i >= image_start and i - image_start < len(image_tokens) else [],
                        description=f"Токен '{text_tokens[i] if i < text_len else image_tokens[i - image_start]}' фокусируется на '{target_token}' ({target_type})",
                        conditions={
                            'source_token': text_tokens[i] if i < text_len else image_tokens[i - image_start],
                            'target_token': target_token,
                            'target_type': target_type,
                            'attention_strength': max_attn
                        },
                        conclusion=f"IF focus_on='{text_tokens[i] if i < text_len else image_tokens[i - image_start]}' THEN prioritize='{target_token}'"
                    )
                    rules.append(rule)
        
        return rules
    
    def generate_rule_summary(self, rules: List[SemanticFuzzyRule]) -> Dict[str, Any]:
        """Генерирует сводку по правилам"""
        if not rules:
            return {
                'total_rules': 0,
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0,
                'avg_strength': 0.0,
                'rule_types': {},
                'text_summary': "Правила не найдены"
            }
        
        # Группируем по типам
        by_type = {}
        for rule in rules:
            if rule.semantic_type not in by_type:
                by_type[rule.semantic_type] = []
            by_type[rule.semantic_type].append(rule)
        
        # Вычисляем статистики
        confidences = [rule.confidence for rule in rules]
        strengths = [rule.attention_strength for rule in rules]
        
        summary = {
            'total_rules': len(rules),
            'avg_confidence': sum(confidences) / len(confidences),
            'max_confidence': max(confidences),
            'min_confidence': min(confidences),
            'avg_strength': sum(strengths) / len(strengths),
            'rule_types': {rule_type: len(type_rules) for rule_type, type_rules in by_type.items()},
            'text_summary': f"Найдено {len(rules)} правил"
        }
        
        return summary

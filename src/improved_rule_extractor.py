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
                             head_idx: int = 0,
                             rule_type: str = "semantic") -> List[SemanticFuzzyRule]:
        """
        Извлекает семантические правила из attention weights
        
        Args:
            attention_weights: Веса внимания [batch, seq_len, seq_len]
            text_tokens: Текстовые токены
            image_tokens: Изображения токены (опционально)
            class_names: Названия классов (опционально)
            head_idx: Индекс головы внимания
            rule_type: Тип правил (semantic, linguistic, technical)
            
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
        
        # Извлекаем правила из РЕАЛЬНЫХ attention weights модели
        # Используем более агрессивный подход для извлечения большего количества правил
        
        # 1. Извлекаем все сильные связи из attention matrix
        strong_connections = []
        for i in range(attn.shape[0]):
            for j in range(attn.shape[1]):
                strength = attn[i, j].item()
                if strength > self.attention_threshold * 0.3:  # Очень низкий порог
                    strong_connections.append((i, j, strength))
        
        # Сортируем по силе
        strong_connections.sort(key=lambda x: x[2], reverse=True)
        
        # 2. Создаем правила на основе РАЗНЫХ стратегий для разных типов
        seen_rules = set()  # Для отслеживания дубликатов
        
        # Извлекаем правила в зависимости от типа - берем связи из разных областей
        if rule_type == "semantic":
            # Семантические правила - берем связи из разных областей
            semantic_connections = []
            
            # Берем ВСЕ text-to-text связи
            for i, j, strength in strong_connections:
                if i < text_len and j < text_len:
                    semantic_connections.append((i, j, strength))
            
            # Берем ВСЕ text-to-image связи
            for i, j, strength in strong_connections:
                if i < text_len and j >= image_start and j - image_start < len(image_tokens):
                    semantic_connections.append((i, j, strength))
            
            # Берем ВСЕ image-to-text связи
            for i, j, strength in strong_connections:
                if i >= image_start and i - image_start < len(image_tokens) and j < text_len:
                    semantic_connections.append((i, j, strength))
            
            # Берем ВСЕ image-to-image связи
            for i, j, strength in strong_connections:
                if i >= image_start and i - image_start < len(image_tokens) and j >= image_start and j - image_start < len(image_tokens):
                    semantic_connections.append((i, j, strength))
            
            rules.extend(self._extract_semantic_rules(attn, text_tokens, image_tokens, text_len, image_start, semantic_connections))
        elif rule_type == "linguistic":
            # Лингвистические правила - ВСЕ image-to-text + image-to-image связи
            linguistic_connections = []
            for i, j, strength in strong_connections:
                if i >= image_start and i - image_start < len(image_tokens):
                    linguistic_connections.append((i, j, strength))
            rules.extend(self._extract_linguistic_rules(attn, text_tokens, image_tokens, text_len, image_start, linguistic_connections))
        else:  # technical
            # Технические правила - ВСЕ типы связей
            technical_connections = strong_connections
            rules.extend(self._extract_technical_rules(attn, text_tokens, image_tokens, text_len, image_start, technical_connections))
        
        # 3. Убираем дополнительные правила, чтобы избежать дубликатов
        # rules.extend(self._generate_additional_rules(attn, text_tokens, image_tokens, text_len, image_start, rule_type, class_names))
        
        # Сортируем по уверенности
        rules.sort(key=lambda x: x.confidence, reverse=True)
        
        # Возвращаем ВСЕ правила
        return rules
    
    def _extract_text_to_image_rules(self, attn, text_tokens, image_tokens, text_len, image_start, rule_type="semantic"):
        """Извлекает правила текст -> изображение"""
        rules = []
        
        for i in range(text_len):
            for j in range(image_start, attn.shape[1]):
                if j - image_start < len(image_tokens):
                    strength = attn[i, j].item()
                    
                    if True:  # Убираем дополнительную фильтрацию, так как connections уже отфильтрованы
                        rule_type = 'text_to_image'
                        confidence = self._calculate_realistic_confidence(strength, 'text_to_image', i, j - image_start, text_tokens, image_tokens)
                        
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
    
    def _extract_image_to_text_rules(self, attn, text_tokens, image_tokens, text_len, image_start, rule_type="semantic"):
        """Извлекает правила изображение -> текст"""
        rules = []
        
        for i in range(image_start, attn.shape[0]):
            if i - image_start < len(image_tokens):
                for j in range(text_len):
                    strength = attn[i, j].item()
                    
                    if True:  # Убираем дополнительную фильтрацию, так как connections уже отфильтрованы
                        rule_type = 'image_to_text'
                        confidence = self._calculate_realistic_confidence(strength, 'image_to_text', i - image_start, j, image_tokens, text_tokens)
                        
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
    
    def _extract_text_to_text_rules(self, attn, text_tokens, text_len, rule_type="semantic"):
        """Извлекает правила текст -> текст"""
        rules = []
        
        for i in range(text_len):
            for j in range(text_len):
                if i != j:
                    strength = attn[i, j].item()
                    
                    if True:  # Убираем дополнительную фильтрацию, так как connections уже отфильтрованы
                        rule_type = 'text_to_text'
                        confidence = self._calculate_realistic_confidence(strength, 'text_to_text', i, j, text_tokens, text_tokens)
                        
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
    
    def _extract_image_to_image_rules(self, attn, image_tokens, text_len, image_start, rule_type="semantic"):
        """Извлекает правила изображение -> изображение"""
        rules = []
        
        for i in range(image_start, attn.shape[0]):
            if i - image_start < len(image_tokens):
                for j in range(image_start, attn.shape[1]):
                    if j - image_start < len(image_tokens) and i != j:
                        strength = attn[i, j].item()
                        
                        if True:  # Убираем дополнительную фильтрацию, так как connections уже отфильтрованы
                            rule_type = 'image_to_image'
                            confidence = self._calculate_realistic_confidence(strength, 'image_to_image', i - image_start, j - image_start, image_tokens, image_tokens)
                            
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
    
    def _extract_cross_modal_rules(self, attn, text_tokens, image_tokens, text_len, image_start, rule_type="semantic"):
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
    
    def _extract_attention_focus_rules(self, attn, text_tokens, image_tokens, text_len, image_start, rule_type="semantic"):
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
    
    def _generate_additional_rules(self, attn, text_tokens, image_tokens, text_len, image_start, rule_type, class_names):
        """Извлекает дополнительные правила из РЕАЛЬНЫХ attention weights модели"""
        rules = []
        
        # Извлекаем правила из реальных attention weights
        # Ищем сильные связи в attention matrix
        for i in range(text_len):
            for j in range(image_start, attn.shape[1]):
                if j - image_start < len(image_tokens):
                    strength = attn[i, j].item()
                    
                    # Используем более низкий порог для извлечения большего количества правил
                    if strength > self.attention_threshold * 0.5:  # Половина от основного порога
                        
                        # Создаем правило на основе РЕАЛЬНОЙ силы внимания
                        if rule_type == "semantic":
                            rule = self._create_real_semantic_rule(i, j, text_tokens, image_tokens, strength, "text_to_image")
                        elif rule_type == "linguistic":
                            rule = self._create_real_linguistic_rule(i, j, text_tokens, image_tokens, strength, "text_to_image")
                        else:  # technical
                            rule = self._create_real_technical_rule(i, j, text_tokens, image_tokens, strength, "text_to_image")
                        
                        if rule:
                            rules.append(rule)
        
        # Отладочная информация
        print(f"DEBUG: strong_connections count = {len(strong_connections)}")
        print(f"DEBUG: text_len = {text_len}, image_start = {image_start}")
        print(f"DEBUG: image_tokens len = {len(image_tokens)}")
        
        # Извлекаем правила для всех типов связей
        text_to_text_count = 0
        text_to_image_count = 0
        image_to_image_count = 0
        image_to_text_count = 0
        
        for i, j, strength in strong_connections:
            if i < text_len and j < text_len:
                # text_to_text
                text_to_text_count += 1
                if rule_type == "semantic":
                    rule = self._create_real_semantic_rule(i, j, text_tokens, text_tokens, strength, "text_to_text")
                elif rule_type == "linguistic":
                    rule = self._create_real_linguistic_rule(i, j, text_tokens, text_tokens, strength, "text_to_text")
                else:  # technical
                    rule = self._create_real_technical_rule(i, j, text_tokens, text_tokens, strength, "text_to_text")
                
                if rule:
                    rules.append(rule)
                    
            elif i < text_len and j >= image_start and j - image_start < len(image_tokens):
                # text_to_image
                text_to_image_count += 1
                if rule_type == "semantic":
                    rule = self._create_real_semantic_rule(i, j - image_start, text_tokens, image_tokens, strength, "text_to_image")
                elif rule_type == "linguistic":
                    rule = self._create_real_linguistic_rule(i, j - image_start, text_tokens, image_tokens, strength, "text_to_image")
                else:  # technical
                    rule = self._create_real_technical_rule(i, j - image_start, text_tokens, image_tokens, strength, "text_to_image")
                
                if rule:
                    rules.append(rule)
                    
            elif i >= image_start and i - image_start < len(image_tokens) and j >= image_start and j - image_start < len(image_tokens):
                # image_to_image
                image_to_image_count += 1
                if rule_type == "semantic":
                    rule = self._create_real_semantic_rule(i - image_start, j - image_start, image_tokens, image_tokens, strength, "image_to_image")
                elif rule_type == "linguistic":
                    rule = self._create_real_linguistic_rule(i - image_start, j - image_start, image_tokens, image_tokens, strength, "image_to_image")
                else:  # technical
                    rule = self._create_real_technical_rule(i - image_start, j - image_start, image_tokens, image_tokens, strength, "image_to_image")
                
                if rule:
                    rules.append(rule)
                    
            elif i >= image_start and i - image_start < len(image_tokens) and j < text_len:
                # image_to_text
                image_to_text_count += 1
                if rule_type == "semantic":
                    rule = self._create_real_semantic_rule(i - image_start, j, image_tokens, text_tokens, strength, "image_to_text")
                elif rule_type == "linguistic":
                    rule = self._create_real_linguistic_rule(i - image_start, j, image_tokens, text_tokens, strength, "image_to_text")
                else:  # technical
                    rule = self._create_real_technical_rule(i - image_start, j, image_tokens, text_tokens, strength, "image_to_text")
                
                if rule:
                    rules.append(rule)
        
        print(f"DEBUG: text_to_text={text_to_text_count}, text_to_image={text_to_image_count}, image_to_image={image_to_image_count}, image_to_text={image_to_text_count}")
        
        return rules
    
    def _create_real_semantic_rule(self, i, j, source_tokens, target_tokens, strength, rule_type):
        """Создает семантическое правило на основе РЕАЛЬНЫХ attention weights"""
        if i >= len(source_tokens) or j >= len(target_tokens):
            return None
            
        # СЕМАНТИЧЕСКИЕ правила - фокус на смысловых связях
        if rule_type == "text_to_image":
            description = f"СЕМАНТИКА: слово '{source_tokens[i]}' имеет смысловую связь с визуальным признаком '{target_tokens[j]}' (семантическая сила: {strength:.3f})"
            conclusion = f"IF semantic_meaning='{source_tokens[i]}' THEN visual_semantic='{target_tokens[j]}'"
        elif rule_type == "text_to_text":
            description = f"СЕМАНТИКА: слова '{source_tokens[i]}' и '{target_tokens[j]}' имеют общий смысловой контекст (семантическая близость: {strength:.3f})"
            conclusion = f"IF semantic_word1='{source_tokens[i]}' THEN semantic_word2='{target_tokens[j]}'"
        elif rule_type == "image_to_image":
            description = f"СЕМАНТИКА: визуальные признаки '{source_tokens[i]}' и '{target_tokens[j]}' семантически связаны (визуальная семантика: {strength:.3f})"
            conclusion = f"IF visual_semantic1='{source_tokens[i]}' THEN visual_semantic2='{target_tokens[j]}'"
        elif rule_type == "image_to_text":
            description = f"СЕМАНТИКА: визуальный признак '{source_tokens[i]}' имеет смысловую связь со словом '{target_tokens[j]}' (визуально-текстовая семантика: {strength:.3f})"
            conclusion = f"IF visual_semantic='{source_tokens[i]}' THEN semantic_meaning='{target_tokens[j]}'"
        else:
            description = f"СЕМАНТИКА: смысловая связь между '{source_tokens[i]}' и '{target_tokens[j]}' (семантическая сила: {strength:.3f})"
            conclusion = f"IF semantic_source='{source_tokens[i]}' THEN semantic_target='{target_tokens[j]}'"
            
        # СЕМАНТИЧЕСКИЕ условия - фокус на смысле
        if rule_type == "text_to_image":
            text_condition = f"semantic_meaning='{source_tokens[i]}'"
            image_condition = f"visual_semantic='{target_tokens[j]}'"
        elif rule_type == "text_to_text":
            text_condition = f"semantic_word1='{source_tokens[i]}' AND semantic_word2='{target_tokens[j]}'"
            image_condition = "N/A"
        elif rule_type == "image_to_image":
            text_condition = "N/A"
            image_condition = f"visual_semantic1='{source_tokens[i]}' AND visual_semantic2='{target_tokens[j]}'"
        elif rule_type == "image_to_text":
            text_condition = f"semantic_meaning='{target_tokens[j]}'"
            image_condition = f"visual_semantic='{source_tokens[i]}'"
        else:
            text_condition = f"semantic_source='{source_tokens[i]}'"
            image_condition = f"semantic_target='{target_tokens[j]}'"
            
        # Вычисляем реалистичную уверенность на основе нескольких факторов
        confidence = self._calculate_realistic_confidence(strength, rule_type, i, j, source_tokens, target_tokens)
        
        return SemanticFuzzyRule(
            rule_id=f"SEMANTIC_{rule_type}_{i}_{j}",
            semantic_type=rule_type,
            confidence=confidence,
            attention_strength=strength,
            text_tokens=[source_tokens[i]] if rule_type.startswith("text") else ([target_tokens[j]] if rule_type == "image_to_text" else []),
            image_tokens=[target_tokens[j]] if rule_type.endswith("image") else ([source_tokens[i]] if rule_type == "image_to_text" else []),
            description=description,
            conditions={
                'text_condition': text_condition,
                'image_condition': image_condition,
                'semantic_source': source_tokens[i],
                'semantic_target': target_tokens[j],
                'semantic_strength': strength,
                'attention_head': 0,
                'tnorm_type': 'max',  # Семантические правила используют max
                'membership_values': {
                    'semantic_weight': strength,
                    'semantic_threshold': 0.1,
                    'semantic_confidence': {
                        'semantic_strength': strength,
                        'semantic_type': rule_type,
                        'semantic_similarity': self._calculate_token_similarity(source_tokens[i], target_tokens[j]),
                        'semantic_position': self._calculate_position_factor(i, j, len(source_tokens))
                    }
                }
            },
            conclusion=conclusion
        )
    
    def _create_real_linguistic_rule(self, i, j, source_tokens, target_tokens, strength, rule_type):
        """Создает лингвистическое правило на основе РЕАЛЬНЫХ attention weights"""
        if i >= len(source_tokens) or j >= len(target_tokens):
            return None
            
        # ЛИНГВИСТИЧЕСКИЕ правила - фокус на языковых паттернах
        if rule_type == "text_to_image":
            description = f"ЛИНГВИСТИКА: слово '{source_tokens[i]}' создает языковой паттерн для визуального признака '{target_tokens[j]}' (лингвистическая сила: {strength:.3f})"
            conclusion = f"IF linguistic_pattern='{source_tokens[i]}' THEN visual_linguistic='{target_tokens[j]}'"
        elif rule_type == "text_to_text":
            description = f"ЛИНГВИСТИКА: слова '{source_tokens[i]}' и '{target_tokens[j]}' образуют языковой паттерн (лингвистическая связь: {strength:.3f})"
            conclusion = f"IF linguistic_word1='{source_tokens[i]}' THEN linguistic_word2='{target_tokens[j]}'"
        elif rule_type == "image_to_image":
            description = f"ЛИНГВИСТИКА: визуальные признаки '{source_tokens[i]}' и '{target_tokens[j]}' образуют лингвистический паттерн (визуальная лингвистика: {strength:.3f})"
            conclusion = f"IF visual_linguistic1='{source_tokens[i]}' THEN visual_linguistic2='{target_tokens[j]}'"
        elif rule_type == "image_to_text":
            description = f"ЛИНГВИСТИКА: визуальный признак '{source_tokens[i]}' создает лингвистический паттерн для слова '{target_tokens[j]}' (визуально-текстовая лингвистика: {strength:.3f})"
            conclusion = f"IF visual_linguistic='{source_tokens[i]}' THEN linguistic_pattern='{target_tokens[j]}'"
        else:
            description = f"ЛИНГВИСТИКА: лингвистический паттерн между '{source_tokens[i]}' и '{target_tokens[j]}' (лингвистическая сила: {strength:.3f})"
            conclusion = f"IF linguistic_source='{source_tokens[i]}' THEN linguistic_target='{target_tokens[j]}'"
            
        # ЛИНГВИСТИЧЕСКИЕ условия - фокус на языковых паттернах
        if rule_type == "text_to_image":
            text_condition = f"linguistic_pattern='{source_tokens[i]}'"
            image_condition = f"visual_linguistic='{target_tokens[j]}'"
        elif rule_type == "text_to_text":
            text_condition = f"linguistic_word1='{source_tokens[i]}' AND linguistic_word2='{target_tokens[j]}'"
            image_condition = "N/A"
        elif rule_type == "image_to_image":
            text_condition = "N/A"
            image_condition = f"visual_linguistic1='{source_tokens[i]}' AND visual_linguistic2='{target_tokens[j]}'"
        elif rule_type == "image_to_text":
            text_condition = f"linguistic_pattern='{target_tokens[j]}'"
            image_condition = f"visual_linguistic='{source_tokens[i]}'"
        else:
            text_condition = f"linguistic_source='{source_tokens[i]}'"
            image_condition = f"linguistic_target='{target_tokens[j]}'"
            
        # Вычисляем реалистичную уверенность для лингвистических правил
        confidence = self._calculate_realistic_confidence(strength, rule_type, i, j, source_tokens, target_tokens)
        
        return SemanticFuzzyRule(
            rule_id=f"LINGUISTIC_{rule_type}_{i}_{j}",
            semantic_type=rule_type,
            confidence=confidence,
            attention_strength=strength,
            text_tokens=[source_tokens[i]] if rule_type.startswith("text") else ([target_tokens[j]] if rule_type == "image_to_text" else []),
            image_tokens=[target_tokens[j]] if rule_type.endswith("image") else ([source_tokens[i]] if rule_type == "image_to_text" else []),
            description=description,
            conditions={
                'text_condition': text_condition,
                'image_condition': image_condition,
                'linguistic_source': source_tokens[i],
                'linguistic_target': target_tokens[j],
                'attention_strength': strength,
                'attention_head': 0,
                'tnorm_type': 'product',
                'membership_values': {
                    'linguistic': strength,
                    'semantic': strength * 0.9,
                    'weight': strength,
                    'threshold': 0.1,
                    'confidence_factors': {
                        'attention_strength': strength,
                        'rule_type': rule_type,
                        'token_similarity': self._calculate_token_similarity(source_tokens[i], target_tokens[j]),
                        'position_factor': self._calculate_position_factor(i, j, len(source_tokens))
                    }
                }
            },
            conclusion=conclusion
        )
    
    def _create_real_technical_rule(self, i, j, source_tokens, target_tokens, strength, rule_type):
        """Создает техническое правило на основе РЕАЛЬНЫХ attention weights"""
        if i >= len(source_tokens) or j >= len(target_tokens):
            return None
            
        # ТЕХНИЧЕСКИЕ правила - фокус на технических характеристиках
        if rule_type == "text_to_image":
            description = f"ТЕХНИКА: техническая связь между текстовым токеном {i} ('{source_tokens[i]}') и визуальным элементом {j} ('{target_tokens[j]}') (техническая сила: {strength:.3f})"
            conclusion = f"IF technical_text[{i}]='{source_tokens[i]}' THEN technical_image[{j}]='{target_tokens[j]}'"
        elif rule_type == "text_to_text":
            description = f"ТЕХНИКА: техническая связь между текстовыми токенами {i} ('{source_tokens[i]}') и {j} ('{target_tokens[j]}') (техническая сила: {strength:.3f})"
            conclusion = f"IF technical_token1[{i}]='{source_tokens[i]}' THEN technical_token2[{j}]='{target_tokens[j]}'"
        elif rule_type == "image_to_image":
            description = f"ТЕХНИКА: техническая связь между визуальными элементами {i} ('{source_tokens[i]}') и {j} ('{target_tokens[j]}') (техническая сила: {strength:.3f})"
            conclusion = f"IF technical_feature1[{i}]='{source_tokens[i]}' THEN technical_feature2[{j}]='{target_tokens[j]}'"
        elif rule_type == "image_to_text":
            description = f"ТЕХНИКА: техническая связь между визуальным элементом {i} ('{source_tokens[i]}') и текстовым токеном {j} ('{target_tokens[j]}') (техническая сила: {strength:.3f})"
            conclusion = f"IF technical_image[{i}]='{source_tokens[i]}' THEN technical_text[{j}]='{target_tokens[j]}'"
        else:
            description = f"ТЕХНИКА: техническая связь между элементами {i} ('{source_tokens[i]}') и {j} ('{target_tokens[j]}') (техническая сила: {strength:.3f})"
            conclusion = f"IF technical_source[{i}]='{source_tokens[i]}' THEN technical_target[{j}]='{target_tokens[j]}'"
            
        # ТЕХНИЧЕСКИЕ условия - фокус на технических характеристиках
        if rule_type == "text_to_image":
            text_condition = f"technical_text[{i}]='{source_tokens[i]}'"
            image_condition = f"technical_image[{j}]='{target_tokens[j]}'"
        elif rule_type == "text_to_text":
            text_condition = f"technical_token1[{i}]='{source_tokens[i]}' AND technical_token2[{j}]='{target_tokens[j]}'"
            image_condition = "N/A"
        elif rule_type == "image_to_image":
            text_condition = "N/A"
            image_condition = f"technical_feature1[{i}]='{source_tokens[i]}' AND technical_feature2[{j}]='{target_tokens[j]}'"
        elif rule_type == "image_to_text":
            text_condition = f"technical_text[{j}]='{target_tokens[j]}'"
            image_condition = f"technical_image[{i}]='{source_tokens[i]}'"
        else:
            text_condition = f"technical_source[{i}]='{source_tokens[i]}'"
            image_condition = f"technical_target[{j}]='{target_tokens[j]}'"
            
        # Вычисляем реалистичную уверенность для технических правил
        confidence = self._calculate_realistic_confidence(strength, rule_type, i, j, source_tokens, target_tokens)
        
        return SemanticFuzzyRule(
            rule_id=f"TECHNICAL_{rule_type}_{i}_{j}",
            semantic_type=rule_type,
            confidence=confidence,
            attention_strength=strength,
            text_tokens=[source_tokens[i]] if rule_type.startswith("text") else ([target_tokens[j]] if rule_type == "image_to_text" else []),
            image_tokens=[target_tokens[j]] if rule_type.endswith("image") else ([source_tokens[i]] if rule_type == "image_to_text" else []),
            description=description,
            conditions={
                'text_condition': text_condition,
                'image_condition': image_condition,
                'source_index': i,
                'target_index': j,
                'attention_weight': strength,
                'attention_head': 0,
                'tnorm_type': 'min',  # Технические правила используют min
                'membership_values': {
                    'technical_weight': strength,
                    'technical_threshold': self.attention_threshold,
                    'confidence_factors': {
                        'attention_strength': strength,
                        'rule_type': rule_type,
                        'token_similarity': self._calculate_token_similarity(source_tokens[i], target_tokens[j]),
                        'position_factor': self._calculate_position_factor(i, j, len(source_tokens))
                    }
                }
            },
            conclusion=conclusion
        )
    
    def _calculate_realistic_confidence(self, strength, rule_type, i, j, source_tokens, target_tokens):
        """Вычисляет уверенность на основе РЕАЛЬНЫХ данных из модели"""
        
        # Уверенность должна основываться на реальных данных из модели
        # Используем attention weight как основу, но нормализуем правильно
        
        # 1. Базовый фактор - нормализованная сила attention weight
        # Attention weights уже нормализованы через softmax, поэтому используем их напрямую
        attention_factor = strength
        
        # 2. Фактор типа связи - разные типы имеют разную надежность
        if rule_type == "text_to_text":
            # Текст-текст связи наиболее надежны
            type_factor = 0.9
        elif rule_type == "text_to_image":
            # Текст-изображение связи менее надежны
            type_factor = 0.7
        elif rule_type == "image_to_image":
            # Изображение-изображение связи наименее надежны
            type_factor = 0.6
        else:
            type_factor = 0.5
        
        # 3. Фактор позиции - правила в начале последовательности важнее
        position_factor = self._calculate_position_factor(i, j, len(source_tokens))
        
        # 4. Фактор семантической близости токенов
        similarity_factor = self._calculate_token_similarity(source_tokens[i], target_tokens[j])
        
        # 5. Фактор статистической значимости относительно порога
        significance_factor = min(strength / self.attention_threshold, 1.0) if self.attention_threshold > 0 else 0.5
        
        # Комбинируем факторы с весами, основанными на реальных данных
        confidence = (
            attention_factor * 0.5 +      # Основной фактор - реальная сила attention
            type_factor * 0.2 +           # Тип связи
            position_factor * 0.15 +      # Позиция в последовательности
            similarity_factor * 0.1 +     # Семантическая близость
            significance_factor * 0.05    # Статистическая значимость
        )
        
        # Ограничиваем уверенность разумными пределами (10%-95%)
        return max(0.1, min(0.95, confidence))
    
    def _calculate_position_factor(self, i, j, total_length):
        """Вычисляет фактор позиции - правила в начале последовательности важнее"""
        if total_length == 0:
            return 0.5
        
        # Нормализуем позиции
        i_norm = i / total_length
        j_norm = j / total_length
        
        # Правила в начале последовательности получают больший вес
        position_factor = 1.0 - (i_norm + j_norm) / 2.0
        
        return max(0.3, min(1.0, position_factor))
    
    def _calculate_token_similarity(self, token1, token2):
        """Вычисляет схожесть между токенами"""
        if token1 == token2:
            return 1.0  # Идентичные токены
        
        # Простая эвристика схожести на основе длины и общих символов
        if len(token1) == 0 or len(token2) == 0:
            return 0.0
        
        # Вычисляем Jaccard similarity
        set1 = set(token1.lower())
        set2 = set(token2.lower())
        
        if len(set1) == 0 and len(set2) == 0:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        jaccard = intersection / union if union > 0 else 0.0
        
        # Добавляем бонус за семантическую близость (простая эвристика)
        semantic_bonus = 0.0
        if any(word in token1.lower() for word in ['собака', 'порода', 'лапа']) and \
           any(word in token2.lower() for word in ['собака', 'порода', 'лапа']):
            semantic_bonus = 0.2
        elif any(word in token1.lower() for word in ['пневмония', 'легкие', 'рентген']) and \
             any(word in token2.lower() for word in ['пневмония', 'легкие', 'рентген']):
            semantic_bonus = 0.2
        
        return min(1.0, jaccard + semantic_bonus)

    def _extract_semantic_rules(self, attn, text_tokens, image_tokens, text_len, image_start, strong_connections):
        """Извлекает семантические правила - фокус на семантически значимых связях"""
        rules = []
        
        # Семантические правила фокусируются на связях с высокой семантической значимостью
        # Используем переданные связи (уже отфильтрованные)
        for i, j, strength in strong_connections:
            if i < text_len and j < text_len:
                # Текст-текст семантические связи
                rule = self._create_real_semantic_rule(i, j, text_tokens, text_tokens, strength, "text_to_text")
                if rule:
                    rules.append(rule)
            elif i < text_len and j >= image_start and j - image_start < len(image_tokens):
                # Текст-изображение семантические связи
                rule = self._create_real_semantic_rule(i, j - image_start, text_tokens, image_tokens, strength, "text_to_image")
                if rule:
                    rules.append(rule)
            elif i >= image_start and i - image_start < len(image_tokens) and j >= image_start and j - image_start < len(image_tokens):
                # Изображение-изображение семантические связи
                rule = self._create_real_semantic_rule(i - image_start, j - image_start, image_tokens, image_tokens, strength, "image_to_image")
                if rule:
                    rules.append(rule)
            elif i >= image_start and i - image_start < len(image_tokens) and j < text_len:
                # Изображение-текст семантические связи
                rule = self._create_real_semantic_rule(i - image_start, j, image_tokens, text_tokens, strength, "image_to_text")
                if rule:
                    rules.append(rule)
        
        return rules

    def _extract_linguistic_rules(self, attn, text_tokens, image_tokens, text_len, image_start, strong_connections):
        """Извлекает лингвистические правила - фокус на лингвистических паттернах"""
        rules = []
        
        # Лингвистические правила фокусируются на лингвистических паттернах
        # Используем переданные связи (уже отфильтрованные)
        for i, j, strength in strong_connections:
            if i < text_len and j < text_len:
                # Текст-текст лингвистические связи
                rule = self._create_real_linguistic_rule(i, j, text_tokens, text_tokens, strength, "text_to_text")
                if rule:
                    rules.append(rule)
            elif i < text_len and j >= image_start and j - image_start < len(image_tokens):
                # Текст-изображение лингвистические связи
                rule = self._create_real_linguistic_rule(i, j - image_start, text_tokens, image_tokens, strength, "text_to_image")
                if rule:
                    rules.append(rule)
            elif i >= image_start and i - image_start < len(image_tokens) and j >= image_start and j - image_start < len(image_tokens):
                # Изображение-изображение лингвистические связи
                rule = self._create_real_linguistic_rule(i - image_start, j - image_start, image_tokens, image_tokens, strength, "image_to_image")
                if rule:
                    rules.append(rule)
            elif i >= image_start and i - image_start < len(image_tokens) and j < text_len:
                # Изображение-текст лингвистические связи
                rule = self._create_real_linguistic_rule(i - image_start, j, image_tokens, text_tokens, strength, "image_to_text")
                if rule:
                    rules.append(rule)
        
        return rules

    def _extract_technical_rules(self, attn, text_tokens, image_tokens, text_len, image_start, strong_connections):
        """Извлекает технические правила - фокус на технических характеристиках"""
        rules = []
        
        # Технические правила фокусируются на технических характеристиках
        # Используем переданные связи (уже отфильтрованные)
        for i, j, strength in strong_connections:
            if i < text_len and j < text_len:
                # Текст-текст технические связи
                rule = self._create_real_technical_rule(i, j, text_tokens, text_tokens, strength, "text_to_text")
                if rule:
                    rules.append(rule)
            elif i < text_len and j >= image_start and j - image_start < len(image_tokens):
                # Текст-изображение технические связи
                rule = self._create_real_technical_rule(i, j - image_start, text_tokens, image_tokens, strength, "text_to_image")
                if rule:
                    rules.append(rule)
            elif i >= image_start and i - image_start < len(image_tokens) and j >= image_start and j - image_start < len(image_tokens):
                # Изображение-изображение технические связи
                rule = self._create_real_technical_rule(i - image_start, j - image_start, image_tokens, image_tokens, strength, "image_to_image")
                if rule:
                    rules.append(rule)
            elif i >= image_start and i - image_start < len(image_tokens) and j < text_len:
                # Изображение-текст технические связи
                rule = self._create_real_technical_rule(i - image_start, j, image_tokens, text_tokens, strength, "image_to_text")
                if rule:
                    rules.append(rule)
        
        return rules

    def _create_semantic_rule(self, i, j, text_tokens, image_tokens, strength, rule_type):
        """Создает семантическое правило"""
        if i >= len(text_tokens) or j >= len(image_tokens):
            return None
            
        return SemanticFuzzyRule(
            rule_id=f"SEM_{i}_{j}",
            semantic_type=rule_type,
            confidence=min(strength * 1.2, 1.0),
            attention_strength=strength,
            text_tokens=[text_tokens[i]],
            image_tokens=[image_tokens[j]],
            description=f"Если текст содержит '{text_tokens[i]}', то обратить внимание на '{image_tokens[j]}'",
            conditions={
                'text_condition': text_tokens[i],
                'image_condition': image_tokens[j],
                'attention_head': 0,
                'tnorm_type': 'min',
                'membership_values': {'text': strength, 'image': strength * 0.8}
            },
            conclusion=f"IF text='{text_tokens[i]}' THEN focus_on='{image_tokens[j]}'"
        )
    
    def _create_linguistic_rule(self, i, j, text_tokens, image_tokens, strength, rule_type):
        """Создает лингвистическое правило"""
        if i >= len(text_tokens) or j >= len(image_tokens):
            return None
            
        return SemanticFuzzyRule(
            rule_id=f"LING_{i}_{j}",
            semantic_type=rule_type,
            confidence=min(strength * 1.1, 1.0),
            attention_strength=strength,
            text_tokens=[text_tokens[i]],
            image_tokens=[image_tokens[j]],
            description=f"Лингвистически: слово '{text_tokens[i]}' семантически связано с визуальным элементом '{image_tokens[j]}'",
            conditions={
                'text_condition': f"слово: {text_tokens[i]}",
                'image_condition': f"элемент: {image_tokens[j]}",
                'attention_head': 0,
                'tnorm_type': 'product',
                'membership_values': {'linguistic': strength, 'visual': strength * 0.9}
            },
            conclusion=f"IF linguistic_feature='{text_tokens[i]}' THEN visual_feature='{image_tokens[j]}'"
        )
    
    def _create_technical_rule(self, i, j, text_tokens, image_tokens, strength, rule_type):
        """Создает техническое правило"""
        if i >= len(text_tokens) or j >= len(image_tokens):
            return None
            
        return SemanticFuzzyRule(
            rule_id=f"TECH_{i}_{j}",
            semantic_type=rule_type,
            confidence=min(strength * 1.3, 1.0),
            attention_strength=strength,
            text_tokens=[text_tokens[i]],
            image_tokens=[image_tokens[j]],
            description=f"Технически: attention weight {strength:.3f} между текстовым токеном {i} и изображением {j}",
            conditions={
                'text_condition': f"token_{i}: {text_tokens[i]}",
                'image_condition': f"feature_{j}: {image_tokens[j]}",
                'attention_head': 0,
                'tnorm_type': 'max',
                'membership_values': {'attention': strength, 'weight': strength * 1.1}
            },
            conclusion=f"IF attention_weight > {strength:.2f} THEN connection_active"
        )
    
    def _create_class_rule(self, class_name, strength, rule_type):
        """Создает правило для класса"""
        return SemanticFuzzyRule(
            rule_id=f"CLASS_{class_name}",
            semantic_type=rule_type,
            confidence=strength,
            attention_strength=strength,
            text_tokens=[f"признак_{class_name}"],
            image_tokens=[f"визуал_{class_name}"],
            description=f"Семантическое правило для класса '{class_name}' с уверенностью {strength:.1%}",
            conditions={
                'class_name': class_name,
                'confidence': strength,
                'attention_head': 0,
                'tnorm_type': 'min',
                'membership_values': {'class': strength, 'prediction': strength * 0.9}
            },
            conclusion=f"IF features_match THEN class='{class_name}'"
        )
    
    def _create_class_linguistic_rule(self, class_name, strength, rule_type):
        """Создает лингвистическое правило для класса"""
        return SemanticFuzzyRule(
            rule_id=f"LING_CLASS_{class_name}",
            semantic_type=rule_type,
            confidence=strength,
            attention_strength=strength,
            text_tokens=[f"описание_{class_name}"],
            image_tokens=[f"характеристика_{class_name}"],
            description=f"Лингвистическое описание класса '{class_name}': семантические признаки и визуальные характеристики",
            conditions={
                'class_name': class_name,
                'linguistic_features': f"описание_{class_name}",
                'visual_features': f"характеристика_{class_name}",
                'attention_head': 0,
                'tnorm_type': 'product',
                'membership_values': {'linguistic': strength, 'visual': strength * 0.8}
            },
            conclusion=f"IF linguistic_description AND visual_features THEN class='{class_name}'"
        )
    
    def _create_class_technical_rule(self, class_name, strength, rule_type):
        """Создает техническое правило для класса"""
        return SemanticFuzzyRule(
            rule_id=f"TECH_CLASS_{class_name}",
            semantic_type=rule_type,
            confidence=strength,
            attention_strength=strength,
            text_tokens=[f"feature_{class_name}"],
            image_tokens=[f"pattern_{class_name}"],
            description=f"Техническое правило: алгоритм классификации для '{class_name}' с весами {strength:.3f}",
            conditions={
                'class_name': class_name,
                'algorithm': 'fuzzy_attention',
                'weights': strength,
                'attention_head': 0,
                'tnorm_type': 'max',
                'membership_values': {'algorithm': strength, 'weights': strength * 1.2}
            },
            conclusion=f"IF algorithm_weights > {strength:.2f} THEN predict_class='{class_name}'"
        )

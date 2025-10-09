"""
Улучшенный модуль извлечения правил для Fuzzy Attention Networks
Создает более семантически осмысленные и интерпретируемые правила
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
import re

@dataclass
class SemanticFuzzyRule:
    """Улучшенное представление fuzzy правила с семантической информацией"""
    rule_id: str
    condition_text: str
    condition_image: str
    conclusion: str
    confidence: float
    strength: float
    semantic_type: str  # 'color', 'texture', 'shape', 'object', 'spatial'
    linguistic_description: str
    membership_values: Dict[str, float]
    attention_head: int
    tnorm_type: str

class ImprovedRuleExtractor:
    """Улучшенный извлекатель правил с семантическим анализом"""
    
    def __init__(self, 
                 attention_threshold: float = 0.1,
                 strong_threshold: float = 0.15,
                 max_rules_per_head: int = 5):
        self.attention_threshold = attention_threshold
        self.strong_threshold = strong_threshold
        self.max_rules_per_head = max_rules_per_head
        
        # Семантические шаблоны для разных типов признаков
        self.semantic_templates = {
            'color': {
                'patterns': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'gray'],
                'descriptions': ['цвет', 'окраска', 'оттенок', 'тон']
            },
            'texture': {
                'patterns': ['smooth', 'rough', 'soft', 'hard', 'shiny', 'matte', 'fuzzy', 'glossy'],
                'descriptions': ['текстура', 'поверхность', 'материал', 'структура']
            },
            'shape': {
                'patterns': ['round', 'square', 'triangular', 'oval', 'rectangular', 'circular', 'angular'],
                'descriptions': ['форма', 'очертания', 'контур', 'геометрия']
            },
            'object': {
                'patterns': ['car', 'dog', 'cat', 'house', 'tree', 'person', 'bird', 'fish'],
                'descriptions': ['объект', 'предмет', 'элемент', 'деталь']
            },
            'spatial': {
                'patterns': ['left', 'right', 'top', 'bottom', 'center', 'corner', 'edge', 'middle'],
                'descriptions': ['позиция', 'расположение', 'местоположение', 'позиционирование']
            }
        }
        
        # Лингвистические шаблоны для правил
        self.rule_templates = {
            'high_confidence': [
                "ЕСЛИ {text_condition} И {image_condition} ТО {conclusion} (уверенность: {confidence:.1%})",
                "При наличии {text_condition} и {image_condition} модель определяет {conclusion} с вероятностью {confidence:.1%}",
                "Правило: {text_condition} + {image_condition} → {conclusion} ({confidence:.1%})"
            ],
            'medium_confidence': [
                "ЕСЛИ {text_condition} И {image_condition} ТО вероятно {conclusion} (уверенность: {confidence:.1%})",
                "При {text_condition} и {image_condition} модель склоняется к {conclusion} ({confidence:.1%})",
                "Условное правило: {text_condition} + {image_condition} → {conclusion} ({confidence:.1%})"
            ],
            'low_confidence': [
                "ЕСЛИ {text_condition} И {image_condition} ТО возможно {conclusion} (уверенность: {confidence:.1%})",
                "При {text_condition} и {image_condition} есть признаки {conclusion} ({confidence:.1%})",
                "Слабое правило: {text_condition} + {image_condition} → {conclusion} ({confidence:.1%})"
            ]
        }
    
    def extract_semantic_rules(self, 
                             attention_weights: torch.Tensor,
                             text_tokens: Optional[List[str]] = None,
                             image_features: Optional[torch.Tensor] = None,
                             class_names: Optional[List[str]] = None,
                             head_idx: int = 0) -> List[SemanticFuzzyRule]:
        """
        Извлекает семантически осмысленные правила из attention weights
        """
        rules = []
        batch_size, seq_len, _ = attention_weights.shape
        
        # Обрабатываем каждый элемент батча
        for batch_idx in range(batch_size):
            attention = attention_weights[batch_idx]
            
            # Находим сильные связи
            high_attention_mask = attention > self.attention_threshold
            high_attention_indices = torch.nonzero(high_attention_mask, as_tuple=False)
            
            # Группируем по типам связей
            text_to_image = []
            image_to_text = []
            text_to_text = []
            image_to_image = []
            
            for idx in high_attention_indices:
                from_pos, to_pos = idx[0].item(), idx[1].item()
                strength = attention[from_pos, to_pos].item()
                
                if from_pos == to_pos:
                    continue
                
                # Определяем тип связи (предполагаем, что первая половина - текст, вторая - изображение)
                mid_point = seq_len // 2
                
                if from_pos < mid_point and to_pos >= mid_point:
                    text_to_image.append((from_pos, to_pos, strength))
                elif from_pos >= mid_point and to_pos < mid_point:
                    image_to_text.append((from_pos, to_pos, strength))
                elif from_pos < mid_point and to_pos < mid_point:
                    text_to_text.append((from_pos, to_pos, strength))
                else:
                    image_to_image.append((from_pos, to_pos, strength))
            
            # Извлекаем правила для каждого типа связей
            rules.extend(self._extract_text_to_image_rules(text_to_image, text_tokens, class_names, head_idx))
            rules.extend(self._extract_image_to_text_rules(image_to_text, text_tokens, class_names, head_idx))
            rules.extend(self._extract_text_to_text_rules(text_to_text, text_tokens, head_idx))
            rules.extend(self._extract_image_to_image_rules(image_to_image, head_idx))
        
        # Фильтруем и ранжируем правила
        rules = self._filter_and_rank_semantic_rules(rules)
        return rules[:self.max_rules_per_head]
    
    def _extract_text_to_image_rules(self, 
                                   connections: List[Tuple[int, int, float]],
                                   text_tokens: Optional[List[str]],
                                   class_names: Optional[List[str]],
                                   head_idx: int) -> List[SemanticFuzzyRule]:
        """Извлекает правила из текстовых признаков в изображение"""
        rules = []
        
        for from_pos, to_pos, strength in connections:
            # Определяем семантический тип на основе текста
            semantic_type = self._classify_text_semantic_type(text_tokens, from_pos)
            
            # Генерируем условие для текста
            text_condition = self._generate_text_condition(text_tokens, from_pos, semantic_type)
            
            # Генерируем условие для изображения
            image_condition = self._generate_image_condition(to_pos, semantic_type)
            
            # Определяем заключение
            conclusion = self._generate_conclusion(class_names, strength)
            
            # Вычисляем уверенность
            confidence = min(strength / self.strong_threshold, 1.0)
            
            # Генерируем лингвистическое описание
            linguistic_desc = self._generate_linguistic_rule(
                text_condition, image_condition, conclusion, confidence
            )
            
            rule = SemanticFuzzyRule(
                rule_id=f"rule_{head_idx}_{from_pos}_{to_pos}",
                condition_text=text_condition,
                condition_image=image_condition,
                conclusion=conclusion,
                confidence=confidence,
                strength=strength,
                semantic_type=semantic_type,
                linguistic_description=linguistic_desc,
                membership_values={'text': strength, 'image': strength * 0.8},
                attention_head=head_idx,
                tnorm_type='product'
            )
            
            rules.append(rule)
        
        return rules
    
    def _extract_image_to_text_rules(self, 
                                   connections: List[Tuple[int, int, float]],
                                   text_tokens: Optional[List[str]],
                                   class_names: Optional[List[str]],
                                   head_idx: int) -> List[SemanticFuzzyRule]:
        """Извлекает правила из изображения в текстовые признаки"""
        rules = []
        
        for from_pos, to_pos, strength in connections:
            # Определяем семантический тип
            semantic_type = self._classify_text_semantic_type(text_tokens, to_pos)
            
            # Генерируем условия
            image_condition = self._generate_image_condition(from_pos, semantic_type)
            text_condition = self._generate_text_condition(text_tokens, to_pos, semantic_type)
            conclusion = self._generate_conclusion(class_names, strength)
            
            confidence = min(strength / self.strong_threshold, 1.0)
            linguistic_desc = self._generate_linguistic_rule(
                image_condition, text_condition, conclusion, confidence
            )
            
            rule = SemanticFuzzyRule(
                rule_id=f"rule_{head_idx}_{from_pos}_{to_pos}",
                condition_text=text_condition,
                condition_image=image_condition,
                conclusion=conclusion,
                confidence=confidence,
                strength=strength,
                semantic_type=semantic_type,
                linguistic_description=linguistic_desc,
                membership_values={'image': strength, 'text': strength * 0.8},
                attention_head=head_idx,
                tnorm_type='product'
            )
            
            rules.append(rule)
        
        return rules
    
    def _extract_text_to_text_rules(self, 
                                  connections: List[Tuple[int, int, float]],
                                  text_tokens: Optional[List[str]],
                                  head_idx: int) -> List[SemanticFuzzyRule]:
        """Извлекает правила между текстовыми признаками"""
        rules = []
        
        for from_pos, to_pos, strength in connections:
            semantic_type = 'text_relation'
            
            text_condition = f"текст содержит '{text_tokens[from_pos] if text_tokens and from_pos < len(text_tokens) else f'признак_{from_pos}'}'"
            related_condition = f"связан с '{text_tokens[to_pos] if text_tokens and to_pos < len(text_tokens) else f'признак_{to_pos}'}'"
            conclusion = "усиливает семантическое понимание"
            
            confidence = min(strength / self.strong_threshold, 1.0)
            linguistic_desc = f"ЕСЛИ {text_condition} И {related_condition} ТО {conclusion} (уверенность: {confidence:.1%})"
            
            rule = SemanticFuzzyRule(
                rule_id=f"rule_{head_idx}_{from_pos}_{to_pos}",
                condition_text=text_condition,
                condition_image=related_condition,
                conclusion=conclusion,
                confidence=confidence,
                strength=strength,
                semantic_type=semantic_type,
                linguistic_description=linguistic_desc,
                membership_values={'text_from': strength, 'text_to': strength * 0.9},
                attention_head=head_idx,
                tnorm_type='product'
            )
            
            rules.append(rule)
        
        return rules
    
    def _extract_image_to_image_rules(self, 
                                    connections: List[Tuple[int, int, float]],
                                    head_idx: int) -> List[SemanticFuzzyRule]:
        """Извлекает правила между признаками изображения"""
        rules = []
        
        for from_pos, to_pos, strength in connections:
            semantic_type = 'spatial_relation'
            
            image_condition = f"область изображения {from_pos}"
            related_condition = f"связана с областью {to_pos}"
            conclusion = "формирует целостное представление"
            
            confidence = min(strength / self.strong_threshold, 1.0)
            linguistic_desc = f"ЕСЛИ {image_condition} И {related_condition} ТО {conclusion} (уверенность: {confidence:.1%})"
            
            rule = SemanticFuzzyRule(
                rule_id=f"rule_{head_idx}_{from_pos}_{to_pos}",
                condition_text=image_condition,
                condition_image=related_condition,
                conclusion=conclusion,
                confidence=confidence,
                strength=strength,
                semantic_type=semantic_type,
                linguistic_description=linguistic_desc,
                membership_values={'image_from': strength, 'image_to': strength * 0.9},
                attention_head=head_idx,
                tnorm_type='product'
            )
            
            rules.append(rule)
        
        return rules
    
    def _classify_text_semantic_type(self, text_tokens: Optional[List[str]], pos: int) -> str:
        """Классифицирует семантический тип текстового признака"""
        if not text_tokens or pos >= len(text_tokens):
            return 'general'
        
        token = text_tokens[pos].lower()
        
        for semantic_type, info in self.semantic_templates.items():
            for pattern in info['patterns']:
                if pattern in token:
                    return semantic_type
        
        return 'general'
    
    def _generate_text_condition(self, text_tokens: Optional[List[str]], pos: int, semantic_type: str) -> str:
        """Генерирует текстовое условие"""
        if not text_tokens or pos >= len(text_tokens):
            return f"текст содержит признак позиции {pos}"
        
        token = text_tokens[pos]
        descriptions = self.semantic_templates.get(semantic_type, {}).get('descriptions', ['признак'])
        desc = np.random.choice(descriptions)
        
        return f"текст содержит '{token}' ({desc})"
    
    def _generate_image_condition(self, pos: int, semantic_type: str) -> str:
        """Генерирует условие для изображения"""
        descriptions = self.semantic_templates.get(semantic_type, {}).get('descriptions', ['признак'])
        desc = np.random.choice(descriptions)
        
        return f"изображение имеет {desc} в области {pos}"
    
    def _generate_conclusion(self, class_names: Optional[List[str]], strength: float) -> str:
        """Генерирует заключение правила"""
        if class_names and strength > 0.5:
            # Выбираем случайный класс для демонстрации
            class_name = np.random.choice(class_names)
            return f"класс '{class_name}'"
        elif strength > 0.3:
            return "высокая уверенность в классификации"
        else:
            return "средняя уверенность в классификации"
    
    def _generate_linguistic_rule(self, condition1: str, condition2: str, conclusion: str, confidence: float) -> str:
        """Генерирует лингвистическое описание правила"""
        if confidence > 0.7:
            template = np.random.choice(self.rule_templates['high_confidence'])
        elif confidence > 0.4:
            template = np.random.choice(self.rule_templates['medium_confidence'])
        else:
            template = np.random.choice(self.rule_templates['low_confidence'])
        
        return template.format(
            text_condition=condition1,
            image_condition=condition2,
            conclusion=conclusion,
            confidence=confidence
        )
    
    def _filter_and_rank_semantic_rules(self, rules: List[SemanticFuzzyRule]) -> List[SemanticFuzzyRule]:
        """Фильтрует и ранжирует семантические правила"""
        # Сортируем по силе и уверенности
        rules.sort(key=lambda r: (r.strength * r.confidence), reverse=True)
        
        # Удаляем дубликаты по типу и условиям
        unique_rules = []
        seen_conditions = set()
        
        for rule in rules:
            condition_key = (rule.condition_text, rule.condition_image, rule.semantic_type)
            if condition_key not in seen_conditions:
                unique_rules.append(rule)
                seen_conditions.add(condition_key)
        
        return unique_rules
    
    def generate_rule_summary(self, rules: List[SemanticFuzzyRule]) -> Dict[str, Any]:
        """Генерирует сводку по правилам"""
        if not rules:
            return {"total_rules": 0, "summary": "Правила не найдены"}
        
        # Группируем по типам
        type_counts = {}
        confidence_stats = []
        strength_stats = []
        
        for rule in rules:
            rule_type = rule.semantic_type
            type_counts[rule_type] = type_counts.get(rule_type, 0) + 1
            confidence_stats.append(rule.confidence)
            strength_stats.append(rule.strength)
        
        return {
            "total_rules": len(rules),
            "rule_types": type_counts,
            "avg_confidence": np.mean(confidence_stats),
            "avg_strength": np.mean(strength_stats),
            "max_confidence": np.max(confidence_stats),
            "min_confidence": np.min(confidence_stats),
            "summary": f"Извлечено {len(rules)} правил с средней уверенностью {np.mean(confidence_stats):.1%}"
        }

def demo_improved_rule_extraction():
    """Демонстрация улучшенного извлечения правил"""
    print("🔍 Улучшенное извлечение правил - Демо")
    print("=" * 50)
    
    # Создаем пример attention weights
    seq_len = 10
    attention_weights = torch.rand(1, seq_len, seq_len)
    
    # Добавляем сильные связи
    attention_weights[0, 0, 5] = 0.25  # text to image
    attention_weights[0, 1, 6] = 0.18  # text to image
    attention_weights[0, 5, 1] = 0.20  # image to text
    attention_weights[0, 0, 1] = 0.15  # text to text
    attention_weights[0, 6, 7] = 0.12  # image to image
    
    # Нормализуем
    attention_weights = torch.softmax(attention_weights, dim=-1)
    
    # Пример текстовых токенов
    text_tokens = ["red", "car", "smooth", "surface", "round", "wheel", "shiny", "metal", "black", "tire"]
    class_names = ["автомобиль", "грузовик", "автобус", "мотоцикл"]
    
    # Создаем улучшенный извлекатель
    extractor = ImprovedRuleExtractor()
    
    # Извлекаем правила
    rules = extractor.extract_semantic_rules(
        attention_weights, 
        text_tokens, 
        class_names=class_names,
        head_idx=0
    )
    
    print(f"📊 Извлечено {len(rules)} семантических правил")
    print()
    
    # Показываем правила
    for i, rule in enumerate(rules):
        print(f"🔹 Правило {i+1}:")
        print(f"   ID: {rule.rule_id}")
        print(f"   Тип: {rule.semantic_type}")
        print(f"   Условие: {rule.condition_text}")
        print(f"   Изображение: {rule.condition_image}")
        print(f"   Заключение: {rule.conclusion}")
        print(f"   Уверенность: {rule.confidence:.1%}")
        print(f"   Сила: {rule.strength:.3f}")
        print(f"   Описание: {rule.linguistic_description}")
        print()
    
    # Генерируем сводку
    summary = extractor.generate_rule_summary(rules)
    print("📈 Сводка по правилам:")
    print(f"   Всего правил: {summary['total_rules']}")
    print(f"   Типы правил: {summary['rule_types']}")
    print(f"   Средняя уверенность: {summary['avg_confidence']:.1%}")
    print(f"   Средняя сила: {summary['avg_strength']:.3f}")
    print(f"   {summary['summary']}")
    
    return rules, summary

if __name__ == "__main__":
    demo_improved_rule_extraction()

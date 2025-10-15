#!/usr/bin/env python3
"""
Усложненная FAN архитектура для высоких показателей (>90%)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertModel
import torchvision.models as models

class AdvancedFuzzyAttention(nn.Module):
    """Усложненная fuzzy attention с множественными membership functions"""
    
    def __init__(self, hidden_dim, num_heads=8, num_membership=7, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_membership = num_membership
        
        # Множественные fuzzy membership functions
        self.fuzzy_centers = nn.Parameter(torch.randn(num_heads, num_membership, self.head_dim) * 0.1)
        self.fuzzy_widths = nn.Parameter(torch.ones(num_heads, num_membership, self.head_dim) * 0.3)
        self.fuzzy_weights = nn.Parameter(torch.ones(num_heads, num_membership) / num_membership)
        
        # Multi-scale attention
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization
        self.layer_norm_q = nn.LayerNorm(hidden_dim)
        self.layer_norm_k = nn.LayerNorm(hidden_dim)
        self.layer_norm_v = nn.LayerNorm(hidden_dim)
        
        # Output projection with residual connection
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Attention temperature
        self.temperature = nn.Parameter(torch.ones(1) * math.sqrt(self.head_dim))
        
    def gaussian_membership(self, x, center, width):
        """Гауссова функция принадлежности"""
        return torch.exp(-0.5 * ((x - center) / (width + 1e-8)) ** 2)
    
    def bell_membership(self, x, center, width):
        """Колоколообразная функция принадлежности"""
        return 1 / (1 + ((x - center) / (width + 1e-8)) ** 2)
    
    def sigmoid_membership(self, x, center, width):
        """Сигмоидальная функция принадлежности"""
        return torch.sigmoid((x - center) / (width + 1e-8))
    
    def forward(self, query, key, value, return_interpretation=False):
        batch_size = query.size(0)
        
        # Layer normalization
        query = self.layer_norm_q(query)
        key = self.layer_norm_k(key)
        value = self.layer_norm_v(value)
        
        # Linear projections
        Q = self.query(query).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.temperature
        
        # Apply multiple fuzzy membership functions
        fuzzy_scores = torch.zeros_like(scores)
        
        for h in range(self.num_heads):
            for f in range(self.num_membership):
                center = self.fuzzy_centers[h, f].unsqueeze(0).unsqueeze(0)
                width = self.fuzzy_widths[h, f].unsqueeze(0).unsqueeze(0)
                weight = self.fuzzy_weights[h, f]
                
                # Комбинируем разные типы membership functions
                gaussian = self.gaussian_membership(scores[:, h], center, width)
                bell = self.bell_membership(scores[:, h], center, width)
                sigmoid = self.sigmoid_membership(scores[:, h], center, width)
                
                # Взвешенная комбинация
                combined = 0.4 * gaussian + 0.4 * bell + 0.2 * sigmoid
                fuzzy_scores[:, h] += weight * combined.mean(dim=-1, keepdim=True)
        
        # Apply softmax
        attention_weights = torch.softmax(fuzzy_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # Output projection with residual connection
        output = self.output(attended) + query
        output = self.dropout(output)
        
        return output, attention_weights

class MultiScaleFusion(nn.Module):
    """Упрощенное многоуровневое слияние признаков"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        # Простое слияние с residual connections
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Attention для взвешивания признаков
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self, text_features, image_features):
        # Concatenate features
        combined = torch.cat([text_features, image_features], dim=-1)
        
        # Attention weights
        attention_weights = self.attention(combined)
        
        # Fusion
        fused = self.fusion(combined)
        
        # Apply attention
        fused = fused * attention_weights
        
        return fused

class AdvancedFANModel(nn.Module):
    """Усложненная FAN модель для высоких показателей"""
    
    def __init__(self, num_classes, num_heads=12, hidden_dim=1024, use_bert=True, use_resnet=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.use_bert = use_bert
        self.use_resnet = use_resnet
        
        # BERT для текста с fine-tuning
        if use_bert:
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            # Размораживаем последние слои для fine-tuning
            for param in self.bert_model.parameters():
                param.requires_grad = False
            # Размораживаем последние 4 слоя
            for layer in self.bert_model.encoder.layer[-4:]:
                for param in layer.parameters():
                    param.requires_grad = True
            
            self.text_projection = nn.Sequential(
                nn.Linear(768, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            self.text_projection = None
        
        # ResNet для изображений с fine-tuning
        if use_resnet:
            self.resnet = models.resnet50(pretrained=True)
            # Размораживаем последние слои
            for param in self.resnet.parameters():
                param.requires_grad = False
            # Размораживаем последние 2 блока
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True
            for param in self.resnet.layer3.parameters():
                param.requires_grad = True
            
            self.resnet.fc = nn.Sequential(
                nn.Linear(self.resnet.fc.in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            self.resnet = None
        
        # Advanced fuzzy attention
        self.text_fuzzy_attention = AdvancedFuzzyAttention(hidden_dim, num_heads, 7, 0.1)
        self.image_fuzzy_attention = AdvancedFuzzyAttention(hidden_dim, num_heads, 7, 0.1)
        
        # Cross-modal attention
        self.cross_attention = AdvancedFuzzyAttention(hidden_dim, num_heads, 5, 0.1)
        
        # Multi-scale fusion
        self.fusion = MultiScaleFusion(hidden_dim)
        
        # Advanced classifier (увеличиваем dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),  # Увеличиваем dropout
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),  # Увеличиваем dropout
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, text_tokens, attention_mask, image, return_explanations=False):
        batch_size = text_tokens.size(0)
        
        # Text encoding with fine-tuning
        if self.use_bert:
            bert_outputs = self.bert_model(text_tokens, attention_mask=attention_mask)
            text_features = bert_outputs.last_hidden_state.mean(dim=1)
            text_features = self.text_projection(text_features)
        else:
            text_features = torch.zeros(batch_size, self.hidden_dim, device=text_tokens.device)
        
        # Image encoding with fine-tuning
        if self.use_resnet:
            image_features = self.resnet(image)
        else:
            image_features = torch.zeros(batch_size, self.hidden_dim, device=image.device)
        
        # Fuzzy attention на тексте
        text_attended, text_attention_weights = self.text_fuzzy_attention(
            text_features.unsqueeze(1), text_features.unsqueeze(1), text_features.unsqueeze(1)
        )
        text_attended = text_attended.squeeze(1)
        
        # Fuzzy attention на изображениях
        image_attended, image_attention_weights = self.image_fuzzy_attention(
            image_features.unsqueeze(1), image_features.unsqueeze(1), image_features.unsqueeze(1)
        )
        image_attended = image_attended.squeeze(1)
        
        # Cross-modal attention
        cross_attended, cross_attention_weights = self.cross_attention(
            text_attended.unsqueeze(1), image_attended.unsqueeze(1), image_attended.unsqueeze(1)
        )
        cross_attended = cross_attended.squeeze(1)
        
        # Multi-scale fusion
        fused = self.fusion(text_attended, image_attended)
        
        # Add cross-modal information
        final_features = fused + cross_attended
        
        # Classification
        logits = self.classifier(final_features)
        probs = torch.softmax(logits, dim=1)
        
        result = {
            'logits': logits,
            'probs': probs,
            'predictions': torch.argmax(logits, dim=1),
            'confidence': torch.max(probs, dim=1)[0]
        }
        
        if return_explanations:
            result['explanations'] = {
                'text_attention': text_attention_weights,
                'image_attention': image_attention_weights,
                'cross_attention': cross_attention_weights,
                'text_features': text_features,
                'image_features': image_features,
                'fused_features': final_features
            }
            
        return result

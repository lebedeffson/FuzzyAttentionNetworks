#!/usr/bin/env python3
"""
Универсальная FAN модель для разных датасетов
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertModel
import torchvision.models as models

class SimpleFuzzyAttention(nn.Module):
    """Простая fuzzy attention для универсального использования"""
    
    def __init__(self, hidden_dim, num_heads=4, num_membership=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_membership = num_membership
        
        # Fuzzy membership functions
        self.fuzzy_centers = nn.Parameter(torch.randn(num_heads, num_membership, self.head_dim) * 0.1)
        self.fuzzy_widths = nn.Parameter(torch.ones(num_heads, num_membership, self.head_dim) * 0.5)
        
        # Attention layers
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def bell_membership(self, x, center, width):
        """Колоколообразная функция принадлежности"""
        return 1 / (1 + ((x - center) / (width + 1e-8)) ** 2)
        
    def forward(self, query, key, value, return_interpretation=False):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply fuzzy membership functions
        fuzzy_scores = torch.zeros_like(scores)
        
        for h in range(self.num_heads):
            for f in range(self.num_membership):
                center = self.fuzzy_centers[h, f].unsqueeze(0).unsqueeze(0)
                width = self.fuzzy_widths[h, f].unsqueeze(0).unsqueeze(0)
                
                membership = self.bell_membership(scores[:, h], center, width)
                fuzzy_scores[:, h] += membership.mean(dim=-1, keepdim=True)
        
        # Normalize
        fuzzy_scores = fuzzy_scores / self.num_membership
        
        # Apply softmax
        attention_weights = torch.softmax(fuzzy_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # Output projection
        output = self.output(attended)
        
        return output, attention_weights

class UniversalFANModel(nn.Module):
    """Универсальная FAN модель для разных датасетов"""
    
    def __init__(self, num_classes, num_heads=4, hidden_dim=512, use_bert=True, use_resnet=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.use_bert = use_bert
        self.use_resnet = use_resnet
        
        # BERT для текста
        if use_bert:
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            # Замораживаем BERT
            for param in self.bert_model.parameters():
                param.requires_grad = False
            
            # Проекционный слой для BERT features
            self.text_projection = nn.Linear(768, hidden_dim)
        else:
            self.text_projection = None
        
        # ResNet для изображений
        if use_resnet:
            self.resnet = models.resnet18(pretrained=True)
            # Замораживаем ResNet
            for param in self.resnet.parameters():
                param.requires_grad = False
            
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, hidden_dim)
        else:
            self.resnet = None
        
        # Fuzzy attention
        self.text_fuzzy_attention = SimpleFuzzyAttention(hidden_dim, num_heads, 5)
        self.image_fuzzy_attention = SimpleFuzzyAttention(hidden_dim, num_heads, 5)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, text_tokens, attention_mask, image, return_explanations=False):
        batch_size = text_tokens.size(0)
        
        # Text encoding
        if self.use_bert:
            with torch.no_grad():
                bert_outputs = self.bert_model(text_tokens, attention_mask=attention_mask)
                text_features = bert_outputs.last_hidden_state.mean(dim=1)
            text_features = self.text_projection(text_features)
        else:
            # Fallback для случаев без BERT
            text_features = torch.zeros(batch_size, self.hidden_dim, device=text_tokens.device)
        
        # Image encoding
        if self.use_resnet:
            with torch.no_grad():
                image_features = self.resnet(image)
        else:
            # Fallback для случаев без ResNet
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
        
        # Fusion
        combined = torch.cat([text_attended, image_attended], dim=1)
        fused = self.fusion(combined)
        
        # Classification
        logits = self.classifier(fused)
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
                'text_features': text_features,
                'image_features': image_features
            }
            
        return result

class ModelManager:
    """Менеджер для работы с разными моделями"""
    
    def __init__(self):
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self, dataset_name, model_path):
        """Загрузить модель для конкретного датасета"""
        from .dataset_manager import DatasetManager
        
        dataset_manager = DatasetManager()
        dataset_info = dataset_manager.get_dataset_info(dataset_name)
        
        # Создаем модель
        if dataset_name == 'hateful_memes':
            model = UniversalFANModel(
                num_classes=2,
                num_heads=8,
                hidden_dim=768,
                use_bert=True,
                use_resnet=True
            )
        elif dataset_name == 'cifar10':
            model = UniversalFANModel(
                num_classes=10,
                num_heads=4,
                hidden_dim=512,
                use_bert=True,
                use_resnet=True
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Загружаем веса
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        self.models[dataset_name] = model
        return model
    
    def get_model(self, dataset_name):
        """Получить модель для датасета"""
        if dataset_name not in self.models:
            from .dataset_manager import DatasetManager
            dataset_manager = DatasetManager()
            dataset_info = dataset_manager.get_dataset_info(dataset_name)
            self.load_model(dataset_name, dataset_info['model_path'])
        
        return self.models[dataset_name]
    
    def predict(self, dataset_name, text_tokens, attention_mask, image, return_explanations=False):
        """Предсказание для конкретного датасета"""
        model = self.get_model(dataset_name)
        
        with torch.no_grad():
            result = model(text_tokens, attention_mask, image, return_explanations)
        
        return result


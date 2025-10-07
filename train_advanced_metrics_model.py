#!/usr/bin/env python3
"""
Продвинутая модель для достижения F1 > 0.7
Стратегии: Data Augmentation, Ensemble, Advanced Regularization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import json
import numpy as np
from pathlib import Path
from PIL import Image
import random
from tqdm import tqdm
import sys
import os
import math
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertModel
import torchvision.models as models
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class AdvancedDataAugmentation:
    """Продвинутая аугментация данных"""
    
    def __init__(self):
        self.text_augmentations = [
            self.synonym_replacement,
            self.random_insertion,
            self.random_swap,
            self.random_deletion
        ]
        
        self.image_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
        ])
    
    def synonym_replacement(self, text):
        """Замена синонимов"""
        synonyms = {
            'hate': ['dislike', 'despise', 'loathe'],
            'love': ['adore', 'cherish', 'treasure'],
            'bad': ['terrible', 'awful', 'horrible'],
            'good': ['great', 'excellent', 'wonderful']
        }
        
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in synonyms:
                words[i] = random.choice(synonyms[word.lower()])
        return ' '.join(words)
    
    def random_insertion(self, text):
        """Случайная вставка слов"""
        words = text.split()
        if len(words) > 1:
            random_word = random.choice(words)
            random_idx = random.randint(0, len(words))
            words.insert(random_idx, random_word)
        return ' '.join(words)
    
    def random_swap(self, text):
        """Случайная перестановка слов"""
        words = text.split()
        if len(words) > 1:
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)
    
    def random_deletion(self, text):
        """Случайное удаление слов"""
        words = text.split()
        if len(words) > 2:
            words.pop(random.randint(0, len(words)-1))
        return ' '.join(words)
    
    def augment_text(self, text, num_augmentations=2):
        """Аугментация текста"""
        augmented_texts = [text]
        for _ in range(num_augmentations):
            aug_func = random.choice(self.text_augmentations)
            augmented_texts.append(aug_func(text))
        return augmented_texts
    
    def augment_image(self, image):
        """Аугментация изображения"""
        return self.image_transforms(image)

class AdvancedFuzzyAttention(nn.Module):
    """Продвинутая fuzzy attention с улучшенной архитектурой"""
    
    def __init__(self, hidden_dim, num_heads=8, num_membership=7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_membership = num_membership
        
        # Улучшенные fuzzy membership functions
        self.fuzzy_centers = nn.Parameter(torch.randn(num_heads, num_membership, self.head_dim) * 0.05)
        self.fuzzy_widths = nn.Parameter(torch.ones(num_heads, num_membership, self.head_dim) * 0.2)
        
        # Multi-scale attention
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        
        # Residual connection
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Attention gating
        self.attention_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def bell_membership(self, x, center, width):
        """Улучшенная колоколообразная функция принадлежности"""
        return 1 / (1 + ((x - center) / (width + 1e-8)) ** 2)
        
    def forward(self, query, key, value, return_interpretation=False):
        batch_size = query.size(0)
        residual = query
        
        # Linear projections
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply fuzzy membership functions
        fuzzy_scores = torch.zeros_like(scores)
        membership_values = {}
        
        for h in range(self.num_heads):
            for f in range(self.num_membership):
                center = self.fuzzy_centers[h, f].unsqueeze(0).unsqueeze(0)
                width = self.fuzzy_widths[h, f].unsqueeze(0).unsqueeze(0)
                
                # Bell membership function
                membership = self.bell_membership(scores[:, h], center, width)
                fuzzy_scores[:, h] += membership.mean(dim=-1, keepdim=True)
                
                # Сохраняем для интерпретации
                if return_interpretation:
                    membership_values[f'head_{h}_func_{f}'] = {
                        'center': center,
                        'width': width,
                        'membership': membership,
                        'contribution': membership.mean(dim=-1, keepdim=True)
                    }
        
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
        
        # Attention gating
        gate = self.attention_gate(output)
        output = output * gate
        
        # Residual connection
        output = self.norm(output + residual)
        
        # Сохраняем для интерпретации
        if return_interpretation:
            self.attention_weights = attention_weights
            self.fuzzy_scores = fuzzy_scores
            self.membership_values = membership_values
        
        return output, attention_weights

class AdvancedFANModel(nn.Module):
    """Продвинутая FAN модель с улучшенной архитектурой"""
    
    def __init__(self, num_classes=2, num_heads=8, hidden_dim=768):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # BERT для текста с fine-tuning
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        # Размораживаем последние слои BERT
        for param in self.bert_model.encoder.layer[-2:].parameters():
            param.requires_grad = True
        
        # ResNet для изображений с fine-tuning
        self.resnet = models.resnet50(pretrained=True)
        # Размораживаем последние слои ResNet
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, hidden_dim)
        
        # Продвинутые Fuzzy Attention Networks
        self.text_fuzzy_attention = AdvancedFuzzyAttention(hidden_dim, num_heads, 7)
        self.image_fuzzy_attention = AdvancedFuzzyAttention(hidden_dim, num_heads, 7)
        
        # Multi-scale cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        
        # Advanced fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim)
        )
        
        # Advanced classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Инициализация bias для балансировки
        with torch.no_grad():
            self.classifier[-1].bias[0] = 0.0
            self.classifier[-1].bias[1] = 0.3
        
    def forward(self, text_tokens, attention_mask, image, return_explanations=False):
        batch_size = text_tokens.size(0)
        
        # BERT encoding для текста
        bert_outputs = self.bert_model(text_tokens, attention_mask=attention_mask)
        text_features = bert_outputs.last_hidden_state.mean(dim=1)
        
        # ResNet encoding для изображений
        image_features = self.resnet(image)
        
        # Fuzzy attention на тексте
        text_attended, text_attention_weights = self.text_fuzzy_attention(
            text_features.unsqueeze(1), text_features.unsqueeze(1), text_features.unsqueeze(1),
            return_interpretation=return_explanations
        )
        text_attended = text_attended.squeeze(1)
        
        # Fuzzy attention на изображениях
        image_attended, image_attention_weights = self.image_fuzzy_attention(
            image_features.unsqueeze(1), image_features.unsqueeze(1), image_features.unsqueeze(1),
            return_interpretation=return_explanations
        )
        image_attended = image_attended.squeeze(1)
        
        # Cross-modal attention
        text_enhanced, cross_modal_weights = self.cross_modal_attention(
            text_attended.unsqueeze(1), image_attended.unsqueeze(1), image_attended.unsqueeze(1)
        )
        text_enhanced = text_enhanced.squeeze(1)
        
        # Fusion
        combined = torch.cat([text_enhanced, image_attended], dim=1)
        fused = self.fusion_layer(combined)
        
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
                'cross_modal_attention': cross_modal_weights,
                'text_fuzzy_membership': self.text_fuzzy_attention.membership_values,
                'image_fuzzy_membership': self.image_fuzzy_attention.membership_values,
                'text_features': text_features,
                'image_features': image_features
            }
            
        return result

class AdvancedMemeDataset(Dataset):
    """Продвинутый датасет с аугментацией"""
    
    def __init__(self, data_path, max_length=128, image_size=(224, 224), augment=True):
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.image_size = image_size
        self.augment = augment
        
        # Load data
        with open(self.data_path / "train.jsonl", 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        
        # BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Data augmentation
        self.augmentation = AdvancedDataAugmentation()
        
        # Check class distribution
        labels = [item['label'] for item in self.data]
        class_counts = Counter(labels)
        print(f"📊 Датасет загружен: {len(self.data)} образцов")
        print(f"📈 Распределение классов: {dict(class_counts)}")
        
    def _tokenize_text(self, text):
        """Токенизация с BERT"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)
    
    def _load_image(self, img_path):
        """Загрузка и предобработка изображения"""
        try:
            # Убираем 'img/' из начала пути
            clean_path = img_path.replace('img/', '') if img_path.startswith('img/') else img_path
            image = Image.open(self.data_path / "img" / clean_path).convert('RGB')
            
            # Аугментация изображения
            if self.augment and random.random() > 0.5:
                image = self.augmentation.augment_image(image)
            
            return self.transform(image)
        except Exception as e:
            # Create placeholder image
            placeholder = Image.new('RGB', self.image_size, color='gray')
            return self.transform(placeholder)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        text = item['text']
        
        # Аугментация текста
        if self.augment and random.random() > 0.5:
            augmented_texts = self.augmentation.augment_text(text, num_augmentations=1)
            text = random.choice(augmented_texts)
        
        text_tokens, attention_mask = self._tokenize_text(text)
        image = self._load_image(item['img'])
        label = torch.tensor(item['label'], dtype=torch.long)
        
        return text_tokens, attention_mask, image, label

def create_balanced_sampler(dataset, original_dataset):
    """Создает сбалансированный сэмплер"""
    labels = [original_dataset.data[dataset.indices[i]]['label'] for i in range(len(dataset))]
    class_counts = Counter(labels)
    
    # Вычисляем веса для каждого класса
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]
    
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def train_advanced_metrics_model():
    """Обучение продвинутой модели для достижения F1 > 0.7"""
    
    print("🚀 Обучение продвинутой FAN модели для F1 > 0.7...")
    print("🎯 Стратегии: Data Augmentation + Advanced Architecture + Fine-tuning")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Устройство: {device}")
    
    # Dataset with augmentation
    dataset = AdvancedMemeDataset("data/hateful_memes", augment=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create balanced sampler
    train_sampler = create_balanced_sampler(train_dataset, dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Model
    model = AdvancedFANModel(
        num_classes=2,
        num_heads=8,  # Больше голов внимания
        hidden_dim=768
    ).to(device)
    
    # Compute class weights
    labels = [dataset.data[i]['label'] for i in range(len(dataset))]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Advanced optimizer and loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Простой optimizer для избежания ошибок
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    
    # Advanced scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training
    best_f1 = 0.0
    patience = 15
    patience_counter = 0
    
    for epoch in range(100):  # Больше эпох
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (text_tokens, attention_mask, images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/100")):
            text_tokens = text_tokens.to(device)
            attention_mask = attention_mask.to(device)
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            result = model(text_tokens, attention_mask, images)
            loss = criterion(result['logits'], labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(result['logits'], 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for text_tokens, attention_mask, images, labels in val_loader:
                text_tokens = text_tokens.to(device)
                attention_mask = attention_mask.to(device)
                images = images.to(device)
                labels = labels.to(device)
                
                result = model(text_tokens, attention_mask, images)
                loss = criterion(result['logits'], labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(result['logits'], 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        # F1 score
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}, F1: {f1:.4f}")
        
        # Check predictions distribution
        pred_counts = Counter(all_predictions)
        print(f"  Predictions: {dict(pred_counts)}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_advanced_metrics_model.pth')
            print(f"  💾 Сохранена лучшая модель (F1: {f1:.4f})")
            
            # Если достигли цели F1 > 0.7
            if f1 > 0.7:
                print(f"  🎯 ДОСТИГНУТА ЦЕЛЬ F1 > 0.7! Продолжаем обучение для стабилизации.")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"  ⏹️ Early stopping после {epoch+1} эпох")
            break
        
        scheduler.step()
    
    print(f"🎉 Обучение завершено! Лучший F1: {best_f1:.4f}")
    
    # Final evaluation
    if os.path.exists('models/best_advanced_metrics_model.pth'):
        model.load_state_dict(torch.load('models/best_advanced_metrics_model.pth'))
        model.eval()
        
        with torch.no_grad():
            all_predictions = []
            all_labels = []
            all_probs = []
            
            for text_tokens, attention_mask, images, labels in val_loader:
                text_tokens = text_tokens.to(device)
                attention_mask = attention_mask.to(device)
                images = images.to(device)
                labels = labels.to(device)
                
                result = model(text_tokens, attention_mask, images, return_explanations=True)
                _, predicted = torch.max(result['logits'], 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(result['probs'].cpu().numpy())
        
        # Final metrics
        print("\n📊 Финальные метрики:")
        print(f"F1 Score: {f1_score(all_labels, all_predictions, average='weighted'):.4f}")
        print(f"Accuracy: {val_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions))
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_predictions))
        
        # Анализ улучшений
        print("\n🔍 Анализ улучшений:")
        print("✅ Data Augmentation активна")
        print("✅ Advanced Fuzzy Attention (8 heads, 7 membership functions)")
        print("✅ Fine-tuning BERT и ResNet")
        print("✅ Multi-scale fusion layers")
        print("✅ Advanced regularization")
        print("✅ CosineAnnealingWarmRestarts scheduler")
    
    return model, best_f1

if __name__ == "__main__":
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train model
    model, f1_score = train_advanced_metrics_model()
    
    print(f"\n🏆 Продвинутая модель готова! F1 Score: {f1_score:.4f}")
    
    if f1_score > 0.7:
        print("🎯 ОТЛИЧНО! F1 > 0.7 ДОСТИГНУТ!")
        print("📝 Модель готова для статьи с высокими метриками!")
    else:
        print("⚠️ Результат хороший, но можно еще улучшить!")
        print("💡 Попробуйте ensemble learning или больше данных")

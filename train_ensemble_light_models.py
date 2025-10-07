#!/usr/bin/env python3
"""
Ensemble Light Models Training Script
Ensemble из легких моделей для улучшения качества
"""

import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt

# Настройки
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Training Ensemble Light Models")
print("=" * 50)
print(f"Using device: {DEVICE}")

# Создаем директории
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

class LightMemeDataset(Dataset):
    """Легкий датасет с простой обработкой"""
    
    def __init__(self, data, max_length=64, is_training=True):
        self.data = data
        self.max_length = max_length
        self.is_training = is_training
        
        # Простая обработка изображений
        self.image_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Обработка текста
        text = item.get('text', '')
        if not text:
            text = item.get('question', '')
        
        # Простая токенизация
        words = text.split()[:self.max_length]
        text_tokens = [hash(word) % 10000 for word in words]
        text_tokens = text_tokens + [0] * (self.max_length - len(text_tokens))
        text_tokens = text_tokens[:self.max_length]
        
        # Обработка изображения
        image_path = item.get('img', '')
        if not image_path:
            image_path = item.get('image_path', '')
        
        try:
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
            else:
                image = Image.new('RGB', (64, 64), color=(128, 128, 128))
        except:
            image = Image.new('RGB', (64, 64), color=(128, 128, 128))
        
        image = self.image_transform(image)
        
        # Лейбл
        label = int(item.get('label', 0))
        
        return {
            'text_tokens': torch.tensor(text_tokens, dtype=torch.long),
            'image': image,
            'label': label,
            'text': text
        }

class LightMemeModel(nn.Module):
    """Легкая модель для мемов"""
    
    def __init__(self, vocab_size=10000, text_dim=128, image_dim=12288, 
                 hidden_dim=256, num_classes=2, model_id=0):
        super().__init__()
        
        self.model_id = model_id
        
        # Text encoder (простой)
        self.text_embedding = nn.Embedding(vocab_size, text_dim)
        self.text_encoder = nn.LSTM(text_dim, hidden_dim, batch_first=True)
        
        # Image encoder (простой CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Fuzzy attention (упрощенный)
        self.fuzzy_attention = SimpleFuzzyAttention(hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, text_tokens, image):
        batch_size = text_tokens.size(0)
        
        # Text encoding
        text_emb = self.text_embedding(text_tokens)
        text_output, _ = self.text_encoder(text_emb)
        text_features = text_output.mean(dim=1)
        
        # Image encoding
        image_features = self.image_encoder(image)
        image_features = image_features.view(batch_size, -1)
        
        # Проекция изображения в hidden_dim
        image_proj = nn.Linear(128, text_features.size(1)).to(image.device)
        image_features = image_proj(image_features)
        
        # Fuzzy attention
        text_attended = self.fuzzy_attention(text_features)
        
        # Fusion
        combined = torch.cat([text_attended, image_features], dim=-1)
        
        # Classification
        logits = self.classifier(combined)
        probs = torch.softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'probs': probs,
            'predictions': torch.argmax(logits, dim=-1),
            'confidence': torch.max(probs, dim=-1)[0]
        }

class SimpleFuzzyAttention(nn.Module):
    """Упрощенный fuzzy attention"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Fuzzy membership functions
        self.fuzzy_centers = nn.Parameter(torch.randn(3))
        self.fuzzy_widths = nn.Parameter(torch.ones(3))
        
        # Attention weights
        self.attention_weights = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # Простой fuzzy attention
        attention_scores = self.attention_weights(x)
        
        # Fuzzy membership
        fuzzy_scores = torch.zeros_like(attention_scores)
        for i in range(3):
            center = self.fuzzy_centers[i]
            width = self.fuzzy_widths[i]
            membership = torch.exp(-((attention_scores - center) ** 2) / (2 * width ** 2))
            fuzzy_scores += membership
        
        # Normalize
        fuzzy_scores = fuzzy_scores / 3.0
        
        # Apply attention
        attended = x * fuzzy_scores
        
        return attended

class EnsembleLightModels(nn.Module):
    """Ensemble из легких моделей"""
    
    def __init__(self, num_models=3, vocab_size=10000, text_dim=128, 
                 image_dim=12288, hidden_dim=256, num_classes=2):
        super().__init__()
        
        self.num_models = num_models
        
        # Создаем несколько моделей
        self.models = nn.ModuleList([
            LightMemeModel(vocab_size, text_dim, image_dim, hidden_dim, num_classes, i)
            for i in range(num_models)
        ])
        
        # Ensemble classifier
        self.ensemble_classifier = nn.Sequential(
            nn.Linear(num_models * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, text_tokens, image):
        # Получаем предсказания от всех моделей
        all_logits = []
        all_probs = []
        
        for model in self.models:
            outputs = model(text_tokens, image)
            all_logits.append(outputs['logits'])
            all_probs.append(outputs['probs'])
        
        # Объединяем предсказания
        combined_logits = torch.cat(all_logits, dim=-1)
        ensemble_logits = self.ensemble_classifier(combined_logits)
        ensemble_probs = torch.softmax(ensemble_logits, dim=-1)
        
        return {
            'logits': ensemble_logits,
            'probs': ensemble_probs,
            'predictions': torch.argmax(ensemble_logits, dim=-1),
            'confidence': torch.max(ensemble_probs, dim=-1)[0],
            'individual_logits': all_logits,
            'individual_probs': all_probs
        }

def load_dataset():
    """Загружаем датасет"""
    print("📊 Загружаем датасет...")
    
    with open('data/hateful_memes/train.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    
    print(f"📊 Загружено {len(data)} образцов")
    
    # Разделяем на train/val
    random.shuffle(data)
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"📊 Train: {len(train_data)}, Val: {len(val_data)}")
    
    return train_data, val_data

def collate_fn(batch):
    """Collate function for DataLoader"""
    text_tokens = torch.stack([item['text_tokens'] for item in batch])
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    texts = [item['text'] for item in batch]
    
    return {
        'text_tokens': text_tokens,
        'image': images,
        'label': labels,
        'text': texts
    }

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Одна эпоха обучения"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        text_tokens = batch['text_tokens'].to(device)
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(text_tokens, images)
        loss = criterion(outputs['logits'], labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = outputs['predictions'].detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels_np)
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, f1

def evaluate(model, dataloader, criterion, device):
    """Оценка модели"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            text_tokens = batch['text_tokens'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(text_tokens, images)
            loss = criterion(outputs['logits'], labels)
            
            total_loss += loss.item()
            
            preds = outputs['predictions'].detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            probs = outputs['probs'].detach().cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            all_probs.extend(probs)
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, f1, precision, recall, accuracy

def main():
    """Основная функция"""
    # Загружаем данные
    train_data, val_data = load_dataset()
    
    # Создаем датасеты
    train_dataset = LightMemeDataset(train_data, is_training=True)
    val_dataset = LightMemeDataset(val_data, is_training=False)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # Ensemble модель
    model = EnsembleLightModels(num_models=3).to(DEVICE)
    print(f"Ensemble model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Оптимизатор и loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Обучение
    best_f1 = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(15):  # Меньше эпох для ensemble
        print(f"\nEpoch {epoch+1}/15")
        print("-" * 30)
        
        # Train
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        # Val
        val_loss, val_f1, val_precision, val_recall, val_accuracy = evaluate(
            model, val_loader, criterion, DEVICE
        )
        
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
        
        # Сохраняем лучшую модель
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_ensemble_light_model.pth')
            print(f"✅ New best model saved! (Val F1: {val_f1:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Финальная оценка
    print("\n📊 Final Evaluation...")
    model.load_state_dict(torch.load('models/best_ensemble_light_model.pth'))
    val_loss, val_f1, val_precision, val_recall, val_accuracy = evaluate(
        model, val_loader, criterion, DEVICE
    )
    
    print(f"\n📈 Final Results:")
    print(f"   Accuracy: {val_accuracy:.4f}")
    print(f"   F1 Score: {val_f1:.4f}")
    print(f"   Precision: {val_precision:.4f}")
    print(f"   Recall: {val_recall:.4f}")
    
    # Сохраняем результаты
    results = {
        'accuracy': val_accuracy,
        'f1_score': val_f1,
        'precision': val_precision,
        'recall': val_recall,
        'epochs': epoch + 1,
        'best_f1': best_f1,
        'model_type': 'ensemble_light',
        'num_models': 3
    }
    
    with open('results/ensemble_light_model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    if val_f1 >= 0.70:
        print("🎉 Target reached! F1 >= 0.70")
    else:
        print(f"⚠️ Target not reached. F1 = {val_f1:.4f} < 0.70")
    
    print(f"\n🎉 Ensemble light model training completed!")
    print(f"Best model saved to: models/best_ensemble_light_model.pth")
    print(f"Results saved to: results/ensemble_light_model_results.json")

if __name__ == "__main__":
    main()

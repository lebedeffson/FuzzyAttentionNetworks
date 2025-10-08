#!/usr/bin/env python3
"""
Обучение усложненной FAN модели на Stanford Dogs для достижения >90%
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import sys
import os

# Добавляем src в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset_manager import DatasetManager
from advanced_fan_model import AdvancedFANModel

class AdvancedStanfordDogsTrainer:
    """Тренер для усложненной Stanford Dogs FAN модели"""
    
    def __init__(self, data_dir, model_dir, device='cuda'):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"🖥️ Используем устройство: {self.device}")
        
        # Создаем датасеты
        self.dataset_manager = DatasetManager()
        self.train_dataset = self.dataset_manager.create_dataset('stanford_dogs', 'train')
        self.val_dataset = self.dataset_manager.create_dataset('stanford_dogs', 'val')
        
        print(f"📊 Train samples: {len(self.train_dataset)}")
        print(f"📊 Val samples: {len(self.val_dataset)}")
        print(f"📊 Classes: {self.train_dataset.num_classes}")
        
        # Создаем DataLoader'ы с аугментацией
        train_transform = self._get_train_transforms()
        val_transform = self._get_val_transforms()
        
        # Обновляем трансформации в датасетах
        self.train_dataset.transform = train_transform
        self.val_dataset.transform = val_transform
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=8,  # Меньший batch size для сложной модели
            shuffle=True, 
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=8, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Создаем усложненную модель
        self.model = AdvancedFANModel(
            num_classes=self.train_dataset.num_classes,
            num_heads=8,  # 8 голов для 1024 размерности
            hidden_dim=1024,  # Больше размерности
            use_bert=True,
            use_resnet=True
        ).to(self.device)
        
        # Подсчитываем параметры
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 Всего параметров: {total_params:,}")
        print(f"📊 Обучаемых параметров: {trainable_params:,}")
        
        # Оптимизатор с разными learning rates для разных частей
        self.optimizer = optim.AdamW([
            {'params': [p for n, p in self.model.named_parameters() if 'bert' in n and p.requires_grad], 'lr': 1e-5},
            {'params': [p for n, p in self.model.named_parameters() if 'resnet' in n and p.requires_grad], 'lr': 1e-5},
            {'params': [p for n, p in self.model.named_parameters() if 'bert' not in n and 'resnet' not in n and p.requires_grad], 'lr': 1e-4}
        ], weight_decay=1e-4)
        
        # Scheduler с warmup
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=[1e-4, 1e-4, 1e-3],
            epochs=100,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1
        )
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Метрики
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        
        self.best_f1 = 0.0
        self.best_accuracy = 0.0
        
    def _get_train_transforms(self):
        """Трансформации для обучения с аугментацией"""
        import torchvision.transforms as transforms
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_val_transforms(self):
        """Трансформации для валидации"""
        import torchvision.transforms as transforms
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def train_epoch(self, epoch):
        """Обучение одной эпохи"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            # Получаем данные
            text_tokens = batch['text_tokens'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Обнуляем градиенты
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(text_tokens, attention_mask, images)
            logits = outputs['logits']
            
            # Вычисляем потери
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Обновляем веса
            self.optimizer.step()
            self.scheduler.step()
            
            # Статистика
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Обновляем прогресс-бар
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{current_lr:.2e}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        self.train_losses.append(avg_loss)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, epoch):
        """Валидация одной эпохи"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f'Validation {epoch+1}'):
                # Получаем данные
                text_tokens = batch['text_tokens'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(text_tokens, attention_mask, images)
                logits = outputs['logits']
                
                # Вычисляем потери
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Предсказания
                _, predicted = torch.max(logits.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Вычисляем метрики
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        self.val_f1_scores.append(f1)
        
        return avg_loss, accuracy, f1, precision, recall
    
    def save_model(self, epoch, accuracy, f1_score, is_best=False):
        """Сохранить модель"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': accuracy,
            'f1_score': f1_score,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_f1_scores': self.val_f1_scores
        }
        
        # Сохраняем checkpoint
        checkpoint_path = self.model_dir / f'advanced_checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Сохраняем лучшую модель
        if is_best:
            best_path = self.model_dir / 'best_advanced_stanford_dogs_fan_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Сохранена лучшая модель: {best_path}")
    
    def plot_metrics(self):
        """Построить графики метрик"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(self.val_accuracies, label='Val Accuracy', color='green')
        ax2.axhline(y=0.9, color='red', linestyle='--', label='Target 90%')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # F1 Score
        ax3.plot(self.val_f1_scores, label='Val F1 Score', color='orange')
        ax3.axhline(y=0.9, color='red', linestyle='--', label='Target 90%')
        ax3.set_title('Validation F1 Score')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.legend()
        ax3.grid(True)
        
        # Learning Rate
        lrs = [self.scheduler.get_last_lr()[0] for _ in range(len(self.train_losses))]
        ax4.plot(lrs, label='Learning Rate', color='purple')
        ax4.set_title('Learning Rate Schedule')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'advanced_training_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def train(self, num_epochs=100, patience=15):
        """Полное обучение модели"""
        print(f"🚀 Начинаем обучение усложненной модели на {num_epochs} эпох")
        print("🎯 Цель: F1 > 0.90, Accuracy > 90%")
        print("=" * 60)
        
        start_time = time.time()
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\n📈 Эпоха {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # Обучение
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Валидация
            val_loss, val_acc, val_f1, val_prec, val_rec = self.validate_epoch(epoch)
            
            # Выводим метрики
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Val F1: {val_f1:.4f}, Val Prec: {val_prec:.4f}, Val Rec: {val_rec:.4f}")
            
            # Проверяем на лучшую модель
            is_best = False
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.best_accuracy = val_acc
                best_epoch = epoch
                is_best = True
                patience_counter = 0
                print(f"🎉 Новый лучший F1: {val_f1:.4f}")
            else:
                patience_counter += 1
            
            # Сохраняем модель
            self.save_model(epoch, val_acc, val_f1, is_best)
            
            # Проверяем достижение цели
            if val_f1 >= 0.90 and val_acc >= 0.90:
                print(f"🎯 ДОСТИГНУТА ЦЕЛЬ! F1: {val_f1:.4f}, Accuracy: {val_acc:.4f}")
                break
            
            # Early stopping
            if patience_counter >= patience:
                print(f"⏹️ Early stopping на эпохе {epoch+1}")
                break
        
        # Финальная статистика
        total_time = time.time() - start_time
        print(f"\n🏁 Обучение завершено!")
        print(f"⏱️ Время: {total_time/60:.1f} минут")
        print(f"📊 Лучший F1: {self.best_f1:.4f}")
        print(f"📊 Лучшая Accuracy: {self.best_accuracy:.4f}")
        print(f"📊 Лучшая эпоха: {best_epoch+1}")
        
        # Строим графики
        self.plot_metrics()
        
        return self.best_f1, self.best_accuracy

def main():
    """Основная функция"""
    print("🐕 Обучение усложненной FAN модели на Stanford Dogs")
    print("🎯 Цель: F1 > 0.90, Accuracy > 90%")
    print("=" * 60)
    
    # Проверяем наличие данных
    data_dir = Path("data/stanford_dogs_fan")
    if not data_dir.exists():
        print("❌ Датасет не найден! Сначала запустите download_stanford_dogs.py")
        return
    
    # Создаем тренер
    trainer = AdvancedStanfordDogsTrainer(
        data_dir=data_dir,
        model_dir="models/stanford_dogs",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Обучаем модель
    best_f1, best_acc = trainer.train(num_epochs=100, patience=15)
    
    # Результат
    if best_f1 >= 0.90 and best_acc >= 0.90:
        print(f"\n🎉 УСПЕХ! Достигнуты все цели!")
        print(f"✅ F1 Score >= 0.90: {best_f1:.4f}")
        print(f"✅ Accuracy >= 90%: {best_acc:.4f}")
        print("🚀 Модель готова для конференции уровня A!")
    else:
        print(f"\n⚠️ Цели не достигнуты полностью")
        print(f"❌ F1 Score: {best_f1:.4f} (требуется >= 0.90)")
        print(f"❌ Accuracy: {best_acc:.4f} (требуется >= 0.90)")
        print("💡 Попробуйте увеличить количество эпох или изменить архитектуру")

if __name__ == "__main__":
    main()

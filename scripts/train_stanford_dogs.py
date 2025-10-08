#!/usr/bin/env python3
"""
Полное обучение FAN модели на Stanford Dogs датасете
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
from universal_fan_model import UniversalFANModel

class StanfordDogsTrainer:
    """Тренер для Stanford Dogs FAN модели"""
    
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
        
        # Создаем DataLoader'ы
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=16, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=16, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Создаем модель
        self.model = UniversalFANModel(
            num_classes=self.train_dataset.num_classes,
            num_heads=8,  # Больше голов для сложной задачи
            hidden_dim=768,  # Больше размерности
            use_bert=True,
            use_resnet=True
        ).to(self.device)
        
        # Оптимизатор и функция потерь
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.criterion = nn.CrossEntropyLoss()
        
        # Метрики
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        
        self.best_f1 = 0.0
        self.best_accuracy = 0.0
        
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
            
            # Статистика
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Обновляем прогресс-бар
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
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
        checkpoint_path = self.model_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Сохраняем лучшую модель
        if is_best:
            best_path = self.model_dir / 'best_stanford_dogs_fan_model.pth'
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
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # F1 Score
        ax3.plot(self.val_f1_scores, label='Val F1 Score', color='orange')
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
        plt.savefig(self.model_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def train(self, num_epochs=50, patience=10):
        """Полное обучение модели"""
        print(f"🚀 Начинаем обучение на {num_epochs} эпох")
        print("=" * 50)
        
        start_time = time.time()
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\n📈 Эпоха {epoch+1}/{num_epochs}")
            print("-" * 30)
            
            # Обучение
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Валидация
            val_loss, val_acc, val_f1, val_prec, val_rec = self.validate_epoch(epoch)
            
            # Обновляем learning rate
            self.scheduler.step()
            
            # Выводим метрики
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Val F1: {val_f1:.4f}, Val Prec: {val_prec:.4f}, Val Rec: {val_rec:.4f}")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
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
            
            # Early stopping
            if patience_counter >= patience:
                print(f"⏹️ Early stopping на эпохе {epoch+1}")
                break
            
            # Проверяем достижение цели
            if val_f1 >= 0.75:
                print(f"🎯 Достигнута цель F1 >= 0.75: {val_f1:.4f}")
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
    print("🐕 Обучение FAN модели на Stanford Dogs")
    print("=" * 50)
    
    # Проверяем наличие данных
    data_dir = Path("data/stanford_dogs_fan")
    if not data_dir.exists():
        print("❌ Датасет не найден! Сначала запустите download_stanford_dogs.py")
        return
    
    # Создаем тренер
    trainer = StanfordDogsTrainer(
        data_dir=data_dir,
        model_dir="models/stanford_dogs",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Обучаем модель
    best_f1, best_acc = trainer.train(num_epochs=50, patience=10)
    
    # Результат
    if best_f1 >= 0.75:
        print(f"\n🎉 УСПЕХ! Достигнута цель F1 >= 0.75: {best_f1:.4f}")
    else:
        print(f"\n⚠️ Цель не достигнута. F1: {best_f1:.4f}")
        print("💡 Попробуйте изменить гиперпараметры или архитектуру")

if __name__ == "__main__":
    main()


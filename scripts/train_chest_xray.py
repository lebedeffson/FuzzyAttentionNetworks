#!/usr/bin/env python3
"""
Обучение FAN модели на Chest X-Ray Pneumonia датасете (реальный медицинский датасет)
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import sys
import os

# Добавляем src в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset_manager import DatasetManager
from advanced_fan_model import AdvancedFANModel

class ChestXRayTrainer:
    """Тренер для Chest X-Ray FAN модели"""
    
    def __init__(self, data_dir, model_dir, device='cuda'):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"🖥️ Используем устройство: {self.device}")
        
        # Создаем датасеты
        self.dataset_manager = DatasetManager()
        self.train_dataset = self.dataset_manager.create_dataset('chest_xray', 'train')
        self.val_dataset = self.dataset_manager.create_dataset('chest_xray', 'val')
        
        print(f"📊 Train samples: {len(self.train_dataset)}")
        print(f"📊 Val samples: {len(self.val_dataset)}")
        print(f"📊 Classes: {self.train_dataset.num_classes}")
        print(f"🏥 Medical task: Pneumonia detection from chest X-rays")
        
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
        
        # Создаем усложненную модель (как в HAM10000)
        self.model = AdvancedFANModel(
            num_classes=self.train_dataset.num_classes,
            num_heads=8,  # 8 голов
            hidden_dim=1024,  # Используем стандартный hidden_dim
            use_bert=True,
            use_resnet=True
        ).to(self.device)
        
        # Оптимизатор и функция потерь (еще больше регуляризации)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.3, patience=2)
        
        # Early stopping
        self.best_f1 = 0.0
        self.patience = 5
        self.patience_counter = 0
        
        # Метрики
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        self.best_f1 = 0.0
        self.best_epoch = 0
        
        print(f"🏥 Модель создана для медицинской диагностики пневмонии")
        print(f"📊 Параметров модели: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, epoch):
        """Обучение одной эпохи"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/20 [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            texts = batch['text_tokens'].to(self.device)
            attention_masks = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Прямой проход
            model_outputs = self.model(texts, attention_masks, images)
            outputs = model_outputs['logits']
            loss = self.criterion(outputs, labels)
            
            # Обратный проход
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Статистика
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Обновляем прогресс-бар
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, epoch):
        """Валидация одной эпохи"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/20 [Val]')
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                texts = batch['text_tokens'].to(self.device)
                attention_masks = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Прямой проход
                model_outputs = self.model(texts, attention_masks, images)
                outputs = model_outputs['logits']
                loss = self.criterion(outputs, labels)
                
                # Статистика
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Обновляем прогресс-бар
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Вычисляем метрики
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        
        return avg_loss, accuracy, f1, precision, recall, all_predictions, all_labels
    
    def train(self, num_epochs=20):
        """Полное обучение модели"""
        print(f"\n🏥 Начинаем обучение FAN модели на Chest X-Ray датасете")
        print(f"📊 Эпох: {num_epochs}")
        print(f"🏥 Медицинская задача: Диагностика пневмонии по рентгеновским снимкам")
        print("=" * 70)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Обучение
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Валидация
            val_loss, val_acc, val_f1, val_precision, val_recall, predictions, labels = self.validate_epoch(epoch)
            
            # Обновляем learning rate
            self.scheduler.step(val_f1)
            
            # Early stopping
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.patience_counter = 0
                # Сохраняем лучшую модель
                self.save_model(epoch, val_f1, val_acc, start_time)
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                print(f"🛑 Early stopping на эпохе {epoch+1}")
                break
            
            # Сохраняем метрики
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.val_f1_scores.append(val_f1)
            
            # Проверяем лучшую модель
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.best_epoch = epoch
                self.save_model(epoch, val_f1, val_acc)
            
            epoch_time = time.time() - epoch_start
            
            # Выводим статистику
            print(f"\n📊 Epoch {epoch+1}/{num_epochs} завершена за {epoch_time:.1f}s")
            print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"   Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
            print(f"   Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            if val_f1 == self.best_f1:
                print(f"   🏆 Новый лучший F1: {val_f1:.4f}")
        
        total_time = time.time() - start_time
        print(f"\n🎉 Обучение завершено за {total_time/60:.1f} минут")
        print(f"🏆 Лучший F1 Score: {self.best_f1:.4f} на эпохе {self.best_epoch+1}")
        
        # Сохраняем финальные метрики
        self.save_training_metrics()
        self.plot_training_curves()
        
        return self.best_f1
    
    def save_model(self, epoch, f1_score, accuracy, start_time):
        """Сохранить модель"""
        model_path = self.model_dir / f"best_chest_xray_fan_model.pth"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'f1_score': f1_score,
            'accuracy': accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_f1_scores': self.val_f1_scores,
            'training_time': time.time() - start_time,
            'dataset_info': {
                'name': 'Chest X-Ray Pneumonia',
                'classes': self.train_dataset.class_names,
                'num_classes': self.train_dataset.num_classes,
                'medical_domain': 'Radiology',
                'diagnostic_task': 'Pneumonia detection'
            }
        }
        
        torch.save(checkpoint, model_path)
        print(f"💾 Модель сохранена: {model_path}")
    
    def save_training_metrics(self):
        """Сохранить метрики обучения"""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_f1_scores': self.val_f1_scores,
            'best_f1': self.best_f1,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.train_losses),
            'dataset': 'chest_xray',
            'medical_task': 'pneumonia_detection'
        }
        
        metrics_path = self.model_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def plot_training_curves(self):
        """Построить графики обучения"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.set_title('Chest X-Ray FAN Training - Loss Curves', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, self.val_accuracies, 'g-', label='Val Accuracy', linewidth=2)
        ax2.set_title('Chest X-Ray FAN Training - Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # F1 Score curves
        ax3.plot(epochs, self.val_f1_scores, 'm-', label='Val F1 Score', linewidth=2)
        ax3.set_title('Chest X-Ray FAN Training - F1 Score', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Combined metrics
        ax4.plot(epochs, self.val_accuracies, 'g-', label='Accuracy', linewidth=2)
        ax4.plot(epochs, self.val_f1_scores, 'm-', label='F1 Score', linewidth=2)
        ax4.set_title('Chest X-Ray FAN Training - Combined Metrics', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'chest_xray_training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Графики обучения сохранены: {self.model_dir / 'chest_xray_training_metrics.png'}")

def main():
    """Основная функция"""
    print("🫁 Обучение FAN модели на Chest X-Ray Pneumonia датасете")
    print("=" * 60)
    print("🏥 Реальный медицинский датасет рентгеновских снимков")
    print("🔬 Задача: Диагностика пневмонии")
    print("=" * 60)
    
    # Проверяем наличие данных
    data_dir = Path("data/chest_xray_fan")
    if not data_dir.exists():
        print("❌ Датасет не найден! Сначала запустите:")
        print("   python scripts/download_chest_xray.py")
        return False
    
    # Создаем тренер
    trainer = ChestXRayTrainer(
        data_dir=data_dir,
        model_dir="models/chest_xray",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Обучаем модель
    best_f1 = trainer.train(num_epochs=20)
    
    print(f"\n🎉 Обучение завершено!")
    print(f"🏆 Лучший F1 Score: {best_f1:.4f}")
    print(f"📁 Модель сохранена в: models/chest_xray/")
    print(f"🏥 Готово для медицинской диагностики пневмонии!")
    
    return True

if __name__ == "__main__":
    main()

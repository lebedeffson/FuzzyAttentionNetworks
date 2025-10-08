#!/usr/bin/env python3
"""
Скрипт для обучения FAN модели на датасете HAM10000
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

# Добавляем src в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset_manager import DatasetManager
from advanced_fan_model import AdvancedFANModel

class Trainer:
    """Менеджер для обучения FAN моделей"""
    
    def __init__(self, dataset_name, batch_size=8, learning_rate=1e-4):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.dataset_manager = DatasetManager()
        
        # Создаем даталоадеры
        self.train_dataloader = self.dataset_manager.create_dataloader(
            dataset_name, split='train', batch_size=batch_size, use_balanced_sampling=True
        )
        self.val_dataloader = self.dataset_manager.create_dataloader(
            dataset_name, split='val', batch_size=batch_size, use_balanced_sampling=False
        )
        
        self.train_dataset = self.dataset_manager.create_dataset(dataset_name, split='train')
        self.val_dataset = self.dataset_manager.create_dataset(dataset_name, split='val')
        
        # Создаем усложненную модель
        self.model = AdvancedFANModel(
            num_classes=self.train_dataset.num_classes,
            num_heads=8,  # 8 голов (512 делится на 8)
            hidden_dim=512,  # Оптимальная размерность для HAM10000
            use_bert=True,
            use_resnet=True
        ).to(self.device)
        
        # Оптимизатор и loss
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=15, eta_min=1e-6)
        self.criterion = nn.CrossEntropyLoss()
        
        # Метрики
        self.train_losses = []
        self.val_losses = []
        self.f1_scores = []
        self.accuracies = []
        
        print(f"🚀 Инициализирован тренер для {dataset_name}")
        print(f"📊 Классов: {self.train_dataset.num_classes}")
        print(f"📊 Train samples: {len(self.train_dataset)}")
        print(f"📊 Val samples: {len(self.val_dataset)}")
        print(f"🖥️ Device: {self.device}")

    def train_epoch(self):
        """Обучение одной эпохи"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Перемещаем данные на устройство
            text_tokens = batch['text_tokens'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            image_features = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Обнуляем градиенты
            self.optimizer.zero_grad()
            
            # Forward pass
            model_output = self.model(text_tokens, attention_mask, image_features)
            if isinstance(model_output, dict):
                outputs = model_output['logits']
            else:
                outputs = model_output
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Собираем метрики
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Обновляем progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        # Вычисляем метрики
        avg_loss = total_loss / len(self.train_dataloader)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, f1, accuracy

    def validate_epoch(self):
        """Валидация одной эпохи"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Перемещаем данные на устройство
                text_tokens = batch['text_tokens'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                image_features = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                model_output = self.model(text_tokens, attention_mask, image_features)
                if isinstance(model_output, dict):
                    outputs = model_output['logits']
                else:
                    outputs = model_output
                loss = self.criterion(outputs, labels)
                
                # Собираем метрики
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Вычисляем метрики
        avg_loss = total_loss / len(self.val_dataloader)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        
        return avg_loss, f1, accuracy, precision, recall

    def train(self, num_epochs=15):
        """Полное обучение модели"""
        print(f"🚀 Начинаем обучение на {num_epochs} эпох")
        print("=" * 60)
        
        best_f1 = 0
        best_accuracy = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\n📅 Эпоха {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # Обучение
            train_loss, train_f1, train_acc = self.train_epoch()
            
            # Валидация
            val_loss, val_f1, val_acc, val_precision, val_recall = self.validate_epoch()
            
            # Обновляем scheduler
            self.scheduler.step()
            
            # Сохраняем метрики
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.f1_scores.append(val_f1)
            self.accuracies.append(val_acc)
            
            # Выводим результаты
            print(f"📊 Train Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Acc: {train_acc:.4f}")
            print(f"📊 Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Acc: {val_acc:.4f}")
            print(f"📊 Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
            print(f"📊 Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Сохраняем лучшую модель
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_accuracy = val_acc
                patience_counter = 0
                self.save_model(f"best_{self.dataset_name}_fan_model.pth")
                print(f"✅ Новая лучшая модель! F1: {best_f1:.4f}, Acc: {best_accuracy:.4f}")
            else:
                patience_counter += 1
                print(f"⏳ Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping на эпохе {epoch+1}")
                break
        
        print(f"\n🎉 Обучение завершено!")
        print(f"🏆 Лучший F1 Score: {best_f1:.4f}")
        print(f"🏆 Лучшая Accuracy: {best_accuracy:.4f}")
        
        # Сохраняем графики
        self.save_plots()
        
        return best_f1, best_accuracy

    def save_model(self, filename):
        """Сохранить модель"""
        model_dir = Path(f"models/{self.dataset_name}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'num_classes': self.train_dataset.num_classes,
            'class_names': self.train_dataset.class_names,
            'dataset_name': self.dataset_name
        }, model_path)
        
        print(f"💾 Модель сохранена: {model_path}")

    def save_plots(self):
        """Сохранить графики обучения"""
        model_dir = Path(f"models/{self.dataset_name}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # График loss
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Val Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.f1_scores, label='F1 Score', color='green')
        plt.plot(self.accuracies, label='Accuracy', color='orange')
        plt.title('F1 Score and Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(model_dir / f"{self.dataset_name}_training_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Графики сохранены: {model_dir / f'{self.dataset_name}_training_metrics.png'}")

def main():
    print("🏥 Обучение FAN модели на HAM10000")
    print("=" * 50)
    
    # Проверяем CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA доступна: {torch.cuda.get_device_name()}")
    else:
        print("⚠️ CUDA не доступна, используется CPU")
    
    # Создаем тренер
    trainer = Trainer(
        dataset_name='ham10000',
        batch_size=8,
        learning_rate=1e-4
    )
    
    # Обучаем модель
    best_f1, best_accuracy = trainer.train(num_epochs=15)
    
    # Проверяем достижение цели
    target_f1 = 0.90
    target_accuracy = 0.90
    
    print(f"\n🎯 Результаты:")
    print(f"📊 F1 Score: {best_f1:.4f} (цель: {target_f1:.2f})")
    print(f"📊 Accuracy: {best_accuracy:.4f} (цель: {target_accuracy:.2f})")
    
    if best_f1 >= target_f1 and best_accuracy >= target_accuracy:
        print("🎉 ЦЕЛЬ ДОСТИГНУТА! Модель готова для конференции уровня A!")
    else:
        print("⚠️ Цель не достигнута, но модель показывает хорошие результаты")
    
    print(f"\n📁 Модель сохранена в: models/ham10000/")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Финальный скрипт для очистки и подготовки проекта к публикации
"""

import os
import shutil
from pathlib import Path

def clean_project():
    """Очистка проекта от ненужных файлов"""
    print("🧹 Очистка проекта...")
    
    # Удаляем ненужные файлы
    files_to_remove = [
        'run_web_interface.py',  # Старый интерфейс
        'demos/final_web_interface.py',  # Старый интерфейс
        'models/best_cifar10_fan_model.pth',  # Старая модель
        'models/best_improved_cifar10_fan_model.pth',  # Старая модель
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"  ❌ Удален: {file_path}")
    
    # Очищаем пустые папки
    empty_dirs = ['datasets', 'results']
    for dir_path in empty_dirs:
        if os.path.exists(dir_path) and not os.listdir(dir_path):
            os.rmdir(dir_path)
            print(f"  ❌ Удалена пустая папка: {dir_path}")
    
    print("✅ Очистка завершена!")

def create_final_structure():
    """Создание финальной структуры проекта"""
    print("📁 Создание финальной структуры...")
    
    # Создаем папки если их нет
    dirs_to_create = [
        'docs',
        'examples',
        'scripts'
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✅ Создана папка: {dir_path}")
    
    print("✅ Структура создана!")

def create_example_usage():
    """Создание примера использования"""
    print("📝 Создание примера использования...")
    
    example_code = '''#!/usr/bin/env python3
"""
Пример использования FAN моделей
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset_manager import DatasetManager
from universal_fan_model import ModelManager
from transformers import BertTokenizer
import torch
from PIL import Image
import torchvision.transforms as transforms

def main():
    """Пример работы с FAN моделями"""
    
    # Инициализация
    dataset_manager = DatasetManager()
    model_manager = ModelManager()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Выбор датасета
    dataset_name = 'cifar10'  # или 'hateful_memes'
    
    print(f"🎯 Работа с датасетом: {dataset_name}")
    
    # Загрузка датасета
    dataset = dataset_manager.create_dataset(dataset_name, 'train')
    print(f"📊 Загружено {len(dataset)} образцов")
    
    # Загрузка модели
    model = model_manager.get_model(dataset_name)
    print("🤖 Модель загружена")
    
    # Подготовка данных
    text = "This is a sample text for testing"
    image = Image.new('RGB', (224, 224), color='blue')
    
    # Токенизация текста
    text_tokens = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=64,
        return_tensors='pt'
    )
    
    # Трансформация изображения
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    
    # Предсказание
    with torch.no_grad():
        result = model_manager.predict(
            dataset_name,
            text_tokens['input_ids'],
            text_tokens['attention_mask'],
            image_tensor,
            return_explanations=True
        )
    
    # Результаты
    prediction = result['predictions'].item()
    confidence = result['confidence'].item()
    
    print(f"🔮 Предсказание: {dataset.class_names[prediction]}")
    print(f"📊 Уверенность: {confidence:.2%}")
    
    # Вероятности по классам
    probs = result['probs'].cpu().numpy()[0]
    print("\\n📈 Вероятности по классам:")
    for i, (class_name, prob) in enumerate(zip(dataset.class_names, probs)):
        print(f"  {class_name}: {prob:.3f}")

if __name__ == "__main__":
    main()
'''
    
    with open('examples/basic_usage.py', 'w', encoding='utf-8') as f:
        f.write(example_code)
    
    print("  ✅ Создан: examples/basic_usage.py")

def create_documentation():
    """Создание документации"""
    print("📚 Создание документации...")
    
    # Создаем файл с архитектурой
    arch_doc = '''# 🏗️ FAN Architecture Documentation

## Overview
Fuzzy Attention Networks (FAN) implement a novel approach to multimodal learning using fuzzy logic principles for attention mechanisms.

## Key Components

### 1. Universal FAN Model
- **File**: `src/universal_fan_model.py`
- **Purpose**: Core model architecture supporting multiple datasets
- **Features**: 
  - Configurable for different datasets
  - Transfer learning with BERT and ResNet
  - Fuzzy attention mechanisms

### 2. Dataset Manager
- **File**: `src/dataset_manager.py`
- **Purpose**: Unified interface for dataset handling
- **Features**:
  - Support for multiple datasets
  - Automatic data loading and preprocessing
  - Balanced sampling for class imbalance

### 3. Web Interface
- **File**: `demos/universal_web_interface.py`
- **Purpose**: Interactive web application
- **Features**:
  - Dataset selection
  - Real-time predictions
  - Interpretability visualizations

## Model Variants

### Hateful Memes Model
- **Architecture**: BERT + ResNet50 + 8-Head FAN
- **Performance**: F1: 0.5649, Accuracy: 59%
- **Use Case**: Binary classification of hateful content

### CIFAR-10 Model
- **Architecture**: BERT + ResNet18 + 4-Head FAN
- **Performance**: F1: 0.8808, Accuracy: 85%
- **Use Case**: 10-class image classification

## Fuzzy Attention Mechanism

The core innovation is the use of fuzzy membership functions for attention:

1. **Bell-shaped Functions**: Soft attention boundaries
2. **Learnable Parameters**: Centers and widths
3. **Multi-head Architecture**: Parallel attention heads
4. **Interpretability**: Human-readable patterns

## Usage

See `examples/basic_usage.py` for detailed usage examples.
'''
    
    with open('docs/architecture.md', 'w', encoding='utf-8') as f:
        f.write(arch_doc)
    
    print("  ✅ Создана документация: docs/architecture.md")

def main():
    """Основная функция финализации"""
    print("🚀 Финализация проекта FAN...")
    print("=" * 50)
    
    # Очистка
    clean_project()
    
    # Структура
    create_final_structure()
    
    # Примеры
    create_example_usage()
    
    # Документация
    create_documentation()
    
    print("=" * 50)
    print("✅ Проект финализирован!")
    print("🎯 Готов к публикации и использованию")
    print("🌐 Запуск: python run_universal_interface.py")

if __name__ == "__main__":
    main()


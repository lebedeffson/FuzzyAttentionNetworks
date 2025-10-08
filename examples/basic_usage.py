#!/usr/bin/env python3
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
    print("\n📈 Вероятности по классам:")
    for i, (class_name, prob) in enumerate(zip(dataset.class_names, probs)):
        print(f"  {class_name}: {prob:.3f}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Универсальный менеджер датасетов для FAN моделей
Поддерживает Hateful Memes и CIFAR-10
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import json
import numpy as np
from pathlib import Path
from PIL import Image
import random
from tqdm import tqdm
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertModel
import torchvision.models as models
import torch.nn.functional as F
from collections import Counter
import math

class BaseFANDataset(Dataset):
    """Базовый класс для FAN датасетов"""
    
    def __init__(self, data_dir, split='train', max_length=64):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_length = max_length
        
        # Простой токенизатор (без BERT для избежания проблем с Hugging Face)
        class SimpleTokenizer:
            def __init__(self):
                self.vocab = {}
                self.vocab_size = 10000
                
            def encode(self, text, max_length=64, padding=True, truncation=True):
                # Простая токенизация по словам
                words = text.lower().split()[:max_length]
                tokens = [hash(word) % self.vocab_size for word in words]
                
                if padding and len(tokens) < max_length:
                    tokens.extend([0] * (max_length - len(tokens)))
                elif truncation and len(tokens) > max_length:
                    tokens = tokens[:max_length]
                    
                return tokens
                
            def __call__(self, text, max_length=64, padding=True, truncation=True, return_tensors=None):
                tokens = self.encode(text, max_length, padding, truncation)
                if return_tensors == 'pt':
                    import torch
                    return {'input_ids': torch.tensor([tokens])}
                return {'input_ids': [tokens]}
        
        self.tokenizer = SimpleTokenizer()
        
        # Базовые трансформации с аугментацией
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        
        # Загружаем изображение
        img_path = self.data_dir / entry.get('img', entry.get('image_path', ''))
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)
        
        # Токенизируем текст
        text = entry['text']
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Создаем attention_mask если его нет
        if 'attention_mask' not in encoding:
            input_ids = encoding['input_ids']
            attention_mask = (input_ids != 0).long()
            encoding['attention_mask'] = attention_mask
        
        return {
            'image': image_tensor,
            'text_tokens': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': entry['label'],
            'text': text,
            'class_name': entry.get('class_name', f'class_{entry["label"]}')
        }


class CIFAR10FANDataset(BaseFANDataset):
    """Датасет CIFAR-10 для FAN модели"""
    
    def __init__(self, data_dir, split='train'):
        super().__init__(data_dir, split)
        
        # Загружаем метаданные
        metadata_file = self.data_dir / "train.jsonl"
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.entries = [json.loads(line) for line in f]
        
        # Разделяем на train/val
        random.shuffle(self.entries)
        split_idx = int(0.8 * len(self.entries))
        
        if split == 'train':
            self.entries = self.entries[:split_idx]
        else:
            self.entries = self.entries[split_idx:]
        
        self.num_classes = 10
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']

class StanfordDogsFANDataset(BaseFANDataset):
    """Датасет Stanford Dogs для FAN модели"""
    
    def __init__(self, data_dir, split='train'):
        super().__init__(data_dir, split)
        
        # Загружаем метаданные
        metadata_file = self.data_dir / "train.jsonl"
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.entries = [json.loads(line) for line in f]
        
        # Разделяем на train/val
        random.shuffle(self.entries)
        split_idx = int(0.8 * len(self.entries))
        
        if split == 'train':
            self.entries = self.entries[:split_idx]
        else:
            self.entries = self.entries[split_idx:]
        
        # Загружаем информацию о классах
        info_file = self.data_dir / "dataset_info.json"
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
                self.class_names = info.get('classes', [])
                self.num_classes = len(self.class_names)
        else:
            # Fallback
            unique_labels = set(entry['label'] for entry in self.entries)
            self.num_classes = len(unique_labels)
            self.class_names = [f'Dog_Class_{i}' for i in range(self.num_classes)]

class HAM10000FANDataset(BaseFANDataset):
    """Датасет HAM10000 для FAN модели (кожные заболевания)"""
    
    def __init__(self, data_dir, split='train'):
        super().__init__(data_dir, split)
        
        # Загружаем метаданные
        metadata_file = self.data_dir / "train.jsonl"
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.entries = [json.loads(line) for line in f]
        
        # Разделяем на train/val
        random.shuffle(self.entries)
        split_idx = int(0.8 * len(self.entries))
        
        if split == 'train':
            self.entries = self.entries[:split_idx]
        else:
            self.entries = self.entries[split_idx:]
        
        # Загружаем информацию о классах
        info_file = self.data_dir / "dataset_info.json"
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
                self.class_names = info.get('classes', [])
                self.num_classes = len(self.class_names)
        else:
            # Fallback
            unique_labels = set(entry['label'] for entry in self.entries)
            self.num_classes = len(unique_labels)
            self.class_names = [f'Skin_Class_{i}' for i in range(self.num_classes)]

class ChestXRayFANDataset(BaseFANDataset):
    """Датасет Chest X-Ray Pneumonia для FAN модели"""
    
    def __init__(self, data_dir, split='train'):
        super().__init__(data_dir, split)
        
        # Загружаем метаданные
        metadata_file = self.data_dir / "train.jsonl"
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.entries = [json.loads(line) for line in f]
        
        # Разделяем на train/val (70/30) - больше данных для валидации
        random.shuffle(self.entries)
        split_idx = int(0.7 * len(self.entries))
        
        if split == 'train':
            self.entries = self.entries[:split_idx]
        else:
            self.entries = self.entries[split_idx:]
        
        # Загружаем информацию о классах
        info_file = self.data_dir / "dataset_info.json"
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
                self.class_names = info.get('classes', [])
                self.num_classes = len(self.class_names)
        else:
            # Fallback
            unique_labels = set(entry['label'] for entry in self.entries)
            self.num_classes = len(unique_labels)
            self.class_names = [f'Chest_Class_{i}' for i in range(self.num_classes)]

class DatasetManager:
    """Менеджер для работы с разными датасетами"""
    
    def __init__(self):
        self.datasets = {
            'stanford_dogs': {
                'dataset_class': StanfordDogsFANDataset,
                'data_path': 'data/stanford_dogs_fan',
                'model_path': 'models/stanford_dogs/best_advanced_stanford_dogs_fan_model.pth',
                'description': 'Stanford Dogs Classification - 20 Classes'
            },
            'cifar10': {
                'dataset_class': CIFAR10FANDataset,
                'data_path': 'data/cifar10_fan',
                'model_path': 'models/cifar10/best_simple_cifar10_fan_model.pth',
                'description': 'CIFAR-10 Classification - 10 Classes'
            },
            'ham10000': {
                'dataset_class': HAM10000FANDataset,
                'data_path': 'data/ham10000_fan',
                'model_path': 'models/ham10000/best_ham10000_fan_model.pth',
                'description': 'HAM10000 Skin Lesion Classification - 7 Classes'
            },
            'chest_xray': {
                'dataset_class': ChestXRayFANDataset,
                'data_path': 'data/chest_xray_fan',
                'model_path': 'models/chest_xray/best_chest_xray_fan_model.pth',
                'description': 'Chest X-Ray Pneumonia Classification - 2 Classes'
            }
        }
    
    def get_dataset_info(self, dataset_name):
        """Получить информацию о датасете"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return self.datasets[dataset_name]
    
    def create_dataset(self, dataset_name, split='train'):
        """Создать датасет"""
        info = self.get_dataset_info(dataset_name)
        return info['dataset_class'](info['data_path'], split)
    
    def create_balanced_sampler(self, dataset):
        """Создать сбалансированный сэмплер"""
        labels = [dataset[i]['label'] for i in range(len(dataset))]
        class_counts = Counter(labels)
        
        # Вычисляем веса классов
        class_weights = {}
        total_samples = len(labels)
        num_classes = len(class_counts)
        
        for class_id, count in class_counts.items():
            class_weights[class_id] = total_samples / (num_classes * count)
        
        # Создаем веса для каждого образца
        sample_weights = [class_weights[label] for label in labels]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def create_dataloader(self, dataset_name, split='train', batch_size=16, use_balanced_sampling=True):
        """Создать DataLoader"""
        dataset = self.create_dataset(dataset_name, split)
        
        if split == 'train' and use_balanced_sampling:
            sampler = self.create_balanced_sampler(dataset)
            shuffle = False
        else:
            sampler = None
            shuffle = (split == 'train')
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def get_available_datasets(self):
        """Получить список доступных датасетов"""
        return list(self.datasets.keys())



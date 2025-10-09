#!/usr/bin/env python3
"""
Скрипт для скачивания и обработки Stanford Dogs датасета
"""

import os
import requests
import zipfile
import shutil
from pathlib import Path
import random
from PIL import Image
import json
from tqdm import tqdm
import torchvision.transforms as transforms

def download_file(url, filename):
    """Скачать файл с прогресс-баром"""
    print(f"Скачиваем {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = file.write(chunk)
            progress_bar.update(size)
    
    print(f"✅ {filename} скачан успешно!")

def download_stanford_dogs():
    """Скачать Stanford Dogs датасет"""
    data_dir = Path("data/stanford_dogs")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # URL для скачивания Stanford Dogs
    url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    archive_path = data_dir / "images.tar"
    
    if not archive_path.exists():
        download_file(url, str(archive_path))
    
    # Распаковываем архив
    extract_dir = data_dir / "extracted"
    if not extract_dir.exists():
        print("Распаковываем архив...")
        os.system(f"cd {data_dir} && tar -xf images.tar")
        print("✅ Архив распакован!")
    
    return data_dir

def create_fan_dataset(data_dir, max_samples_per_class=15):
    """Создать FAN датасет из Stanford Dogs"""
    print("Создаем FAN датасет...")
    
    # Создаем структуру папок
    fan_dir = Path("data/stanford_dogs_fan")
    fan_dir.mkdir(parents=True, exist_ok=True)
    img_dir = fan_dir / "img"
    img_dir.mkdir(exist_ok=True)
    
    # Получаем список всех классов
    images_dir = data_dir / "Images"
    if not images_dir.exists():
        # Пробуем найти папку с изображениями
        for subdir in data_dir.iterdir():
            if subdir.is_dir() and "Images" in subdir.name:
                images_dir = subdir
                break
    
    if not images_dir.exists():
        raise FileNotFoundError("Не найдена папка с изображениями!")
    
    # Получаем все классы
    all_classes = [d.name for d in images_dir.iterdir() if d.is_dir()]
    print(f"Найдено {len(all_classes)} классов")
    
    # Выбираем 20 случайных классов для FAN датасета
    selected_classes = random.sample(all_classes, min(20, len(all_classes)))
    print(f"Выбрано {len(selected_classes)} классов для FAN датасета")
    
    # Создаем метаданные
    entries = []
    class_names = []
    
    for class_idx, class_name in enumerate(selected_classes):
        class_dir = images_dir / class_name
        if not class_dir.exists():
            continue
            
        # Получаем все изображения в классе
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
        
        # Выбираем случайные изображения
        selected_images = random.sample(image_files, min(max_samples_per_class, len(image_files)))
        
        class_names.append(class_name.replace("_", " ").title())
        
        for img_file in selected_images:
            # Копируем изображение
            new_img_name = f"{class_idx:03d}_{img_file.stem}.jpg"
            new_img_path = img_dir / new_img_name
            
            # Открываем и сохраняем как JPG
            try:
                with Image.open(img_file) as img:
                    img = img.convert('RGB')
                    img.save(new_img_path, 'JPEG', quality=95)
            except Exception as e:
                print(f"Ошибка при обработке {img_file}: {e}")
                continue
            
            # Создаем описание для текста
            description = f"A photo of a {class_name.replace('_', ' ')} dog"
            
            # Добавляем запись
            entries.append({
                'img': f'img/{new_img_name}',
                'text': description,
                'label': class_idx,
                'class_name': class_names[-1]
            })
    
    # Сохраняем метаданные
    with open(fan_dir / "train.jsonl", 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Создаем dataset_info.json
    dataset_info = {
        "name": "Stanford Dogs FAN Subset",
        "description": "Subset of Stanford Dogs dataset adapted for Fuzzy Attention Networks",
        "total_samples": len(entries),
        "classes": class_names,
        "samples_per_class": max_samples_per_class,
        "image_size": "224x224",
        "source": "Stanford Dogs (http://vision.stanford.edu/aditya86/ImageNetDogs/)",
        "license": "Academic Use",
        "created_for": "Fuzzy Attention Networks research"
    }
    
    with open(fan_dir / "dataset_info.json", 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ FAN датасет создан!")
    print(f"📊 Всего образцов: {len(entries)}")
    print(f"📊 Классов: {len(class_names)}")
    print(f"📊 Образцов на класс: {max_samples_per_class}")
    
    return fan_dir

def main():
    """Основная функция"""
    print("🐕 Скачивание и обработка Stanford Dogs датасета")
    print("=" * 50)
    
    # Скачиваем датасет
    data_dir = download_stanford_dogs()
    
    # Создаем FAN датасет
    fan_dir = create_fan_dataset(data_dir, max_samples_per_class=15)
    
    print("\n🎉 Готово!")
    print(f"📁 FAN датасет сохранен в: {fan_dir}")
    print("🚀 Теперь можно запускать обучение!")

if __name__ == "__main__":
    main()



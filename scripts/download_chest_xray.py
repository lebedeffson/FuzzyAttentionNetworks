#!/usr/bin/env python3
"""
Скачивание Chest X-Ray Pneumonia датасета с Kaggle (реальный медицинский датасет)
"""

import os
import sys
import json
import shutil
import zipfile
from pathlib import Path
import random
from PIL import Image
import kaggle
import time

def download_chest_xray():
    """Скачать реальный Chest X-Ray Pneumonia датасет с Kaggle"""
    
    print("🫁 Скачивание Chest X-Ray Pneumonia датасета с Kaggle")
    print("=" * 60)
    print("📊 Цель: Реальный медицинский датасет рентгеновских снимков")
    print("🏥 Классификация: Normal vs Pneumonia")
    
    # Создаем директории
    download_dir = Path("data/kaggle_chest_temp")
    if download_dir.exists():
        shutil.rmtree(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 Скачиваем реальный датасет...")
    print("🔗 Источник: paultimothymooney/chest-xray-pneumonia")
    print("⚠️ Это настоящие медицинские рентгеновские снимки!")
    
    try:
        # Скачиваем реальный датасет
        print("📸 Скачиваем Chest X-Ray датасет...")
        print("📊 Размер датасета: ~1.2 GB")
        print("⏳ Это может занять 5-10 минут...")
        
        # Показываем прогресс
        import time
        start_time = time.time()
        
        kaggle.api.dataset_download_files(
            'paultimothymooney/chest-xray-pneumonia',
            path=str(download_dir),
            unzip=True
        )
        
        download_time = time.time() - start_time
        print(f"✅ Скачивание завершено за {download_time/60:.1f} минут")
        
        print("✅ Реальный датасет скачан!")
        return process_chest_xray_dataset(download_dir)
        
    except Exception as e:
        print(f"❌ Ошибка при скачивании: {e}")
        return False

def process_chest_xray_dataset(download_dir):
    """Обработать скачанный Chest X-Ray датасет"""
    
    print("🔍 Обрабатываем реальные рентгеновские снимки...")
    
    # Ищем папки с данными (игнорируем __MACOSX)
    train_dir = None
    test_dir = None
    
    for item in download_dir.rglob("*"):
        if item.is_dir() and "train" in item.name.lower() and "__MACOSX" not in str(item):
            # Проверяем, что в папке есть подпапки с классами
            has_classes = any(subdir.is_dir() for subdir in item.iterdir() if subdir.name in ['NORMAL', 'PNEUMONIA'])
            if has_classes:
                train_dir = item
        elif item.is_dir() and "test" in item.name.lower() and "__MACOSX" not in str(item):
            test_dir = item
    
    if not train_dir:
        print("❌ Не найдена папка train")
        return False
    
    print(f"📁 Train папка: {train_dir}")
    if test_dir:
        print(f"📁 Test папка: {test_dir}")
    
    # Собираем все изображения
    all_images = []
    
    # Обрабатываем train папку
    for class_dir in train_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            print(f"📂 Обрабатываем класс: {class_name}")
            
            for img_file in class_dir.glob("*.jpeg"):
                # Пропускаем системные файлы macOS
                if '__MACOSX' in str(img_file) or img_file.name.startswith('._'):
                    continue
                all_images.append((img_file, class_name))
    
    print(f"📊 Найдено {len(all_images)} реальных рентгеновских снимков")
    
    if len(all_images) == 0:
        print("❌ Не найдены изображения")
        return False
    
    # Создаем выборку (берем все доступные изображения)
    sample_size = min(1000, len(all_images))  # Ограничиваем для быстрого обучения
    sample_images = random.sample(all_images, sample_size)
    
    print(f"📸 Создаем выборку из {sample_size} реальных снимков")
    
    # Создаем структуру для FAN
    fan_dir = Path("data/chest_xray_fan")
    if fan_dir.exists():
        shutil.rmtree(fan_dir)
    fan_dir.mkdir(parents=True, exist_ok=True)
    
    # Классы Chest X-Ray
    class_names = [
        "Normal",      # 0 - Нормальные легкие
        "Pneumonia"    # 1 - Пневмония
    ]
    
    # Обрабатываем изображения
    metadata = []
    for i, (img_path, class_name) in enumerate(sample_images):
        new_name = f"chest_xray_{i:04d}.jpeg"
        new_path = fan_dir / new_name
        
        try:
            # Открываем и конвертируем изображение
            img = Image.open(img_path).convert('RGB')
            
            # Изменяем размер до 224x224
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Сохраняем
            img.save(new_path, 'JPEG', quality=95)
            
            # Определяем класс
            label = 0 if class_name.lower() == "normal" else 1
            class_display_name = class_names[label]
            
            # Создаем медицинское описание
            if label == 0:
                text_description = "Chest X-ray showing normal lung appearance with clear lung fields, no signs of infection or abnormality"
            else:
                text_description = "Chest X-ray showing signs of pneumonia with lung consolidation, opacity, and potential infection patterns"
            
            metadata.append({
                "img": new_name,
                "text": text_description,
                "label": label,
                "class_name": class_display_name,
                "original_path": str(img_path),
                "medical_condition": class_name
            })
            
        except Exception as e:
            print(f"⚠️ Ошибка при обработке {img_path}: {e}")
            continue
    
    # Сохраняем train.jsonl
    with open(fan_dir / "train.jsonl", 'w', encoding='utf-8') as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Сохраняем dataset_info.json
    dataset_info = {
        "name": "Chest X-Ray Pneumonia Real Dataset",
        "description": "Real medical chest X-ray images for pneumonia classification from Kaggle",
        "classes": class_names,
        "num_classes": 2,
        "total_samples": len(metadata),
        "source": "Kaggle: paultimothymooney/chest-xray-pneumonia",
        "download_date": "2024-12-19",
        "note": "Real medical X-ray images - genuine diagnostic data",
        "medical_domain": "Radiology",
        "diagnostic_task": "Pneumonia detection",
        "image_type": "Chest X-ray",
        "classes_description": {
            "Normal": "Healthy lungs with clear lung fields",
            "Pneumonia": "Lung infection with consolidation and opacity"
        }
    }
    
    with open(fan_dir / "dataset_info.json", 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Создан реальный медицинский датасет")
    print(f"📁 Путь: {fan_dir}")
    print(f"📊 Изображений: {len(metadata)}")
    print(f"🏷️ Классов: {len(class_names)}")
    print(f"📏 Размер: 224x224")
    print(f"🏥 Это настоящие медицинские рентгеновские снимки!")
    print(f"🔬 Диагностическая задача: Обнаружение пневмонии")
    
    # Статистика по классам
    normal_count = sum(1 for entry in metadata if entry['label'] == 0)
    pneumonia_count = sum(1 for entry in metadata if entry['label'] == 1)
    
    print(f"\n📊 Статистика классов:")
    print(f"   Normal: {normal_count} снимков")
    print(f"   Pneumonia: {pneumonia_count} снимков")
    
    # Удаляем временную папку
    shutil.rmtree(download_dir)
    print("🗑️ Временные файлы удалены")
    
    return True

def main():
    """Основная функция"""
    try:
        if download_chest_xray():
            print("\n🎉 Реальный Chest X-Ray датасет готов!")
            print("📁 Путь: data/chest_xray_fan/")
            print("💡 Теперь можно обучать модель на реальных медицинских данных")
            print("🏥 Это настоящие рентгеновские снимки для диагностики пневмонии!")
        else:
            print("\n❌ Не удалось скачать датасет")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()

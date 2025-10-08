#!/usr/bin/env python3
"""
Полный эксперимент с Stanford Dogs датасетом
Скачивание, обучение и тестирование
"""

import subprocess
import sys
import os
from pathlib import Path
import torch

def run_script(script_path, description):
    """Запустить скрипт с описанием"""
    print(f"\n🚀 {description}")
    print("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True)
        print("✅ Успешно выполнено!")
        if result.stdout:
            print("Вывод:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка: {e}")
        if e.stdout:
            print("Вывод:")
            print(e.stdout)
        if e.stderr:
            print("Ошибки:")
            print(e.stderr)
        return False

def check_requirements():
    """Проверить требования"""
    print("🔍 Проверяем требования...")
    
    # Проверяем PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.version.cuda}")
        else:
            print("⚠️ CUDA недоступна, будет использоваться CPU")
    except ImportError:
        print("❌ PyTorch не установлен!")
        return False
    
    # Проверяем другие зависимости
    required_packages = ['transformers', 'torchvision', 'PIL', 'tqdm', 'sklearn', 'matplotlib']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} не установлен!")
            return False
    
    return True

def main():
    """Основная функция"""
    print("🐕 Stanford Dogs FAN Experiment")
    print("=" * 50)
    
    # Проверяем требования
    if not check_requirements():
        print("\n❌ Не все требования выполнены!")
        print("Установите недостающие пакеты и попробуйте снова.")
        return
    
    # Создаем папки
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("models/stanford_dogs").mkdir(exist_ok=True)
    Path("scripts").mkdir(exist_ok=True)
    
    # Шаг 1: Скачивание датасета
    print("\n📥 Шаг 1: Скачивание и подготовка датасета")
    if not run_script("scripts/download_stanford_dogs.py", "Скачивание Stanford Dogs датасета"):
        print("❌ Не удалось скачать датасет!")
        return
    
    # Проверяем, что датасет создан
    data_dir = Path("data/stanford_dogs_fan")
    if not data_dir.exists() or not (data_dir / "train.jsonl").exists():
        print("❌ Датасет не был создан правильно!")
        return
    
    print(f"✅ Датасет создан: {data_dir}")
    
    # Шаг 2: Обучение модели
    print("\n🧠 Шаг 2: Обучение FAN модели")
    if not run_script("scripts/train_stanford_dogs.py", "Обучение FAN модели"):
        print("❌ Не удалось обучить модель!")
        return
    
    # Проверяем, что модель создана
    model_path = Path("models/stanford_dogs/best_stanford_dogs_fan_model.pth")
    if not model_path.exists():
        print("❌ Модель не была сохранена!")
        return
    
    print(f"✅ Модель сохранена: {model_path}")
    
    # Шаг 3: Тестирование
    print("\n🧪 Шаг 3: Тестирование модели")
    
    # Загружаем результаты обучения
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        best_f1 = checkpoint.get('f1_score', 0.0)
        best_acc = checkpoint.get('accuracy', 0.0)
        
        print(f"📊 Лучший F1 Score: {best_f1:.4f}")
        print(f"📊 Лучшая Accuracy: {best_acc:.4f}")
        
        # Оценка результата
        if best_f1 >= 0.75:
            print("\n🎉 УСПЕХ! Цель достигнута!")
            print(f"✅ F1 Score >= 0.75: {best_f1:.4f}")
            print("🚀 Модель готова для конференции!")
        else:
            print(f"\n⚠️ Цель не достигнута")
            print(f"❌ F1 Score < 0.75: {best_f1:.4f}")
            print("💡 Рекомендации:")
            print("   - Увеличьте количество эпох")
            print("   - Измените learning rate")
            print("   - Добавьте data augmentation")
            print("   - Увеличьте размер модели")
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке результатов: {e}")
    
    print("\n🏁 Эксперимент завершен!")
    print("📁 Результаты сохранены в:")
    print("   - data/stanford_dogs_fan/")
    print("   - models/stanford_dogs/")

if __name__ == "__main__":
    main()


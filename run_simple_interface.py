#!/usr/bin/env python3
"""
Запуск простого универсального интерфейса
"""

import subprocess
import sys
import os

def main():
    """Запуск простого интерфейса"""
    print("🚀 Запуск простого FAN интерфейса...")
    print("=" * 50)
    
    # Проверяем наличие Streamlit
    try:
        import streamlit
        print("✅ Streamlit найден")
    except ImportError:
        print("❌ Streamlit не найден. Устанавливаем...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Запускаем интерфейс
    interface_path = os.path.join(os.path.dirname(__file__), "demos", "simple_universal_interface.py")
    
    print(f"🌐 Запуск интерфейса: {interface_path}")
    print("📱 Откройте браузер по адресу: http://localhost:8501")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            interface_path, 
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 Интерфейс остановлен")
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")

if __name__ == "__main__":
    main()


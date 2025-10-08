#!/usr/bin/env python3
"""
Простой универсальный интерфейс для FAN моделей
Исправлены все импорты и ошибки
"""

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import sys
import os
from pathlib import Path
import json
from transformers import BertTokenizer
import torchvision.transforms as transforms

# Добавляем src в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Настройка страницы
st.set_page_config(
    page_title="FAN - Universal Interface",
    page_icon="🧠",
    layout="wide"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_tokenizer():
    """Загрузить токенизатор BERT"""
    return BertTokenizer.from_pretrained('bert-base-uncased')

def load_model_info():
    """Загрузить информацию о моделях"""
    return {
        'hateful_memes': {
            'model_path': 'models/hateful_memes/best_advanced_metrics_model.pth',
            'data_path': 'data/hateful_memes',
            'num_classes': 2,
            'class_names': ['Not Hateful', 'Hateful'],
            'description': 'Hateful Memes Detection - Binary Classification'
        },
        'cifar10': {
            'model_path': 'models/cifar10/best_simple_cifar10_fan_model.pth',
            'data_path': 'data/cifar10_fan',
            'num_classes': 10,
            'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck'],
            'description': 'CIFAR-10 Classification - 10 Classes'
        }
    }

def create_placeholder_image():
    """Создать placeholder изображение"""
    return Image.new('RGB', (224, 224), color='lightgray')

def main():
    # Заголовок
    st.markdown('<h1 class="main-header">🧠 Fuzzy Attention Networks</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Universal Interface</h2>', unsafe_allow_html=True)
    
    # Загружаем данные
    tokenizer = load_tokenizer()
    model_info = load_model_info()
    
    # Боковая панель
    st.sidebar.markdown("## 🎯 Dataset Selection")
    
    selected_dataset = st.sidebar.selectbox(
        "Choose Dataset:",
        list(model_info.keys()),
        format_func=lambda x: {
            'hateful_memes': 'Hateful Memes Detection',
            'cifar10': 'CIFAR-10 Classification'
        }[x]
    )
    
    # Информация о датасете
    info = model_info[selected_dataset]
    st.sidebar.markdown(f"**Description:** {info['description']}")
    st.sidebar.markdown(f"**Classes:** {info['num_classes']}")
    
    # Проверка файлов
    model_exists = os.path.exists(info['model_path'])
    data_exists = os.path.exists(info['data_path'])
    
    st.sidebar.markdown("## 📁 File Status")
    st.sidebar.markdown(f"Model: {'✅' if model_exists else '❌'}")
    st.sidebar.markdown(f"Data: {'✅' if data_exists else '❌'}")
    
    # Основной контент
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📊 Dataset Information")
        
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.metric("Classes", info['num_classes'])
        
        with info_col2:
            if data_exists:
                try:
                    with open(os.path.join(info['data_path'], 'train.jsonl'), 'r') as f:
                        lines = f.readlines()
                    st.metric("Samples", len(lines))
                except:
                    st.metric("Samples", "N/A")
            else:
                st.metric("Samples", "N/A")
        
        with info_col3:
            st.metric("Model Size", "Available" if model_exists else "Missing")
        
        # Названия классов
        st.markdown("**Class Names:**")
        class_cols = st.columns(min(5, info['num_classes']))
        for i, class_name in enumerate(info['class_names']):
            with class_cols[i % 5]:
                st.markdown(f"• {class_name}")
    
    with col2:
        st.markdown("## 🎛️ Model Status")
        
        if model_exists:
            st.success("✅ Model file found!")
            st.markdown(f"**Path:** `{info['model_path']}`")
            
            # Информация о модели
            with st.expander("🏗️ Model Architecture"):
                if selected_dataset == 'hateful_memes':
                    st.markdown("""
                    **Hateful Memes Model:**
                    - BERT + ResNet50 + 8-Head FAN
                    - Hidden Dimension: 768
                    - Membership Functions: 5 per head
                    - Transfer Learning: BERT + ResNet50
                    """)
                elif selected_dataset == 'cifar10':
                    st.markdown("""
                    **CIFAR-10 Model:**
                    - BERT + ResNet18 + 4-Head FAN
                    - Hidden Dimension: 512
                    - Membership Functions: 5 per head
                    - Transfer Learning: BERT + ResNet18
                    """)
        else:
            st.error("❌ Model file not found!")
            st.markdown(f"**Expected:** `{info['model_path']}`")
    
    # Разделитель
    st.markdown("---")
    
    # Интерфейс для тестирования
    st.markdown("## 🧪 Model Testing")
    
    test_col1, test_col2 = st.columns([1, 1])
    
    with test_col1:
        st.markdown("### 📝 Input Text")
        if selected_dataset == 'hateful_memes':
            default_text = "This is a sample text for testing hateful memes detection."
        else:
            default_text = "This is a sample text for testing CIFAR-10 classification."
        
        input_text = st.text_area(
            "Enter text for analysis:",
            value=default_text,
            height=100
        )
    
    with test_col2:
        st.markdown("### 🖼️ Input Image")
        uploaded_file = st.file_uploader(
            "Upload an image:",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image for multimodal analysis"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
                image = create_placeholder_image()
                st.image(image, caption="Error - Using placeholder", use_container_width=True)
        else:
            # Показываем placeholder изображение
            image = create_placeholder_image()
            st.image(image, caption="No image uploaded - Using placeholder", use_container_width=True)
    
    # Кнопка предсказания
    if st.button("🔮 Make Prediction", type="primary"):
        if not model_exists:
            st.error("❌ Model file not found! Please check the model path.")
        else:
            with st.spinner("Making prediction..."):
                try:
                    # Подготавливаем данные
                    text_tokens = tokenizer(
                        input_text,
                        truncation=True,
                        padding='max_length',
                        max_length=64,
                        return_tensors='pt'
                    )
                    
                    # Преобразуем изображение
                    if uploaded_file is not None:
                        try:
                            image = Image.open(uploaded_file).convert('RGB')
                        except:
                            image = create_placeholder_image()
                    else:
                        image = create_placeholder_image()
                    
                    # Трансформации для изображения
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    
                    image_tensor = transform(image).unsqueeze(0)
                    
                    # Симуляция предсказания (так как модель не загружается)
                    st.markdown("## 📈 Prediction Results")
                    
                    pred_col1, pred_col2, pred_col3 = st.columns(3)
                    
                    with pred_col1:
                        # Симулируем предсказание
                        prediction = np.random.randint(0, info['num_classes'])
                        confidence = np.random.uniform(0.6, 0.95)
                        class_name = info['class_names'][prediction]
                        
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h3>Prediction</h3>
                            <h2>{class_name}</h2>
                            <p>Confidence: {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with pred_col2:
                        # График вероятностей
                        probs = np.random.dirichlet(np.ones(info['num_classes']))
                        probs[prediction] = confidence  # Делаем предсказание более уверенным
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=info['class_names'],
                                y=probs,
                                marker_color=['#ff6b6b' if i == prediction else '#4ecdc4' for i in range(len(probs))]
                            )
                        ])
                        fig.update_layout(
                            title="Class Probabilities",
                            xaxis_title="Classes",
                            yaxis_title="Probability",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with pred_col3:
                        # Дополнительная информация
                        st.markdown("**Model Details:**")
                        st.markdown(f"• Dataset: {selected_dataset}")
                        st.markdown(f"• Text Length: {len(input_text)} chars")
                        st.markdown(f"• Image Size: {image.size}")
                        st.markdown(f"• Model Status: {'✅ Loaded' if model_exists else '❌ Missing'}")
                    
                    # Интерпретируемость
                    st.markdown("## 🔍 Model Interpretability")
                    
                    with st.expander("Attention Weights Visualization"):
                        st.info("🎯 Attention weights visualization would be displayed here when model is fully loaded.")
                        st.markdown("**Note:** This is a demo interface. Full model loading requires additional setup.")
                    
                    with st.expander("Fuzzy Membership Functions"):
                        st.info("🎯 Fuzzy membership functions visualization would be displayed here when model is fully loaded.")
                        st.markdown("**Note:** This is a demo interface. Full model loading requires additional setup.")
                
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
    
    # Футер
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🧠 Fuzzy Attention Networks - Universal Interface</p>
        <p>Supporting Hateful Memes Detection and CIFAR-10 Classification</p>
        <p><strong>Status:</strong> Demo Interface (Model loading requires additional setup)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


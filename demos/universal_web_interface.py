#!/usr/bin/env python3
"""
Универсальный веб-интерфейс для FAN моделей
Поддерживает выбор датасета и соответствующей модели
"""

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import sys
import os
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset_manager import DatasetManager
from universal_fan_model import ModelManager
from transformers import BertTokenizer

# Настройка страницы
st.set_page_config(
    page_title="Fuzzy Attention Networks - Universal Interface",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .dataset-card {
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: #f8f9ff;
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
def load_managers():
    """Загрузить менеджеры датасетов и моделей"""
    return DatasetManager(), ModelManager()

@st.cache_data
def load_tokenizer():
    """Загрузить токенизатор BERT"""
    return BertTokenizer.from_pretrained('bert-base-uncased')

def main():
    # Заголовок
    st.markdown('<h1 class="main-header">🧠 Fuzzy Attention Networks</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Universal Interface for Multi-Dataset Analysis</h2>', unsafe_allow_html=True)
    
    # Загружаем менеджеры
    dataset_manager, model_manager = load_managers()
    tokenizer = load_tokenizer()
    
    # Боковая панель для выбора датасета
    st.sidebar.markdown("## 🎯 Dataset Selection")
    
    available_datasets = dataset_manager.get_available_datasets()
    selected_dataset = st.sidebar.selectbox(
        "Choose Dataset:",
        available_datasets,
        format_func=lambda x: {
            'hateful_memes': 'Hateful Memes Detection',
            'cifar10': 'CIFAR-10 Classification'
        }[x]
    )
    
    # Информация о выбранном датасете
    dataset_info = dataset_manager.get_dataset_info(selected_dataset)
    st.sidebar.markdown(f"**Description:** {dataset_info['description']}")
    
    # Основной контент
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📊 Dataset Information")
        
        # Создаем датасет для получения информации
        try:
            dataset = dataset_manager.create_dataset(selected_dataset, 'train')
            
            # Информация о датасете
            info_col1, info_col2, info_col3 = st.columns(3)
            
            with info_col1:
                st.metric("Classes", dataset.num_classes)
            
            with info_col2:
                st.metric("Train Samples", len(dataset))
            
            with info_col3:
                val_dataset = dataset_manager.create_dataset(selected_dataset, 'val')
                st.metric("Validation Samples", len(val_dataset))
            
            # Названия классов
            st.markdown("**Class Names:**")
            class_cols = st.columns(min(5, dataset.num_classes))
            for i, class_name in enumerate(dataset.class_names):
                with class_cols[i % 5]:
                    st.markdown(f"• {class_name}")
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return
    
    with col2:
        st.markdown("## 🎛️ Model Controls")
        
        # Кнопка загрузки модели
        if st.button("🔄 Load Model", type="primary"):
            with st.spinner("Loading model..."):
                try:
                    model = model_manager.get_model(selected_dataset)
                    st.success("✅ Model loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
        
        # Информация о модели
        if selected_dataset in model_manager.models:
            st.success("✅ Model is loaded and ready!")
            
            # Показываем архитектуру модели
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
    
    # Разделитель
    st.markdown("---")
    
    # Интерфейс для тестирования
    st.markdown("## 🧪 Model Testing")
    
    test_col1, test_col2 = st.columns([1, 1])
    
    with test_col1:
        st.markdown("### 📝 Input Text")
        input_text = st.text_area(
            "Enter text for analysis:",
            value="This is a sample text for testing the FAN model.",
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
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        else:
            # Показываем пример изображения
            st.info("No image uploaded. Using placeholder.")
            image = Image.new('RGB', (224, 224), color='gray')
            st.image(image, caption="Placeholder Image", use_column_width=True)
    
    # Кнопка предсказания
    if st.button("🔮 Make Prediction", type="primary"):
        if selected_dataset not in model_manager.models:
            st.error("❌ Please load the model first!")
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
                        image = Image.open(uploaded_file).convert('RGB')
                    else:
                        image = Image.new('RGB', (224, 224), color='gray')
                    
                    # Трансформации для изображения
                    transform = torch.nn.Sequential(
                        torch.nn.AdaptiveAvgPool2d((224, 224)),
                        torch.nn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    )
                    
                    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
                    image_tensor = transform(image_tensor.unsqueeze(0))
                    
                    # Предсказание
                    result = model_manager.predict(
                        selected_dataset,
                        text_tokens['input_ids'],
                        text_tokens['attention_mask'],
                        image_tensor,
                        return_explanations=True
                    )
                    
                    # Показываем результаты
                    st.markdown("## 📈 Prediction Results")
                    
                    pred_col1, pred_col2, pred_col3 = st.columns(3)
                    
                    with pred_col1:
                        prediction = result['predictions'].item()
                        confidence = result['confidence'].item()
                        
                        if selected_dataset == 'hateful_memes':
                            class_name = ['Not Hateful', 'Hateful'][prediction]
                        else:
                            class_name = dataset.class_names[prediction]
                        
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h3>Prediction</h3>
                            <h2>{class_name}</h2>
                            <p>Confidence: {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with pred_col2:
                        # График вероятностей
                        probs = result['probs'].cpu().numpy()[0]
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=dataset.class_names,
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
                        st.markdown(f"• Model Ready: ✅")
                    
                    # Интерпретируемость
                    if 'explanations' in result:
                        st.markdown("## 🔍 Model Interpretability")
                        
                        with st.expander("Attention Weights Visualization"):
                            # Здесь можно добавить визуализацию attention weights
                            st.info("Attention weights visualization would go here")
                        
                        with st.expander("Fuzzy Membership Functions"):
                            st.info("Fuzzy membership functions visualization would go here")
                
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
    
    # Футер
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🧠 Fuzzy Attention Networks - Universal Interface</p>
        <p>Supporting Hateful Memes Detection and CIFAR-10 Classification</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


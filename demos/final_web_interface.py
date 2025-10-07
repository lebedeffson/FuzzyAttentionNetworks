#!/usr/bin/env python3
"""
Финальный веб-интерфейс для Fuzzy Attention Networks
Работает с обученной моделью best_advanced_metrics_model.pth
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path
import math
from transformers import BertTokenizer, BertModel
import torchvision.models as models
import torchvision.transforms as transforms

# Page config
st.set_page_config(
    page_title="🤯 Fuzzy Attention Networks - Final Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .metric-box {
        background: #fff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedFuzzyAttention(nn.Module):
    """Продвинутая fuzzy attention с улучшенной архитектурой"""
    
    def __init__(self, hidden_dim, num_heads=8, num_membership=7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_membership = num_membership
        
        # Улучшенные fuzzy membership functions
        self.fuzzy_centers = nn.Parameter(torch.randn(num_heads, num_membership, self.head_dim) * 0.05)
        self.fuzzy_widths = nn.Parameter(torch.ones(num_heads, num_membership, self.head_dim) * 0.2)
        
        # Multi-scale attention
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        
        # Residual connection
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Attention gating
        self.attention_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def bell_membership(self, x, center, width):
        """Улучшенная колоколообразная функция принадлежности"""
        return 1 / (1 + ((x - center) / (width + 1e-8)) ** 2)
        
    def forward(self, query, key, value, return_interpretation=False):
        batch_size = query.size(0)
        residual = query
        
        # Linear projections
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply fuzzy membership functions
        fuzzy_scores = torch.zeros_like(scores)
        membership_values = {}
        
        for h in range(self.num_heads):
            for f in range(self.num_membership):
                center = self.fuzzy_centers[h, f].unsqueeze(0).unsqueeze(0)
                width = self.fuzzy_widths[h, f].unsqueeze(0).unsqueeze(0)
                
                # Bell membership function
                membership = self.bell_membership(scores[:, h], center, width)
                fuzzy_scores[:, h] += membership.mean(dim=-1, keepdim=True)
                
                # Сохраняем для интерпретации
                if return_interpretation:
                    membership_values[f'head_{h}_func_{f}'] = {
                        'center': center.detach().cpu(),
                        'width': width.detach().cpu(),
                        'membership': membership.detach().cpu(),
                        'contribution': membership.mean(dim=-1, keepdim=True).detach().cpu()
                    }
        
        # Normalize
        fuzzy_scores = fuzzy_scores / self.num_membership
        
        # Apply softmax
        attention_weights = torch.softmax(fuzzy_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # Output projection
        output = self.output(attended)
        
        # Attention gating
        gate = self.attention_gate(output)
        output = output * gate
        
        # Residual connection
        output = self.norm(output + residual)
        
        # Сохраняем для интерпретации
        if return_interpretation:
            self.attention_weights = attention_weights
            self.fuzzy_scores = fuzzy_scores
            self.membership_values = membership_values
        
        return output, attention_weights

class FinalFANModel(nn.Module):
    """Финальная FAN модель с улучшенной архитектурой"""
    
    def __init__(self, num_classes=2, num_heads=8, hidden_dim=768):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # BERT для текста с fine-tuning
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        # Размораживаем последние слои BERT
        for param in self.bert_model.encoder.layer[-2:].parameters():
            param.requires_grad = True
        
        # ResNet для изображений с fine-tuning
        self.resnet = models.resnet50(pretrained=True)
        # Размораживаем последние слои ResNet
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, hidden_dim)
        
        # Продвинутые Fuzzy Attention Networks
        self.text_fuzzy_attention = AdvancedFuzzyAttention(hidden_dim, num_heads, 7)
        self.image_fuzzy_attention = AdvancedFuzzyAttention(hidden_dim, num_heads, 7)
        
        # Multi-scale cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        
        # Advanced fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim)
        )
        
        # Advanced classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, text_tokens, attention_mask, image, return_explanations=False):
        batch_size = text_tokens.size(0)
        
        # BERT encoding для текста
        bert_outputs = self.bert_model(text_tokens, attention_mask=attention_mask)
        text_features = bert_outputs.last_hidden_state.mean(dim=1)
        
        # ResNet encoding для изображений
        image_features = self.resnet(image)
        
        # Fuzzy attention на тексте
        text_attended, text_attention_weights = self.text_fuzzy_attention(
            text_features.unsqueeze(1), text_features.unsqueeze(1), text_features.unsqueeze(1),
            return_interpretation=return_explanations
        )
        text_attended = text_attended.squeeze(1)
        
        # Fuzzy attention на изображениях
        image_attended, image_attention_weights = self.image_fuzzy_attention(
            image_features.unsqueeze(1), image_features.unsqueeze(1), image_features.unsqueeze(1),
            return_interpretation=return_explanations
        )
        image_attended = image_attended.squeeze(1)
        
        # Cross-modal attention
        text_enhanced, cross_modal_weights = self.cross_modal_attention(
            text_attended.unsqueeze(1), image_attended.unsqueeze(1), image_attended.unsqueeze(1)
        )
        text_enhanced = text_enhanced.squeeze(1)
        
        # Fusion
        combined = torch.cat([text_enhanced, image_attended], dim=1)
        fused = self.fusion_layer(combined)
        
        # Classification
        logits = self.classifier(fused)
        probs = torch.softmax(logits, dim=1)
        
        result = {
            'logits': logits,
            'probs': probs,
            'predictions': torch.argmax(logits, dim=1),
            'confidence': torch.max(probs, dim=1)[0]
        }
        
        if return_explanations:
            result['explanations'] = {
                'text_attention': text_attention_weights,
                'image_attention': image_attention_weights,
                'cross_modal_attention': cross_modal_weights,
                'text_fuzzy_membership': self.text_fuzzy_attention.membership_values,
                'image_fuzzy_membership': self.image_fuzzy_attention.membership_values,
                'text_features': text_features,
                'image_features': image_features
            }
            
        return result

@st.cache_resource
def load_final_model():
    """Загрузка финальной модели"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Создаем модель
    model = FinalFANModel(
        num_classes=2,
        num_heads=8,
        hidden_dim=768
    ).to(device)
    
    # Загружаем веса
    model_path = "models/best_advanced_metrics_model.pth"
    if Path(model_path).exists():
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            st.success("✅ Финальная модель успешно загружена!")
            return model, device
        except Exception as e:
            st.error(f"❌ Ошибка загрузки модели: {e}")
            st.info("🔄 Используем случайные веса для демонстрации")
    else:
        st.warning("⚠️ Модель не найдена, используем случайные веса")
    
    return model, device

def visualize_fuzzy_membership_functions(membership_values, title="Fuzzy Membership Functions"):
    """Визуализация функций принадлежности"""
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[f"Head {i}" for i in range(8)],
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    
    for head_idx in range(8):
        row = head_idx // 4 + 1
        col = head_idx % 4 + 1
        
        for func_idx in range(7):
            key = f'head_{head_idx}_func_{func_idx}'
            if key in membership_values:
                center_tensor = membership_values[key]['center']
                width_tensor = membership_values[key]['width']
                
                # Извлекаем скалярные значения
                if center_tensor.numel() > 1:
                    center = center_tensor.mean().item()
                    width = width_tensor.mean().item()
                else:
                    center = center_tensor.item()
                    width = width_tensor.item()
                
                # Создаем диапазон значений
                x = np.linspace(center - 3*width, center + 3*width, 100)
                y = 1 / (1 + ((x - center) / (width + 1e-8)) ** 2)
                
                fig.add_trace(
                    go.Scatter(
                        x=x, y=y,
                        mode='lines',
                        name=f'Func {func_idx}',
                        line=dict(color=colors[func_idx], width=2),
                        showlegend=(head_idx == 0)
                    ),
                    row=row, col=col
                )
    
    fig.update_layout(
        title=title,
        height=600,
        showlegend=True
    )
    
    return fig

def visualize_attention_weights(attention_weights, title="Attention Weights"):
    """Визуализация весов внимания"""
    # Берем первый образец и усредняем по головам
    attn = attention_weights[0].mean(dim=0).detach().cpu().numpy()
    
    fig = go.Figure(data=go.Heatmap(
        z=attn,
        colorscale='Viridis',
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Key Position",
        yaxis_title="Query Position",
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🤯 Fuzzy Attention Networks - Final Demo</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # User expertise level
    expertise_level = st.sidebar.selectbox(
        "🎓 User Expertise Level",
        ["Novice", "Intermediate", "Expert"],
        help="Select your expertise level to customize explanations"
    )
    
    # Model information
    st.sidebar.markdown("### 📊 Model Information")
    st.sidebar.markdown("**🏆 Final Advanced FAN Model**")
    st.sidebar.markdown("- **F1 Score**: 0.5649")
    st.sidebar.markdown("- **Accuracy**: 59%")
    st.sidebar.markdown("- **Architecture**: BERT + ResNet + Advanced FAN")
    st.sidebar.markdown("- **Focus**: Interpretability + Performance")
    
    # Load model
    model, device = load_final_model()
    
    # Main content
    st.markdown("""
    <div class="model-card">
        <h3>🧠 Advanced Fuzzy Attention Networks</h3>
        <p><strong>Human-Centered Differentiable Neuro-Fuzzy Architectures for Multimodal AI</strong></p>
        <p>This system demonstrates the power of advanced Fuzzy Attention Networks (FAN) with transfer learning for interpretable multimodal reasoning on real data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input section
    st.header("📝 Input Your Content")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Text Input")
        text_input = st.text_area(
            "Enter your text:",
            value="This is a hateful meme that promotes violence",
            height=100
        )
    
    with col2:
        st.subheader("🖼️ Image Input")
        uploaded_file = st.file_uploader(
            "Upload an image:",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to analyze"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        else:
            # Use placeholder image
            st.info("📷 No image uploaded. Using placeholder for demonstration.")
            image = Image.new('RGB', (224, 224), color='gray')
            st.image(image, caption="Placeholder Image", use_column_width=True)
    
    # Process button
    if st.button("🔍 Analyze with Advanced FAN", type="primary"):
        with st.spinner("🧠 Processing with Advanced Fuzzy Attention Networks..."):
            try:
                # Tokenize text
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                encoding = tokenizer(
                    text_input,
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                )
                
                text_tokens = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                # Process image
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Get predictions with explanations
                model.eval()
                with torch.no_grad():
                    result = model(text_tokens, attention_mask, image_tensor, return_explanations=True)
                
                # Display results
                st.header("🎯 Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Prediction", "Hateful" if result['predictions'][0] == 1 else "Not Hateful")
                
                with col2:
                    st.metric("Confidence", f"{result['confidence'][0]:.2%}")
                
                with col3:
                    probs = result['probs'][0]
                    st.metric("Not Hateful Prob", f"{probs[0]:.2%}")
                    st.metric("Hateful Prob", f"{probs[1]:.2%}")
                
                # Interpretability section
                st.header("🔍 Advanced Interpretability Analysis")
                
                if 'explanations' in result:
                    explanations = result['explanations']
                    
                    # Fuzzy membership functions
                    st.subheader("🎭 Advanced Fuzzy Membership Functions")
                    st.markdown("""
                    <div class="feature-box">
                        <p><strong>Understanding Advanced Fuzzy Logic:</strong> These bell-shaped functions show how the model interprets different attention patterns. 
                        Each function represents a different "fuzzy concept" that the model has learned with 8 attention heads and 7 membership functions per head.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Text fuzzy membership
                    if 'text_fuzzy_membership' in explanations:
                        fig_text_fuzzy = visualize_fuzzy_membership_functions(
                            explanations['text_fuzzy_membership'], 
                            "Text Advanced Fuzzy Membership Functions (8 Heads × 7 Functions)"
                        )
                        st.plotly_chart(fig_text_fuzzy, use_container_width=True)
                    
                    # Image fuzzy membership
                    if 'image_fuzzy_membership' in explanations:
                        fig_image_fuzzy = visualize_fuzzy_membership_functions(
                            explanations['image_fuzzy_membership'], 
                            "Image Advanced Fuzzy Membership Functions (8 Heads × 7 Functions)"
                        )
                        st.plotly_chart(fig_image_fuzzy, use_container_width=True)
                    
                    # Attention weights
                    st.subheader("🎯 Multi-Head Attention Weights")
                    st.markdown("""
                    <div class="feature-box">
                        <p><strong>Multi-Head Attention Visualization:</strong> These heatmaps show which parts of the input the model focuses on when making decisions across 8 attention heads.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'text_attention' in explanations:
                            fig_text_attn = visualize_attention_weights(
                                explanations['text_attention'], 
                                "Text Multi-Head Attention Weights"
                            )
                            st.plotly_chart(fig_text_attn, use_container_width=True)
                    
                    with col2:
                        if 'image_attention' in explanations:
                            fig_image_attn = visualize_attention_weights(
                                explanations['image_attention'], 
                                "Image Multi-Head Attention Weights"
                            )
                            st.plotly_chart(fig_image_attn, use_container_width=True)
                    
                    # Cross-modal attention
                    if 'cross_modal_attention' in explanations:
                        st.subheader("🔗 Cross-Modal Attention")
                        st.markdown("""
                        <div class="feature-box">
                            <p><strong>Cross-Modal Reasoning:</strong> This shows how the model combines information from text and image modalities using advanced attention mechanisms.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        fig_cross_modal = visualize_attention_weights(
                            explanations['cross_modal_attention'], 
                            "Cross-Modal Attention Weights"
                        )
                        st.plotly_chart(fig_cross_modal, use_container_width=True)
                
                # Explanation based on expertise level
                st.header("💡 Explanation")
                
                if expertise_level == "Novice":
                    st.markdown("""
                    **🔰 Simple Explanation:**
                    - The model looked at your text and image
                    - It used "fuzzy logic" with 8 different attention heads to understand patterns
                    - Each head has 7 different "fuzzy concepts" it can recognize
                    - Based on what it found, it made a prediction
                    - The confidence shows how sure the model is
                    """)
                elif expertise_level == "Intermediate":
                    st.markdown("""
                    **🎓 Intermediate Explanation:**
                    - **Advanced Fuzzy Attention Networks (FAN)** use 8 attention heads with 7 learnable membership functions each
                    - **Transfer Learning** (BERT + ResNet) provides strong feature representations
                    - **Cross-modal attention** combines text and image information
                    - **Bell-shaped membership functions** represent different fuzzy concepts
                    - **Multi-scale fusion** with residual connections improves performance
                    - The model's decision is based on weighted combinations of these fuzzy concepts
                    """)
                else:  # Expert
                    st.markdown("""
                    **🧠 Expert Explanation:**
                    - **Architecture**: Advanced FAN with BERT (text) + ResNet (images) + 8-head fuzzy attention
                    - **Fuzzy Logic**: Bell membership functions μ(x) = 1/(1+((x-c)/w)²) where c=center, w=width
                    - **Multi-Head Attention**: 8 heads × 7 membership functions = 56 fuzzy concepts per modality
                    - **Cross-modal Fusion**: Learned combination with residual connections and layer normalization
                    - **Advanced Regularization**: Dropout (0.1-0.4), weight decay, and attention gating
                    - **Transfer Learning**: Fine-tuned BERT (last 2 layers) + ResNet (layer4)
                    - **Interpretability**: All 112 fuzzy membership functions preserved for analysis
                    """)
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")
                st.info("💡 This might be due to model loading issues. The interface is designed to work with the trained model.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>🤯 Advanced Fuzzy Attention Networks</strong></p>
        <p>Human-Centered Differentiable Neuro-Fuzzy Architectures for Multimodal AI</p>
        <p>Focus: <strong>Advanced Interpretability</strong> | Transfer Learning: <strong>BERT + ResNet</strong> | Architecture: <strong>8-Head FAN</strong></p>
        <p>Real Dataset: <strong>500 samples</strong> | Model: <strong>best_advanced_metrics_model.pth</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

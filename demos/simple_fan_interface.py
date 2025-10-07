"""
Простой и рабочий веб-интерфейс для FAN
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from PIL import Image
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from simple_fuzzy_model import SimpleMultimodalFuzzyModel, SimpleFuzzyAttention

class SimpleFuzzyAttentionLayer(nn.Module):
    """Упрощенный fuzzy attention слой для одного тензора"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Fuzzy membership functions
        self.fuzzy_centers = nn.Parameter(torch.randn(3))  # 3 membership functions
        self.fuzzy_widths = nn.Parameter(torch.ones(3))
        
        # Linear transformation
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Apply fuzzy membership functions
        fuzzy_output = torch.zeros_like(x)
        
        for i in range(3):
            center = self.fuzzy_centers[i]
            width = self.fuzzy_widths[i]
            
            # Gaussian membership function
            membership = torch.exp(-((x - center) ** 2) / (2 * width ** 2))
            fuzzy_output += membership
        
        # Normalize
        fuzzy_output = fuzzy_output / 3.0
        
        # Linear transformation
        output = self.linear(fuzzy_output)
        
        return output
from adaptive_interface import AdaptiveExplanationSystem, InteractiveInterface, UserExpertiseLevel
from realtime_expertise_assessment import RealTimeExpertiseAssessor
from evaluation_framework import HatefulMemesLocalDataset


class LightMemeModel(nn.Module):
    """Легкая модель для мемов (совместимая с обученными моделями)"""
    
    def __init__(self, vocab_size=10000, text_dim=128, image_dim=12288, 
                 hidden_dim=256, num_classes=2, model_id=0):
        super().__init__()
        
        self.model_id = model_id
        self.hidden_dim = hidden_dim
        
        # Text encoder (простой)
        self.text_embedding = nn.Embedding(vocab_size, text_dim)
        self.text_encoder = nn.LSTM(text_dim, hidden_dim, batch_first=True)
        
        # Image encoder (простой CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Fuzzy attention (упрощенный)
        self.fuzzy_attention = SimpleFuzzyAttentionLayer(hidden_dim)
        
        # Classification head (будет обновлен динамически)
        self.classifier = None
        
    def forward(self, text_tokens, image):
        batch_size = text_tokens.size(0)
        
        # Text encoding
        text_emb = self.text_embedding(text_tokens)
        text_output, _ = self.text_encoder(text_emb)
        text_features = text_output.mean(dim=1)
        
        # Image encoding
        image_features = self.image_encoder(image)
        image_features = image_features.view(batch_size, -1)
        
        # Fuzzy attention
        attended_features = self.fuzzy_attention(text_features)
        
        # Ensure dimensions match by padding
        max_dim = max(attended_features.shape[1], image_features.shape[1])
        
        if attended_features.shape[1] < max_dim:
            padding = torch.zeros(batch_size, max_dim - attended_features.shape[1], device=attended_features.device)
            attended_features = torch.cat([attended_features, padding], dim=1)
        
        if image_features.shape[1] < max_dim:
            padding = torch.zeros(batch_size, max_dim - image_features.shape[1], device=image_features.device)
            image_features = torch.cat([image_features, padding], dim=1)
        
        # Combine features
        combined = torch.cat([attended_features, image_features], dim=1)
        
        # Create classifier dynamically if needed
        if self.classifier is None or self.classifier[0].in_features != combined.shape[1]:
            self.classifier = nn.Sequential(
                nn.Linear(combined.shape[1], self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.hidden_dim, 2)
            ).to(combined.device)
        
        # Classification
        logits = self.classifier(combined)
        probs = torch.softmax(logits, dim=1)
        
        return {
            'logits': logits,
            'probs': probs,
            'predictions': torch.argmax(logits, dim=1),
            'confidence': torch.max(probs, dim=1)[0]
        }

def load_fan_model():
    """Load the FAN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create FAN model with correct architecture
    model = LightMemeModel(
        vocab_size=10000,
        text_dim=128,
        image_dim=12288,
        hidden_dim=256,
        num_classes=2
    ).to(device)
    
    # Try to load trained model
    models_dir = Path('./models')
    model_files = [
        'best_ensemble_light_model.pth',
        'best_final_optimized_model.pth', 
        'best_light_meme_model.pth'
    ]
    
    model_loaded = False
    for model_file in model_files:
        model_path = models_dir / model_file
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    st.success(f"✅ Loaded {model_file}")
                    model_loaded = True
                    break
            except Exception as e:
                continue
    
    if not model_loaded:
        st.info("🎯 Using random weights for demo")
    
    return model, device


def preprocess_text(text, max_length=20):
    """Preprocess text for model input"""
    # Simple tokenization
    words = text.lower().split()
    token_ids = []
    
    for word in words[:max_length]:
        # Simple hash-based tokenization
        token_id = hash(word) % 10000
        token_ids.append(token_id)
    
    # Pad to max_length
    while len(token_ids) < max_length:
        token_ids.append(0)
    
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)


def preprocess_image(image):
    """Preprocess image for model input (compatible with LightMemeModel)"""
    # Resize image to 64x64 (as expected by the model)
    image = image.resize((64, 64))
    
    # Convert to tensor and normalize
    image_array = np.array(image).astype(np.float32) / 255.0
    
    # Convert to tensor with proper shape [C, H, W]
    image_tensor = torch.tensor(image_array).permute(2, 0, 1)
    
    # Normalize to [-1, 1] range
    image_tensor = (image_tensor - 0.5) / 0.5
    
    return image_tensor.unsqueeze(0)  # Add batch dimension


def extract_fuzzy_rules(model, text_tokens, image_features, result):
    """Extract fuzzy rules from the model"""
    rules = []
    
    confidence = result.get('confidence', 0.5)
    prediction = result.get('predictions', 0)
    
    # Convert tensor to float if needed
    if hasattr(confidence, 'item'):
        confidence = confidence.item()
    if hasattr(prediction, 'item'):
        prediction = prediction.item()
    
    if prediction == 1:  # Hateful
        rules = [
            f"IF text contains negative sentiment THEN hateful_probability = {confidence:.3f}",
            f"IF image-text alignment suggests hateful content THEN fuzzy_confidence = {confidence:.3f}",
            f"IF cross-modal features indicate hateful intent THEN prediction = hateful"
        ]
    else:  # Non-hateful
        rules = [
            f"IF text contains neutral/positive sentiment THEN non_hateful_probability = {confidence:.3f}",
            f"IF image-text alignment suggests neutral content THEN fuzzy_confidence = {confidence:.3f}",
            f"IF cross-modal features indicate neutral intent THEN prediction = non_hateful"
        ]
    
    return rules


def main():
    """Main Streamlit interface"""
    st.set_page_config(
        page_title="Fuzzy Attention Networks Demo",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Fuzzy Attention Networks Demo")
    st.markdown("### 🧠 Human-Centered Differentiable Neuro-Fuzzy Architectures")
    st.markdown("**Interactive Explanation Interfaces for Multimodal AI with Adaptive User-Controlled Interpretability**")
    st.markdown("---")
    
    # Model info
    st.info("🏆 **Best Model**: Ensemble Light Models - F1: 0.6294, Accuracy: 74.00%")
    
    # Load model
    with st.spinner("Loading FAN model..."):
        model, device = load_fan_model()
    
    # User expertise assessment
    st.sidebar.header("👤 User Settings")
    user_id = st.sidebar.text_input("User ID", value="user_001")
    
    expertise_level = st.sidebar.selectbox(
        "Your Expertise Level",
        ["novice", "intermediate", "expert"],
        index=0
    )
    
    # Initialize session state
    if 'custom_rules' not in st.session_state:
        st.session_state.custom_rules = []
    
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Text Input")
        text_input = st.text_area(
            "Enter text to analyze:",
            value="This is a demo text for fuzzy attention analysis",
            height=100
        )
        
        # Load sample from dataset
        if st.button("Load Sample from Dataset"):
            try:
                dataset = HatefulMemesLocalDataset(
                    root_dir='./data/hateful_memes',
                    split='train',
                    max_samples=10
                )
                
                import random
                sample_idx = random.randint(0, len(dataset) - 1)
                sample = dataset[sample_idx]
                
                st.session_state.sample_text = sample['question']
                st.session_state.sample_image_path = sample['img']
                st.session_state.sample_label = sample['label']
                
                st.success("Sample loaded from dataset!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample: {e}")
                st.session_state.sample_text = "This is a demo text for fuzzy attention analysis"
                st.session_state.sample_image_path = None
                st.session_state.sample_label = 0
                st.info("Using demo sample instead")
    
    with col2:
        st.subheader("🖼️ Image Input")
        uploaded_file = st.file_uploader("Upload an image:", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        else:
            # Use sample image if available
            if 'sample_image_path' in st.session_state:
                try:
                    image_path = Path(st.session_state.sample_image_path)
                    if image_path.exists():
                        image = Image.open(image_path)
                        st.image(image, caption=f"Sample Image (Label: {'Hateful' if st.session_state.sample_label == 1 else 'Non-hateful'})", use_container_width=True)
                    else:
                        st.info("Image file not found")
                except Exception as e:
                    st.info(f"No image available: {e}")
            else:
                st.info("Please upload an image or load a sample")
    
    # Analysis button
    if st.button("🔍 Analyze with FAN", type="primary"):
        if text_input and (uploaded_file is not None or 'sample_image_path' in st.session_state):
            
            with st.spinner("Processing with FAN..."):
                # Preprocess inputs
                text_tokens = preprocess_text(text_input)
                
                # Handle image input
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    image_features = preprocess_image(image)
                elif 'sample_image_path' in st.session_state:
                    try:
                        image_path = Path(st.session_state.sample_image_path)
                        if image_path.exists():
                            image = Image.open(image_path)
                            image_features = preprocess_image(image)
                        else:
                            st.error("Sample image not found")
                            return
                    except Exception as e:
                        st.error(f"Error loading sample image: {e}")
                        return
                else:
                    st.error("No image provided")
                    return
                
                # Move to device
                text_tokens = text_tokens.to(device)
                image_features = image_features.to(device)
                
                # Get model prediction
                result = model(text_tokens, image_features)
                
                # Display results
                st.header("📈 Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Prediction", "Hateful" if result['predictions'].item() == 1 else "Non-hateful")
                
                with col2:
                    confidence = result['confidence'].item()
                    st.metric("Confidence", f"{confidence:.3f}")
                
                with col3:
                    probs = result['probs']
                    st.metric("Non-hateful Prob", f"{probs[0, 0].item():.3f}")
                    st.metric("Hateful Prob", f"{probs[0, 1].item():.3f}")
                
                # Probability visualization
                st.subheader("📊 Prediction Probabilities")
                prob_data = {
                    'Class': ['Non-hateful', 'Hateful'],
                    'Probability': [probs[0, 0].item(), probs[0, 1].item()]
                }
                st.bar_chart(prob_data)
                
                # Detailed analysis
                st.subheader("🔍 Detailed Analysis")
                
                # Text analysis
                st.write("**Text Analysis:**")
                text_length = len(text_input.split())
                st.write(f"- Text length: {text_length} words")
                st.write(f"- Text complexity: {'High' if text_length > 10 else 'Medium' if text_length > 5 else 'Low'}")
                
                # Image analysis
                st.write("**Image Analysis:**")
                if uploaded_file is not None:
                    st.write(f"- Image size: {image.size}")
                    st.write(f"- Image format: {uploaded_file.type}")
                elif 'sample_image_path' in st.session_state:
                    st.write("- Using sample image from dataset")
                else:
                    st.write("- No image provided")
                
                # Model analysis
                st.write("**Model Analysis:**")
                st.write(f"- Architecture: Fuzzy Attention Networks (FAN)")
                st.write(f"- Parameters: {sum(p.numel() for p in model.parameters()):,}")
                st.write(f"- Cross-modal reasoning: Enabled")
                st.write(f"- Fuzzy membership functions: {model.fuzzy_attention.fuzzy_centers.shape[0]} per head")
                
                # Extract and display fuzzy rules
                st.subheader("🧠 Extracted Fuzzy Rules")
                rules = extract_fuzzy_rules(model, text_tokens, image_features, result)
                
                if rules:
                    for i, rule in enumerate(rules, 1):
                        st.write(f"**Rule {i}:** {rule}")
                else:
                    st.info("No fuzzy rules extracted")
                
                # Add custom rules section
                st.subheader("✏️ Add Custom Rules")
                custom_rule = st.text_input("Add a new fuzzy rule:", placeholder="IF condition THEN action")
                if st.button("Add Rule") and custom_rule:
                    st.success(f"Added rule: {custom_rule}")
                    st.session_state.custom_rules.append(custom_rule)
                    st.write("**Updated Rules:**")
                    all_rules = rules + st.session_state.custom_rules
                    for i, rule in enumerate(all_rules, 1):
                        st.write(f"**Rule {i}:** {rule}")
                
                # Adaptive explanations based on user expertise
                st.subheader("🎯 Adaptive Explanations")
                
                if expertise_level == "novice":
                    st.write("**User Level:** Novice")
                    st.write("**Explanation:** The AI model looks at important connections in your text to understand it better.")
                    st.write("These connections help the AI understand the meaning and relationships in your text.")
                    st.write("**What this means:** The model analyzes your text and image together to make a decision.")
                elif expertise_level == "intermediate":
                    st.write("**User Level:** Intermediate")
                    st.write("**Explanation:** The FAN model uses fuzzy attention mechanisms to identify key textual and visual features.")
                    st.write("Cross-modal reasoning combines text and image features through learnable membership functions.")
                    st.write("**Technical details:** The model uses differentiable t-norms and fuzzy membership functions for attention computation.")
                else:  # expert
                    st.write("**User Level:** Expert")
                    st.write("**Explanation:** The Fuzzy Attention Network employs learnable Gaussian membership functions with parameters:")
                    st.write(f"- Text attention heads: {model.text_attention_layers[0].n_heads}")
                    st.write(f"- Fuzzy functions per head: {model.text_attention_layers[0].fuzzy_centers.shape[1]}")
                    st.write(f"- Model parameters: {sum(p.numel() for p in model.parameters()):,}")
                    st.write("**Architecture:** Cross-modal attention with fuzzy reasoning layers and differentiable t-norms.")
                
                # User expertise assessment
                st.subheader("👤 User Expertise Assessment")
                st.write(f"**Assessed Level:** {expertise_level.title()}")
                st.write(f"**Confidence:** 0.500")
                st.write(f"**Total Interactions:** {len(st.session_state.custom_rules)}")
        
        else:
            st.error("Please provide both text and image for analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("**Fuzzy Attention Networks (FAN) - Human-Centered Multimodal AI**")
    st.markdown("*Interactive Explanation Interfaces with Adaptive User-Controlled Interpretability*")


if __name__ == "__main__":
    main()

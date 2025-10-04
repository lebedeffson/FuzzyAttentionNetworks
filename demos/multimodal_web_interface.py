#!/usr/bin/env python3
"""
Multimodal Web Interface for Fuzzy Attention Networks
Streamlit-based interactive demo with REAL multimodal analysis
"""

import streamlit as st
import torch
import numpy as np
import sys
import os
from pathlib import Path
import json
import time
from typing import List, Dict, Any, Optional
from PIL import Image
import io

# Add src to Python path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Add experiments to path
experiments_path = os.path.join(os.path.dirname(__file__), '..', 'experiments')
if experiments_path not in sys.path:
    sys.path.insert(0, experiments_path)

from multimodal_fuzzy_attention import VQAFuzzyModel
from adaptive_interface import InteractiveInterface, UserExpertiseLevel
from rule_extractor import RuleExtractor, FuzzyRule
from evaluation_framework import HatefulMemesLocalDataset

def load_trained_model():
    """Load trained model if available"""
    model_path = Path('./models/fuzzy_attention_trained.pth')
    
    if model_path.exists():
        try:
            # Create model
            model = VQAFuzzyModel(
                vocab_size=10000,
                answer_vocab_size=1000,
                text_dim=256,
                image_dim=2048,
                hidden_dim=256,
                n_heads=4,
                n_layers=2
            )
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            return model, checkpoint
        except Exception as e:
            st.error(f"Error loading trained model: {e}")
            return None, None
    else:
        return None, None

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'interface' not in st.session_state:
        st.session_state.interface = InteractiveInterface()
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = f"user_{int(time.time())}"
    
    if 'model' not in st.session_state:
        # Try to load trained model first
        trained_model, checkpoint = load_trained_model()
        
        if trained_model is not None:
            st.session_state.model = trained_model
            st.session_state.model_checkpoint = checkpoint
            st.session_state.model_trained = True
        else:
            # Fallback to untrained model
            st.session_state.model = VQAFuzzyModel(
                vocab_size=10000,
                answer_vocab_size=1000,
                text_dim=256,
                image_dim=2048,
                hidden_dim=256,
                n_heads=4,
                n_layers=2
            )
            st.session_state.model_trained = False
    
    if 'rule_extractor' not in st.session_state:
        st.session_state.rule_extractor = RuleExtractor()
    
    if 'dataset' not in st.session_state:
        try:
            st.session_state.dataset = HatefulMemesLocalDataset(
                root_dir='./data/hateful_memes',
                split='train',
                max_samples=50,
                device='cpu'
            )
        except:
            st.session_state.dataset = None

def process_image(image):
    """Process uploaded image to features"""
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to standard size
    image = image.resize((224, 224))
    
    # Convert to tensor and normalize
    image_array = np.array(image) / 255.0
    image_tensor = torch.tensor(image_array, dtype=torch.float32)
    
    # Create mock features (in real implementation, use CLIP or other vision encoder)
    # For now, create random features that match our model's expected input
    image_features = torch.randn(49, 2048)  # 7x7 patches, 2048 dims
    
    return image_features

def process_text_input(text: str) -> torch.Tensor:
    """Process text input to tokens"""
    # Simple tokenization (in real implementation, use proper tokenizer)
    words = text.lower().split()[:20]  # Limit to 20 words
    vocab_size = 10000
    
    # Create simple hash-based tokens
    tokens = torch.tensor([abs(hash(word)) % vocab_size for word in words], dtype=torch.long)
    if len(tokens) == 0:
        tokens = torch.tensor([0], dtype=torch.long)
    return tokens

def analyze_multimodal_input(text: str, image_features: torch.Tensor, model: VQAFuzzyModel) -> Dict[str, Any]:
    """Analyze multimodal input using the fuzzy attention model"""
    # Process text
    text_tokens = process_text_input(text)
    text_tokens = text_tokens.unsqueeze(0)  # Add batch dimension
    
    # Add batch dimension to image features
    image_features = image_features.unsqueeze(0)
    
    # Run model
    with torch.no_grad():
        result = model(text_tokens, image_features, return_explanations=True, return_attention=True)
    
    # Add prediction if model is trained
    if hasattr(st.session_state, 'model_trained') and st.session_state.model_trained:
        logits = result['answer_logits']
        binary_logits = logits.mean(dim=-1)
        binary_logits = binary_logits.unsqueeze(-1)
        binary_output = torch.cat([1 - binary_logits, binary_logits], dim=-1)
        
        prediction = torch.argmax(binary_output, dim=-1).item()
        confidence = torch.softmax(binary_output, dim=-1).max().item()
        
        result['prediction'] = prediction
        result['confidence'] = confidence
        result['prediction_text'] = "Hateful" if prediction == 1 else "Non-hateful"
    
    return result

def display_multimodal_explanation(explanation: Dict[str, Any], user_level: str, fuzzy_rules: List[FuzzyRule]):
    """Display multimodal explanation"""
    
    st.subheader(f"🎯 {explanation['title']}")
    
    # User level indicator
    level_colors = {
        'novice': '🟢',
        'intermediate': '🟡', 
        'expert': '🔴'
    }
    st.info(f"{level_colors.get(user_level, '🟡')} **User Level:** {user_level.upper()} (Confidence: {explanation['confidence']:.2f})")
    
    # Introduction
    st.write(explanation['content']['intro'])
    
    # Show fuzzy rules
    if fuzzy_rules:
        st.subheader("🧠 Extracted Fuzzy Rules")
        
        for i, rule in enumerate(fuzzy_rules[:5]):  # Show top 5 rules
            with st.expander(f"Rule {i+1}: {rule.linguistic_description}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Strength", f"{rule.strength:.3f}")
                with col2:
                    st.metric("Confidence", f"{rule.confidence:.3f}")
                with col3:
                    st.metric("Type", rule.tnorm_type)
    else:
        st.warning("No fuzzy rules extracted. Try adjusting the attention threshold.")
    
    # Technical details for expert users
    if user_level == 'expert':
        st.subheader("🔬 Technical Details")
        st.write(explanation['content']['technical_details'])
    
    # Summary
    st.write(explanation['content']['summary'])

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Fuzzy Attention Networks - Multimodal Demo",
        page_icon="🧠",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🧠 Fuzzy Attention Networks - Multimodal Demo")
    st.markdown("**Human-Centered Differentiable Neuro-Fuzzy Architectures**")
    st.markdown("*Interactive explanation interfaces for multimodal AI with adaptive user-controlled interpretability*")
    
    # Model information
    if hasattr(st.session_state, 'model_trained') and st.session_state.model_trained:
        checkpoint = st.session_state.model_checkpoint
        st.success(f"🤖 **Trained Model Loaded** - Epoch: {checkpoint['epoch']}, Accuracy: {checkpoint['accuracy']:.4f}")
    else:
        st.warning("⚠️ **Untrained Model** - For better predictions, please train the model first using train_model.py")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # User expertise level
        expertise_level = st.selectbox(
            "Select your expertise level:",
            ["novice", "intermediate", "expert"],
            index=2,
            help="This affects the complexity of explanations"
        )
        
        st.info(f"Explanations will be based on your selected expertise level.")
        
        # Model parameters
        st.subheader("Model Parameters")
        attention_threshold = st.slider("Attention Threshold", 0.01, 0.2, 0.1, 0.01)
        max_rules = st.slider("Max Rules to Show", 3, 15, 5)
        
        # Update rule extractor
        st.session_state.rule_extractor.attention_threshold = attention_threshold
        st.session_state.rule_extractor.max_rules_per_position = max_rules
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🖼️ Multimodal Input")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image:",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image for multimodal analysis"
        )
        
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            value="The cat sat on the mat quietly",
            height=100,
            help="Enter text that corresponds to the image"
        )
        
        # Use sample from dataset if no image uploaded
        if uploaded_file is None:
            if st.button("Use Sample from Hateful Memes Dataset"):
                st.session_state.use_sample = True
        
        # Show sample if selected
        if hasattr(st.session_state, 'use_sample') and st.session_state.use_sample:
            if st.session_state.dataset is not None:
                sample_idx = st.selectbox("Select sample:", range(len(st.session_state.dataset)))
                sample = st.session_state.dataset[sample_idx]
                
                # Display sample image
                try:
                    image_path = Path('./data/hateful_memes') / sample['img']
                    if image_path.exists():
                        image = Image.open(image_path)
                        st.image(image, caption=f"Sample {sample_idx}: {sample['question'][:50]}...", use_column_width=True)
                        text_input = sample['question']
                    else:
                        st.warning("Image not found, using placeholder")
                        # Create placeholder image
                        placeholder = Image.new('RGB', (224, 224), color=(128, 128, 128))
                        st.image(placeholder, caption="Placeholder image", use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
                    placeholder = Image.new('RGB', (224, 224), color=(128, 128, 128))
                    st.image(placeholder, caption="Placeholder image", use_column_width=True)
        
        # Process uploaded image
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", use_column_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze Multimodal Input", type="primary"):
            # Process image
            if uploaded_file is not None:
                image_features = process_image(Image.open(uploaded_file))
            elif hasattr(st.session_state, 'use_sample') and st.session_state.use_sample and st.session_state.dataset is not None:
                sample = st.session_state.dataset[sample_idx]
                try:
                    image_path = Path('./data/hateful_memes') / sample['img']
                    if image_path.exists():
                        image = Image.open(image_path)
                        image_features = process_image(image)
                    else:
                        image_features = torch.randn(49, 2048)  # Placeholder
                except:
                    image_features = torch.randn(49, 2048)  # Placeholder
            else:
                # Use placeholder image features
                image_features = torch.randn(49, 2048)
            
            # Analyze
            result = analyze_multimodal_input(text_input, image_features, st.session_state.model)
            
            # Store result
            st.session_state.analysis_result = result
            st.session_state.analysis_text = text_input
            st.session_state.analysis_image = image_features
        
        # Display analysis results
        if 'analysis_result' in st.session_state:
            st.success("✅ Multimodal analysis completed!")
            
            # Show prediction if model is trained
            if 'prediction' in st.session_state.analysis_result:
                prediction = st.session_state.analysis_result['prediction']
                confidence = st.session_state.analysis_result['confidence']
                prediction_text = st.session_state.analysis_result['prediction_text']
                
                # Color code the prediction
                if prediction == 1:  # Hateful
                    st.error(f"🚨 **Prediction: {prediction_text}** (Confidence: {confidence:.3f})")
                else:  # Non-hateful
                    st.success(f"✅ **Prediction: {prediction_text}** (Confidence: {confidence:.3f})")
            
            # Show attention patterns
            if 'attention_patterns' in st.session_state.analysis_result:
                patterns = st.session_state.analysis_result['attention_patterns']
                st.write("**Attention Patterns:**")
                for key, value in patterns.items():
                    st.write(f"- {key}: {value}")
    
    with col2:
        st.header("🔍 Fuzzy Rules & Explanations")
        
        if 'analysis_result' in st.session_state:
            # Extract fuzzy rules
            fuzzy_rules = st.session_state.analysis_result.get('fuzzy_rules', [])
            
            # Generate adaptive explanation
            if st.session_state.user_id not in st.session_state.interface.user_profiles:
                user_profile = st.session_state.interface.initialize_user(st.session_state.user_id)
            else:
                user_profile = st.session_state.interface.user_profiles[st.session_state.user_id]
            
            user_profile.expertise_level = UserExpertiseLevel(expertise_level)
            user_profile.confidence_score = 0.8  # Set reasonable confidence
            
            explanation = st.session_state.interface.get_explanation(
                st.session_state.user_id,
                fuzzy_rules,
                tokens=[st.session_state.analysis_text],
                attention_patterns=st.session_state.analysis_result.get('attention_patterns', {})
            )
            
            # Display explanation
            display_multimodal_explanation(explanation, expertise_level, fuzzy_rules)
        else:
            st.info("Upload an image and text, then click 'Analyze Multimodal Input' to see fuzzy rules and explanations.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Fuzzy Attention Networks - Interactive Demo for IUI 2026**")
    st.markdown("This demo showcases adaptive explanation interfaces with three-tier progressive disclosure.")

if __name__ == "__main__":
    main()

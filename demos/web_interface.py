#!/usr/bin/env python3
"""
Web Interface for Fuzzy Attention Networks
Streamlit-based interactive demo with adaptive explanations
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

# Add src to Python path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import our modules
from multimodal_fuzzy_attention import VQAFuzzyModel
from adaptive_interface import InteractiveInterface, UserExpertiseLevel
from rule_extractor import RuleExtractor, FuzzyRule
# Add experiments to path
experiments_path = os.path.join(os.path.dirname(__file__), '..', 'experiments')
if experiments_path not in sys.path:
    sys.path.insert(0, experiments_path)

from evaluation_framework import BeansHFDataset, HatefulMemesLocalDataset

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

def load_sample_data():
    """Load sample data for demonstration"""
    try:
        # Try to load Beans dataset
        dataset = BeansHFDataset(split='train', max_samples=5, device='cpu')
        if len(dataset) > 0:
            sample = dataset[0]
            return {
                'text': sample['question'],
                'image_features': sample['image_features'],
                'tokens': sample['question_tokens'],
                'label': sample.get('answer', 'unknown')
            }
    except:
        pass
    
    # Fallback to mock data
    return {
        'text': "This image shows a cat sitting on a mat",
        'image_features': torch.randn(1, 49, 512),
        'tokens': torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
        'label': 'cat'
    }

def process_text_input(text: str) -> torch.Tensor:
    """Convert text input to tokens"""
    # Simple tokenization (in real implementation, use proper tokenizer)
    words = text.lower().split()
    vocab_size = 10000
    tokens = torch.tensor([abs(hash(word)) % vocab_size for word in words[:20]], dtype=torch.long)
    if len(tokens) == 0:
        tokens = torch.tensor([0], dtype=torch.long)
    return tokens

def analyze_text_with_model(text: str, model: VQAFuzzyModel) -> Dict[str, Any]:
    """Analyze text using the fuzzy attention model"""
    # Process text
    tokens = process_text_input(text)
    tokens = tokens.unsqueeze(0)  # Add batch dimension
    
    # Create mock image features
    image_features = torch.randn(1, 49, 2048)  # Updated to match model
    
    # Run model
    with torch.no_grad():
        result = model(tokens, image_features, return_explanations=True)
    
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

def display_adaptive_explanation(explanation: Dict[str, Any], user_level: str):
    """Display adaptive explanation based on user level"""
    
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
    
    # Rules
    st.write("**Key Relationships:**")
    for i, rule in enumerate(explanation['content']['rules']):
        with st.expander(f"Rule {i+1}: {rule['text'][:50]}..."):
            st.write(f"**Description:** {rule['text']}")
            st.write(f"**Strength:** {rule['strength']:.3f}")
            st.write(f"**Confidence:** {rule['confidence']:.3f}")
            st.write(f"**From Position:** {rule['from_position']}")
            st.write(f"**To Position:** {rule['to_position']}")
    
    # Summary
    st.write("**Summary:**")
    st.write(explanation['content']['summary'])
    
    # Technical details for expert users
    if user_level == 'expert' and explanation['content']['technical_details']:
        st.write("**Technical Details:**")
        st.json(explanation['metadata'])

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Fuzzy Attention Networks Demo",
        page_icon="🧠",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🧠 Fuzzy Attention Networks Demo")
    st.markdown("**Human-Centered Differentiable Neuro-Fuzzy Architectures**")
    st.markdown("Interactive explanation interfaces for multimodal AI with adaptive user-controlled interpretability")
    
    # Sidebar for user controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # User expertise level selector
        st.subheader("User Expertise Level")
        expertise_level = st.selectbox(
            "Select your expertise level:",
            ["novice", "intermediate", "expert"],
            index=1
        )
        
        # Manual expertise adjustment
        if st.checkbox("Enable Real-time Expertise Assessment"):
            st.info("The system will automatically adjust explanations based on your interactions.")
        else:
            st.info("Explanations will be based on your selected expertise level.")
        
        # Model parameters
        st.subheader("Model Parameters")
        attention_threshold = st.slider("Attention Threshold", 0.01, 0.2, 0.1, 0.01)
        max_rules = st.slider("Max Rules to Show", 3, 15, 5)
        
        # Update rule extractor
        st.session_state.rule_extractor.attention_threshold = attention_threshold
        st.session_state.rule_extractor.max_rules_per_position = max_rules
    
    # Model information
    if hasattr(st.session_state, 'model_trained') and st.session_state.model_trained:
        checkpoint = st.session_state.model_checkpoint
        st.success(f"🤖 **Trained Model Loaded** - Epoch: {checkpoint['epoch']}, Accuracy: {checkpoint['accuracy']:.4f}")
    else:
        st.warning("⚠️ **Untrained Model** - For better predictions, please train the model first using train_model.py")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Text")
        
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            value="The cat sat on the mat quietly",
            height=100
        )
        
        # Analyze button
        if st.button("🔍 Analyze Text", type="primary"):
            with st.spinner("Analyzing text with fuzzy attention..."):
                # Process user interaction
                st.session_state.interface.process_interaction(
                    st.session_state.user_id,
                    'request_explanation',
                    'text_analysis',
                    context={'explanation_depth': expertise_level}
                )
                
                # Analyze text
                result = analyze_text_with_model(text_input, st.session_state.model)
                
                # Store result in session state
                st.session_state.analysis_result = result
                st.session_state.analysis_text = text_input
        
        # Display analysis results
        if 'analysis_result' in st.session_state:
            st.success("✅ Analysis completed!")
            
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
            
            # Show basic results
            st.write("**Model Output:**")
            if 'answer_logits' in st.session_state.analysis_result:
                probs = torch.softmax(st.session_state.analysis_result['answer_logits'], dim=-1)
                top_probs, top_indices = torch.topk(probs, 3)
                for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
                    st.write(f"{i+1}. Class {idx.item()}: {prob.item():.3f}")
    
    with col2:
        st.header("🔍 Fuzzy Rules & Explanations")
        
        if 'analysis_result' in st.session_state:
            # Get user profile
            user_profile = st.session_state.interface.user_profiles[st.session_state.user_id]
            
            # Generate adaptive explanation
            rules = st.session_state.analysis_result.get('fuzzy_rules', [])
            tokens = st.session_state.analysis_text.split() if 'analysis_text' in st.session_state else None
            attention_patterns = st.session_state.analysis_result.get('attention_patterns', {})
            
            if rules:
                explanation = st.session_state.interface.get_explanation(
                    st.session_state.user_id,
                    rules,
                    tokens,
                    attention_patterns
                )
                
                # Display explanation
                display_adaptive_explanation(explanation, expertise_level)
            else:
                st.warning("No fuzzy rules extracted. Try adjusting the attention threshold.")
        else:
            st.info("Enter text and click 'Analyze Text' to see fuzzy rules and explanations.")
    
    # Bottom section - User interaction tracking
    with st.expander("📊 User Interaction Analytics"):
        if st.session_state.user_id in st.session_state.interface.user_profiles:
            user_profile = st.session_state.interface.user_profiles[st.session_state.user_id]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Expertise Level", user_profile.expertise_level.value.title())
                st.metric("Confidence Score", f"{user_profile.confidence_score:.3f}")
            
            with col2:
                st.metric("Total Interactions", len(user_profile.interaction_history))
                st.metric("Last Updated", time.strftime("%H:%M:%S", time.localtime(user_profile.last_updated)))
            
            with col3:
                st.write("**Expertise Indicators:**")
                for indicator, value in user_profile.expertise_indicators.items():
                    st.write(f"- {indicator.replace('_', ' ').title()}: {value:.3f}")
            
            # Interaction history
            if user_profile.interaction_history:
                st.write("**Recent Interactions:**")
                for interaction in user_profile.interaction_history[-5:]:
                    st.write(f"- {interaction.action_type} on {interaction.target_element} ({interaction.duration:.1f}s)")
    
    # Footer
    st.markdown("---")
    st.markdown("**Fuzzy Attention Networks** - Interactive Demo for IUI 2026")
    st.markdown("This demo showcases adaptive explanation interfaces with three-tier progressive disclosure.")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Advanced Multimodal Web Interface for Fuzzy Attention Networks
Includes all features from the abstract: learnable components, real-time assessment, interactive editing
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
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

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
from learnable_fuzzy_components import (
    LearnableFuzzyAttention, 
    MembershipFunctionVisualizer,
    CompositionalRuleDeriver
)
from realtime_expertise_assessment import (
    RealTimeExpertiseAssessor, 
    InteractionType
)
from interactive_rule_editor import (
    InteractiveRuleEditor, 
    RuleEditAction
)

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
    
    if 'realtime_assessor' not in st.session_state:
        st.session_state.realtime_assessor = RealTimeExpertiseAssessor()
    
    if 'rule_editor' not in st.session_state:
        st.session_state.rule_editor = InteractiveRuleEditor(st.session_state.rule_extractor)
    
    if 'learnable_attention' not in st.session_state:
        st.session_state.learnable_attention = LearnableFuzzyAttention(
            input_dim=256, num_heads=4, num_membership_functions=5
        )
    
    if 'compositional_deriver' not in st.session_state:
        st.session_state.compositional_deriver = CompositionalRuleDeriver(st.session_state.rule_extractor)
    
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

def process_image_for_demo(image):
    """Process uploaded image to features"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_tensor = torch.tensor(image_array, dtype=torch.float32)
    
    # Create mock features (in real implementation, use CLIP)
    image_features = torch.randn(49, 2048)
    return image_features

def process_text_input(text: str) -> torch.Tensor:
    """Process text input to tokens"""
    words = text.lower().split()[:20]
    vocab_size = 10000
    tokens = torch.tensor([abs(hash(word)) % vocab_size for word in words], dtype=torch.long)
    if len(tokens) == 0:
        tokens = torch.tensor([0], dtype=torch.long)
    return tokens

def analyze_multimodal_input(text: str, image_features: torch.Tensor, model: VQAFuzzyModel) -> Dict[str, Any]:
    """Analyze multimodal input using the fuzzy attention model"""
    # Process text
    text_tokens = process_text_input(text)
    text_tokens = text_tokens.unsqueeze(0)
    
    # Add batch dimension to image features
    image_features = image_features.unsqueeze(0)
    
    # Run model
    with torch.no_grad():
        result = model(text_tokens, image_features, return_explanations=True)
    
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

def visualize_membership_functions():
    """Visualize learnable membership functions"""
    # Create input range
    x_range = torch.linspace(-3, 3, 100)
    
    # Get membership functions from learnable attention
    learnable_attention = st.session_state.learnable_attention
    
    # Create visualizer
    membership_functions = {
        'gaussian': learnable_attention.gaussian_membership,
        'triangular': learnable_attention.triangular_membership
    }
    
    visualizer = MembershipFunctionVisualizer(membership_functions)
    visualizations = visualizer.visualize_membership_functions(x_range)
    
    # Create plots
    fig = go.Figure()
    
    for func_name, values in visualizations.items():
        for i in range(values.shape[1]):
            fig.add_trace(go.Scatter(
                x=x_range.numpy(),
                y=values[:, i].numpy(),
                mode='lines',
                name=f'{func_name} {i+1}',
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title="Learnable Membership Functions",
        xaxis_title="Input Value",
        yaxis_title="Membership Degree",
        hovermode='x unified'
    )
    
    return fig

def display_compositional_rules(fuzzy_rules: List[FuzzyRule], tokens: List[str]):
    """Display compositional rule derivations"""
    if not fuzzy_rules:
        return
    
    # Create compositional derivations
    compositional_rules = st.session_state.compositional_deriver.derive_compositional_rules(
        torch.randn(1, len(tokens), len(tokens)),  # Mock attention matrix
        tokens
    )
    
    if compositional_rules:
        st.subheader("🧩 Compositional Rule Derivations")
        
        for i, comp_rule in enumerate(compositional_rules[:3]):  # Show top 3
            with st.expander(f"Composition {i+1}: {comp_rule['source_token']}"):
                st.write(f"**Source Position:** {comp_rule['source_position']}")
                st.write(f"**Composition Strength:** {comp_rule['composition_strength']:.3f}")
                st.write(f"**Composition Type:** {comp_rule['composition_type']}")
                
                st.write("**Individual Rules:**")
                for j, rule in enumerate(comp_rule['individual_rules'][:3]):
                    st.write(f"  {j+1}. {rule.linguistic_description} (strength: {rule.strength:.3f})")
                
                st.write("**Derivation:**")
                st.text(comp_rule['derivation'])

def display_interactive_rule_editor(fuzzy_rules: List[FuzzyRule]):
    """Display interactive rule editor"""
    if not fuzzy_rules:
        st.info("No rules available for editing")
        return
    
    st.subheader("✏️ Interactive Rule Editor")
    st.write("Edit, validate, and refine fuzzy rules interactively")
    
    # Select rule to edit
    rule_options = {f"Rule {i+1}: {rule.linguistic_description[:50]}...": rule 
                   for i, rule in enumerate(fuzzy_rules[:10])}
    
    selected_rule_name = st.selectbox("Select rule to edit:", list(rule_options.keys()))
    selected_rule = rule_options[selected_rule_name]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Current Rule Properties:**")
        st.write(f"Strength: {selected_rule.strength:.3f}")
        st.write(f"Confidence: {selected_rule.confidence:.3f}")
        st.write(f"Description: {selected_rule.linguistic_description}")
    
    with col2:
        st.write("**Edit Rule:**")
        
        # Edit strength
        new_strength = st.slider("Strength", 0.0, 1.0, selected_rule.strength, 0.01)
        if new_strength != selected_rule.strength:
            if st.button("Update Strength"):
                success = st.session_state.rule_editor.edit_rule(
                    st.session_state.user_id,
                    f"rule_{id(selected_rule)}",
                    RuleEditAction.MODIFY_STRENGTH,
                    new_strength,
                    "User adjusted strength"
                )
                if success:
                    st.success("Strength updated!")
                    st.rerun()
        
        # Edit confidence
        new_confidence = st.slider("Confidence", 0.0, 1.0, selected_rule.confidence, 0.01)
        if new_confidence != selected_rule.confidence:
            if st.button("Update Confidence"):
                success = st.session_state.rule_editor.edit_rule(
                    st.session_state.user_id,
                    f"rule_{id(selected_rule)}",
                    RuleEditAction.MODIFY_CONFIDENCE,
                    new_confidence,
                    "User adjusted confidence"
                )
                if success:
                    st.success("Confidence updated!")
                    st.rerun()
        
        # Edit description
        new_description = st.text_area("Description", selected_rule.linguistic_description)
        if new_description != selected_rule.linguistic_description:
            if st.button("Update Description"):
                success = st.session_state.rule_editor.edit_rule(
                    st.session_state.user_id,
                    f"rule_{id(selected_rule)}",
                    RuleEditAction.MODIFY_DESCRIPTION,
                    new_description,
                    "User updated description"
                )
                if success:
                    st.success("Description updated!")
                    st.rerun()
    
    # Validate rule
    if st.button("Validate Rule"):
        validation = st.session_state.rule_editor.validate_rule(
            f"rule_{id(selected_rule)}",
            {'tokens': ['sample', 'tokens']}
        )
        
        if validation.is_valid:
            st.success(f"✅ Rule is valid (confidence: {validation.confidence:.3f})")
        else:
            st.error(f"❌ Rule validation failed: {validation.feedback}")
        
        if validation.suggested_improvements:
            st.write("**Suggested Improvements:**")
            for improvement in validation.suggested_improvements:
                st.write(f"• {improvement}")

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Advanced Fuzzy Attention Networks",
        page_icon="🧠",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🧠 Advanced Fuzzy Attention Networks")
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
        st.header("🎛️ Advanced Controls")
        
        # User expertise level
        expertise_level = st.selectbox(
            "Select your expertise level:",
            ["novice", "intermediate", "expert"],
            index=2,
            help="This affects the complexity of explanations"
        )
        
        # Real-time expertise assessment
        if st.button("Assess My Expertise"):
            expertise_level_num, confidence, indicators = st.session_state.realtime_assessor.assess_expertise(st.session_state.user_id)
            st.write(f"**Assessed Level:** {['Novice', 'Intermediate', 'Expert'][expertise_level_num]}")
            st.write(f"**Confidence:** {confidence:.3f}")
            st.write("**Indicators:**")
            for key, value in indicators.items():
                st.write(f"  {key}: {value:.3f}")
        
        # Model parameters
        st.subheader("Model Parameters")
        attention_threshold = st.slider("Attention Threshold", 0.01, 0.2, 0.1, 0.01)
        max_rules = st.slider("Max Rules to Show", 3, 15, 5)
        
        # Update rule extractor
        st.session_state.rule_extractor.attention_threshold = attention_threshold
        st.session_state.rule_extractor.max_rules_per_position = max_rules
        
        # Membership function visualization
        if st.button("Show Membership Functions"):
            st.session_state.show_membership_functions = True
    
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
        
        # Use sample from dataset
        if uploaded_file is None:
            if st.button("Use Sample from Hateful Memes Dataset"):
                st.session_state.use_sample = True
        
        # Show sample if selected
        if hasattr(st.session_state, 'use_sample') and st.session_state.use_sample:
            if st.session_state.dataset is not None:
                sample_idx = st.selectbox("Select sample:", range(len(st.session_state.dataset)))
                sample = st.session_state.dataset[sample_idx]
                
                try:
                    image_path = Path('./data/hateful_memes') / sample.get('img', 'img/placeholder.png')
                    if image_path.exists():
                        image = Image.open(image_path)
                        st.image(image, caption=f"Sample {sample_idx}: {sample['question'][:50]}...", use_column_width=True)
                        text_input = sample['question']
                    else:
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
            # Log interaction
            st.session_state.realtime_assessor.log_interaction(
                st.session_state.user_id,
                InteractionType.TEXT_INPUT,
                {'text': text_input, 'has_image': uploaded_file is not None},
                complexity_score=0.7
            )
            
            # Process image
            if uploaded_file is not None:
                image_features = process_image_for_demo(Image.open(uploaded_file))
            elif hasattr(st.session_state, 'use_sample') and st.session_state.use_sample and st.session_state.dataset is not None:
                sample = st.session_state.dataset[sample_idx]
                try:
                    image_path = Path('./data/hateful_memes') / sample.get('img', 'img/placeholder.png')
                    if image_path.exists():
                        image = Image.open(image_path)
                        image_features = process_image_for_demo(image)
                    else:
                        image_features = torch.randn(49, 2048)
                except:
                    image_features = torch.randn(49, 2048)
            else:
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
        st.header("🔍 Advanced Explanations")
        
        if 'analysis_result' in st.session_state:
            # Extract fuzzy rules
            fuzzy_rules = st.session_state.analysis_result.get('fuzzy_rules', [])
            
            # Generate adaptive explanation
            if st.session_state.user_id not in st.session_state.interface.user_profiles:
                user_profile = st.session_state.interface.initialize_user(st.session_state.user_id)
            else:
                user_profile = st.session_state.interface.user_profiles[st.session_state.user_id]
            
            user_profile.expertise_level = UserExpertiseLevel(expertise_level)
            user_profile.confidence_score = 0.8
            
            explanation = st.session_state.interface.get_explanation(
                st.session_state.user_id,
                fuzzy_rules,
                tokens=[st.session_state.analysis_text],
                attention_patterns=st.session_state.analysis_result.get('attention_patterns', {})
            )
            
            # Display explanation
            st.subheader(f"🎯 {explanation['title']}")
            
            level_colors = {'novice': '🟢', 'intermediate': '🟡', 'expert': '🔴'}
            st.info(f"{level_colors.get(expertise_level, '🟡')} **User Level:** {expertise_level.upper()} (Confidence: {explanation['confidence']:.2f})")
            
            st.write(explanation['content']['intro'])
            
            # Show fuzzy rules
            if fuzzy_rules:
                st.subheader("🧠 Extracted Fuzzy Rules")
                
                for i, rule in enumerate(fuzzy_rules[:max_rules]):
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
            
            # Show compositional rules for expert users
            if expertise_level == 'expert':
                display_compositional_rules(fuzzy_rules, [st.session_state.analysis_text])
            
            # Show interactive rule editor
            if fuzzy_rules:
                display_interactive_rule_editor(fuzzy_rules)
            
            # Technical details for expert users
            if expertise_level == 'expert':
                st.subheader("🔬 Technical Details")
                st.write(explanation['content']['technical_details'])
            
            st.write(explanation['content']['summary'])
        else:
            st.info("Upload an image and text, then click 'Analyze Multimodal Input' to see advanced explanations.")
    
    # Membership function visualization
    if hasattr(st.session_state, 'show_membership_functions') and st.session_state.show_membership_functions:
        st.subheader("📊 Learnable Membership Functions")
        fig = visualize_membership_functions()
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**Advanced Fuzzy Attention Networks - Interactive Demo for IUI 2026**")
    st.markdown("This demo showcases all features from the abstract: learnable components, real-time assessment, interactive editing")

if __name__ == "__main__":
    main()

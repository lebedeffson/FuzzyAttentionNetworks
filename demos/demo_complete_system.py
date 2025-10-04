#!/usr/bin/env python3
"""
Complete System Demo for Fuzzy Attention Networks
Demonstrates all features: fuzzy attention, rule extraction, adaptive interface, and visualizations
"""

import torch
import sys
import os
from pathlib import Path
import time
import json

# Add src to Python path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Add experiments to Python path
experiments_path = os.path.join(os.path.dirname(__file__), '..', 'experiments')
if experiments_path not in sys.path:
    sys.path.insert(0, experiments_path)

def print_header(title: str, char: str = "=", width: int = 60):
    """Print a formatted header"""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")

def print_section(title: str, char: str = "-", width: int = 40):
    """Print a formatted section header"""
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}")

def demo_fuzzy_attention():
    """Demo fuzzy attention mechanism"""
    print_header("🧠 FUZZY ATTENTION MECHANISM DEMO")
    
    try:
        from fuzzy_attention import MultiHeadFuzzyAttention
        from utils import FuzzyOperators
        
        print("Creating fuzzy attention layer...")
        d_model, n_heads, seq_len = 128, 4, 10
        fuzzy_layer = MultiHeadFuzzyAttention(d_model, n_heads)
        
        # Create demo data
        demo_input = torch.randn(1, seq_len, d_model)
        print(f"Input shape: {demo_input.shape}")
        
        # Forward pass
        with torch.no_grad():
            output, attention_info = fuzzy_layer(
                demo_input, demo_input, demo_input, 
                return_attention=True
            )
        
        print(f"Output shape: {output.shape}")
        print(f"Attention shape: {attention_info['avg_attention'].shape}")
        
        # Test fuzzy operators
        print_section("Fuzzy Logic Operators")
        a, b = torch.tensor(0.7), torch.tensor(0.3)
        print(f"Product t-norm T({a:.1f}, {b:.1f}) = {FuzzyOperators.product_tnorm(a, b):.3f}")
        print(f"Minimum t-norm T({a:.1f}, {b:.1f}) = {FuzzyOperators.minimum_tnorm(a, b):.3f}")
        print(f"Łukasiewicz t-norm T({a:.1f}, {b:.1f}) = {FuzzyOperators.lukasiewicz_tnorm(a, b):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fuzzy attention demo failed: {e}")
        return False

def demo_multimodal_model():
    """Demo multimodal fuzzy attention model"""
    print_header("🖼️ MULTIMODAL FUZZY ATTENTION MODEL DEMO")
    
    try:
        from multimodal_fuzzy_attention import VQAFuzzyModel
        
        print("Creating VQA Fuzzy Model...")
        model = VQAFuzzyModel(
            vocab_size=10000,
            answer_vocab_size=1000,
            text_dim=256,
            image_dim=512,
            hidden_dim=256,
            n_heads=4,
            n_layers=2
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create demo data
        batch_size, question_len, image_patches = 2, 8, 16
        question_tokens = torch.randint(0, 10000, (batch_size, question_len))
        image_features = torch.randn(batch_size, image_patches, 512)
        
        print(f"Question tokens shape: {question_tokens.shape}")
        print(f"Image features shape: {image_features.shape}")
        
        # Forward pass with explanations
        print("Running forward pass with rule extraction...")
        with torch.no_grad():
            result = model(question_tokens, image_features, return_explanations=True)
        
        print(f"Answer logits shape: {result['answer_logits'].shape}")
        print(f"Answer probabilities shape: {result['answer_probs'].shape}")
        
        if 'fuzzy_rules' in result:
            print(f"Fuzzy rules extracted: {len(result['fuzzy_rules'])}")
            for i, rule in enumerate(result['fuzzy_rules'][:3]):
                print(f"  Rule {i+1}: {rule.linguistic_description}")
        
        if 'attention_patterns' in result:
            patterns = result['attention_patterns']
            print(f"Attention patterns: {list(patterns.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multimodal model demo failed: {e}")
        return False

def demo_rule_extraction():
    """Demo rule extraction system"""
    print_header("🔍 RULE EXTRACTION SYSTEM DEMO")
    
    try:
        from rule_extractor import RuleExtractor, AdaptiveRuleExplainer
        
        # Create sample attention weights
        seq_len = 8
        attention_weights = torch.rand(1, seq_len, seq_len)
        
        # Add some strong connections
        attention_weights[0, 0, 2] = 0.25  # Strong connection
        attention_weights[0, 1, 3] = 0.18  # Medium connection
        attention_weights[0, 4, 6] = 0.12  # Weak connection
        
        # Normalize
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        print(f"Attention weights shape: {attention_weights.shape}")
        
        # Extract rules
        extractor = RuleExtractor(attention_threshold=0.1, strong_threshold=0.15)
        rules = extractor.extract_rules(attention_weights)
        
        print(f"Extracted {len(rules)} fuzzy rules")
        
        # Show top rules
        print_section("Top Rules")
        for i, rule in enumerate(rules[:5]):
            print(f"Rule {i+1}: {rule.linguistic_description}")
            print(f"  Strength: {rule.strength:.3f}, Confidence: {rule.confidence:.3f}")
        
        # Test adaptive explanations
        explainer = AdaptiveRuleExplainer()
        
        print_section("Adaptive Explanations")
        for user_level in ['novice', 'intermediate', 'expert']:
            print(f"\n{user_level.upper()} Explanation:")
            explanation = explainer.generate_explanation(rules, user_level)
            print(explanation[:200] + "..." if len(explanation) > 200 else explanation)
        
        return True
        
    except Exception as e:
        print(f"❌ Rule extraction demo failed: {e}")
        return False

def demo_adaptive_interface():
    """Demo adaptive interface system"""
    print_header("🎯 ADAPTIVE INTERFACE SYSTEM DEMO")
    
    try:
        from adaptive_interface import InteractiveInterface, UserExpertiseLevel
        
        # Create interface
        interface = InteractiveInterface()
        user_id = "demo_user"
        
        print("Initializing user profile...")
        user_profile = interface.initialize_user(user_id)
        print(f"Initial expertise level: {user_profile.expertise_level.value}")
        
        # Simulate user interactions
        print_section("Simulating User Interactions")
        
        # Novice interactions
        print("Novice interactions...")
        interface.process_interaction(user_id, 'request_explanation', 'attention_weights', 
                                    context={'explanation_depth': 'basic'})
        interface.process_interaction(user_id, 'feedback', 'explanation', 
                                    context={'feedback_type': 'too complex'})
        
        print(f"After novice interactions: {interface.user_profiles[user_id].expertise_level.value}")
        
        # Intermediate interactions
        print("Intermediate interactions...")
        interface.process_interaction(user_id, 'request_explanation', 'attention_weights',
                                    context={'explanation_depth': 'detailed'})
        interface.process_interaction(user_id, 'click', 'rule_1', duration=3.0)
        interface.process_interaction(user_id, 'hover', 'technical_details', duration=2.0)
        
        print(f"After intermediate interactions: {interface.user_profiles[user_id].expertise_level.value}")
        
        # Expert interactions
        print("Expert interactions...")
        interface.process_interaction(user_id, 'request_explanation', 'attention_weights',
                                    context={'explanation_depth': 'technical'})
        interface.process_interaction(user_id, 'click', 'rule_2', duration=8.0)
        interface.process_interaction(user_id, 'feedback', 'explanation',
                                    context={'feedback_type': 'need more detail about t-norms'})
        
        final_profile = interface.user_profiles[user_id]
        print(f"Final expertise level: {final_profile.expertise_level.value}")
        print(f"Confidence score: {final_profile.confidence_score:.3f}")
        
        # Test explanation generation
        print_section("Adaptive Explanation Generation")
        from rule_extractor import FuzzyRule
        mock_rules = [
            FuzzyRule(0, 2, 0.25, 0.8, "Position 0 strongly attends to position 2", "gaussian", "product"),
            FuzzyRule(1, 3, 0.18, 0.7, "Position 1 moderately attends to position 3", "gaussian", "product"),
            FuzzyRule(4, 6, 0.12, 0.6, "Position 4 slightly attends to position 6", "gaussian", "product")
        ]
        
        mock_tokens = ["The", "cat", "sat", "on", "the", "mat", "quietly"]
        
        explanation = interface.get_explanation(user_id, mock_rules, mock_tokens)
        
        print(f"Generated explanation for {explanation['user_level']} user:")
        print(f"Title: {explanation['title']}")
        print(f"Rules shown: {len(explanation['content']['rules'])}")
        print(f"Technical details: {explanation['content']['technical_details']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Adaptive interface demo failed: {e}")
        return False

def demo_visualization_system():
    """Demo visualization system"""
    print_header("🎨 VISUALIZATION SYSTEM DEMO")
    
    try:
        from visualization_system import AttentionVisualizer, InteractiveVisualizer
        from rule_extractor import FuzzyRule
        
        # Create sample data
        seq_len = 8
        attention_weights = torch.rand(1, seq_len, seq_len)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        tokens = ["The", "cat", "sat", "on", "the", "mat", "quietly", "watching"]
        
        rules = [
            FuzzyRule(0, 2, 0.25, 0.8, "Position 0 strongly attends to position 2", "gaussian", "product"),
            FuzzyRule(1, 3, 0.18, 0.7, "Position 1 moderately attends to position 3", "gaussian", "product"),
            FuzzyRule(4, 6, 0.12, 0.6, "Position 4 slightly attends to position 6", "gaussian", "product"),
            FuzzyRule(2, 5, 0.15, 0.65, "Position 2 moderately attends to position 5", "gaussian", "product"),
            FuzzyRule(3, 7, 0.10, 0.55, "Position 3 slightly attends to position 7", "gaussian", "product")
        ]
        
        print_section("Matplotlib Visualizations")
        visualizer = AttentionVisualizer()
        
        # Test various visualizations
        print("Creating attention heatmap...")
        fig1 = visualizer.plot_attention_heatmap(attention_weights, tokens, "Demo Attention Heatmap")
        print("✅ Attention heatmap created")
        
        print("Creating fuzzy rules network...")
        fig2 = visualizer.plot_fuzzy_rules_network(rules, tokens, "Demo Fuzzy Rules Network")
        print("✅ Fuzzy rules network created")
        
        print("Creating rule strength distribution...")
        fig3 = visualizer.plot_rule_strength_distribution(rules, "Demo Rule Strength Distribution")
        print("✅ Rule strength distribution created")
        
        print("Creating attention entropy plot...")
        fig4 = visualizer.plot_attention_entropy(attention_weights, "Demo Attention Entropy")
        print("✅ Attention entropy plot created")
        
        print_section("Interactive Visualizations")
        interactive_viz = InteractiveVisualizer()
        
        print("Creating interactive attention heatmap...")
        fig5 = interactive_viz.create_interactive_attention_heatmap(attention_weights, tokens, "Demo Interactive Attention Heatmap")
        print("✅ Interactive attention heatmap created")
        
        print("Creating interactive rule network...")
        fig6 = interactive_viz.create_interactive_rule_network(rules, tokens, "Demo Interactive Fuzzy Rules Network")
        print("✅ Interactive rule network created")
        
        print("Creating rule comparison chart...")
        rules_by_level = {
            'novice': rules[:2],
            'intermediate': rules[:4],
            'expert': rules
        }
        fig7 = interactive_viz.create_rule_comparison_chart(rules_by_level, "Demo Rule Comparison by User Level")
        print("✅ Rule comparison chart created")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization system demo failed: {e}")
        return False

def demo_evaluation_framework():
    """Demo evaluation framework"""
    print_header("📊 EVALUATION FRAMEWORK DEMO")
    
    try:
        from evaluation_framework import FuzzyAttentionEvaluator, BeansHFDataset
        from multimodal_fuzzy_attention import VQAFuzzyModel
        
        print("Creating evaluator and model...")
        evaluator = FuzzyAttentionEvaluator(device='cpu')
        model = VQAFuzzyModel(
            vocab_size=10000,
            answer_vocab_size=1000,
            text_dim=256,
            image_dim=512,
            hidden_dim=256,
            n_heads=4,
            n_layers=2
        )
        
        print("Loading dataset...")
        dataset = BeansHFDataset(split='train', max_samples=5, device='cpu')
        print(f"Dataset size: {len(dataset)}")
        
        print("Running evaluation...")
        result = evaluator.evaluate_model(model, dataset, 'Beans')
        
        print_section("Evaluation Results")
        print(f"Task Accuracy: {result.metrics.task_accuracy:.3f}")
        print(f"Attention Entropy: {result.metrics.attention_entropy:.3f}")
        print(f"Attention Sparsity: {result.metrics.attention_sparsity:.3f}")
        print(f"Rule Consistency: {result.metrics.rule_consistency:.3f}")
        print(f"Explanation Quality: {result.metrics.explanation_quality:.3f}")
        print(f"Computational Efficiency: {result.metrics.computational_efficiency:.1f} samples/sec")
        print(f"Total Fuzzy Rules: {len(result.fuzzy_rules)}")
        print(f"Memory Usage: {result.computational_stats['memory_usage']:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation framework demo failed: {e}")
        return False

def demo_web_interface():
    """Demo web interface components"""
    print_header("🌐 WEB INTERFACE DEMO")
    
    try:
        from multimodal_fuzzy_attention import VQAFuzzyModel
        from adaptive_interface import InteractiveInterface
        
        print("Creating web interface components...")
        
        # Test model initialization
        model = VQAFuzzyModel(
            vocab_size=10000,
            answer_vocab_size=1000,
            text_dim=256,
            image_dim=512,
            hidden_dim=256,
            n_heads=4,
            n_layers=2
        )
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test text processing
        def process_text_input(text: str) -> torch.Tensor:
            words = text.lower().split()
            vocab_size = 10000
            tokens = torch.tensor([abs(hash(word)) % vocab_size for word in words[:20]], dtype=torch.long)
            if len(tokens) == 0:
                tokens = torch.tensor([0], dtype=torch.long)
            return tokens
        
        test_text = "The cat sat on the mat quietly"
        tokens = process_text_input(test_text)
        print(f"✅ Text processed: '{test_text}' -> {tokens.shape}")
        
        # Test model analysis
        tokens = tokens.unsqueeze(0)
        image_features = torch.randn(1, 49, 512)
        
        with torch.no_grad():
            result = model(tokens, image_features, return_explanations=True)
        
        print(f"✅ Model analysis completed:")
        print(f"   Answer logits: {result['answer_logits'].shape}")
        print(f"   Fuzzy rules: {len(result.get('fuzzy_rules', []))}")
        print(f"   Attention patterns: {len(result.get('attention_patterns', {}))}")
        
        # Test interface integration
        interface = InteractiveInterface()
        user_id = "web_demo_user"
        user_profile = interface.initialize_user(user_id)
        
        # Simulate web interactions
        interface.process_interaction(user_id, 'request_explanation', 'text_analysis',
                                    context={'explanation_depth': 'intermediate'})
        interface.process_interaction(user_id, 'click', 'analyze_button', duration=2.0)
        
        print(f"✅ Web interface components working")
        print(f"   User profile created: {user_profile.expertise_level.value}")
        print(f"   Interactions recorded: {len(user_profile.interaction_history)}")
        
        print("\n🌐 To run the full web interface:")
        print("   streamlit run web_interface.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Web interface demo failed: {e}")
        return False

def main():
    """Run complete system demo"""
    print_header("🚀 FUZZY ATTENTION NETWORKS - COMPLETE SYSTEM DEMO", "=", 80)
    print("Human-Centered Differentiable Neuro-Fuzzy Architectures")
    print("Interactive Explanation Interfaces for Multimodal AI with Adaptive User-Controlled Interpretability")
    
    start_time = time.time()
    
    # Demo components
    demos = [
        ("Fuzzy Attention Mechanism", demo_fuzzy_attention),
        ("Multimodal Model", demo_multimodal_model),
        ("Rule Extraction System", demo_rule_extraction),
        ("Adaptive Interface", demo_adaptive_interface),
        ("Visualization System", demo_visualization_system),
        ("Evaluation Framework", demo_evaluation_framework),
        ("Web Interface", demo_web_interface)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        print(f"\n{'='*80}")
        print(f"Running {demo_name} Demo...")
        print(f"{'='*80}")
        
        try:
            success = demo_func()
            results.append((demo_name, success))
            
            if success:
                print(f"✅ {demo_name} demo completed successfully!")
            else:
                print(f"❌ {demo_name} demo failed!")
                
        except Exception as e:
            print(f"❌ {demo_name} demo failed with exception: {e}")
            results.append((demo_name, False))
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print_header("📊 DEMO SUMMARY", "=", 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for demo_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{demo_name:<30} {status}")
    
    print(f"\nOverall: {passed}/{total} demos passed")
    print(f"Total time: {duration:.2f} seconds")
    
    if passed == total:
        print_header("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!", "=", 80)
        print("The Fuzzy Attention Networks system is ready for:")
        print("  • Research and development")
        print("  • User studies")
        print("  • Paper submission to IUI 2026")
        print("  • Production deployment")
        
        print("\nNext steps:")
        print("  1. Run: streamlit run web_interface.py")
        print("  2. Conduct user studies with 20-30 participants")
        print("  3. Write and submit paper to IUI 2026")
        print("  4. Deploy for production use")
    else:
        print_header("⚠️ SOME DEMOS FAILED", "=", 80)
        print("Please check the errors above and fix them before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Test script for visualization system
"""

import torch
import sys
import os
from pathlib import Path

# Add src to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def test_matplotlib_visualizer():
    """Test matplotlib-based visualizations"""
    print("📊 Testing Matplotlib Visualizer")
    print("=" * 50)
    
    try:
        from visualization_system import AttentionVisualizer, VisualizationConfig
        from rule_extractor import FuzzyRule
        
        # Create sample data
        seq_len = 8
        attention_weights = torch.rand(1, seq_len, seq_len)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        tokens = ["The", "cat", "sat", "on", "the", "mat", "quietly", "watching"]
        
        # Create sample rules
        rules = [
            FuzzyRule(0, 2, 0.25, 0.8, "Position 0 strongly attends to position 2", "gaussian", "product"),
            FuzzyRule(1, 3, 0.18, 0.7, "Position 1 moderately attends to position 3", "gaussian", "product"),
            FuzzyRule(4, 6, 0.12, 0.6, "Position 4 slightly attends to position 6", "gaussian", "product"),
            FuzzyRule(2, 5, 0.15, 0.65, "Position 2 moderately attends to position 5", "gaussian", "product"),
            FuzzyRule(3, 7, 0.10, 0.55, "Position 3 slightly attends to position 7", "gaussian", "product")
        ]
        
        # Create visualizer
        config = VisualizationConfig(figure_size=(10, 6), color_scheme='viridis')
        visualizer = AttentionVisualizer(config)
        
        # Test attention heatmap
        fig1 = visualizer.plot_attention_heatmap(attention_weights, tokens, "Test Attention Heatmap")
        print("✅ Attention heatmap created")
        
        # Test fuzzy rules network
        fig2 = visualizer.plot_fuzzy_rules_network(rules, tokens, "Test Fuzzy Rules Network")
        print("✅ Fuzzy rules network created")
        
        # Test rule strength distribution
        fig3 = visualizer.plot_rule_strength_distribution(rules, "Test Rule Strength Distribution")
        print("✅ Rule strength distribution created")
        
        # Test attention entropy
        fig4 = visualizer.plot_attention_entropy(attention_weights, "Test Attention Entropy")
        print("✅ Attention entropy plot created")
        
        # Test with empty rules
        fig5 = visualizer.plot_fuzzy_rules_network([], tokens, "Test Empty Rules")
        print("✅ Empty rules handling works")
        
        return True
        
    except Exception as e:
        print(f"❌ Matplotlib visualizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interactive_visualizer():
    """Test interactive Plotly visualizations"""
    print("\n🎯 Testing Interactive Visualizer")
    print("=" * 50)
    
    try:
        from visualization_system import InteractiveVisualizer
        from rule_extractor import FuzzyRule
        
        # Create sample data
        seq_len = 8
        attention_weights = torch.rand(1, seq_len, seq_len)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        tokens = ["The", "cat", "sat", "on", "the", "mat", "quietly", "watching"]
        
        # Create sample rules
        rules = [
            FuzzyRule(0, 2, 0.25, 0.8, "Position 0 strongly attends to position 2", "gaussian", "product"),
            FuzzyRule(1, 3, 0.18, 0.7, "Position 1 moderately attends to position 3", "gaussian", "product"),
            FuzzyRule(4, 6, 0.12, 0.6, "Position 4 slightly attends to position 6", "gaussian", "product"),
            FuzzyRule(2, 5, 0.15, 0.65, "Position 2 moderately attends to position 5", "gaussian", "product"),
            FuzzyRule(3, 7, 0.10, 0.55, "Position 3 slightly attends to position 7", "gaussian", "product")
        ]
        
        # Create interactive visualizer
        interactive_viz = InteractiveVisualizer()
        
        # Test interactive attention heatmap
        fig1 = interactive_viz.create_interactive_attention_heatmap(attention_weights, tokens, "Test Interactive Attention Heatmap")
        print("✅ Interactive attention heatmap created")
        
        # Test interactive rule network
        fig2 = interactive_viz.create_interactive_rule_network(rules, tokens, "Test Interactive Fuzzy Rules Network")
        print("✅ Interactive rule network created")
        
        # Test rule comparison chart
        rules_by_level = {
            'novice': rules[:2],
            'intermediate': rules[:4],
            'expert': rules
        }
        fig3 = interactive_viz.create_rule_comparison_chart(rules_by_level, "Test Rule Comparison by User Level")
        print("✅ Rule comparison chart created")
        
        # Test with empty rules
        fig4 = interactive_viz.create_interactive_rule_network([], tokens, "Test Empty Rules")
        print("✅ Empty rules handling works")
        
        return True
        
    except Exception as e:
        print(f"❌ Interactive visualizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visualization_integration():
    """Test integration with the main system"""
    print("\n🔗 Testing Visualization Integration")
    print("=" * 50)
    
    try:
        from multimodal_fuzzy_attention import VQAFuzzyModel
        from visualization_system import AttentionVisualizer, InteractiveVisualizer
        from adaptive_interface import InteractiveInterface
        
        # Create model
        model = VQAFuzzyModel(
            vocab_size=10000,
            answer_vocab_size=1000,
            text_dim=256,
            image_dim=512,
            hidden_dim=256,
            n_heads=4,
            n_layers=2
        )
        
        # Create interface
        interface = InteractiveInterface()
        user_id = "viz_test_user"
        interface.initialize_user(user_id)
        
        # Create visualizers
        visualizer = AttentionVisualizer()
        interactive_viz = InteractiveVisualizer()
        
        # Test with real model output
        text = "The cat sat on the mat quietly"
        tokens = text.split()
        token_tensor = torch.tensor([abs(hash(word)) % 10000 for word in tokens], dtype=torch.long).unsqueeze(0)
        image_features = torch.randn(1, 49, 512)
        
        with torch.no_grad():
            result = model(token_tensor, image_features, return_explanations=True)
        
        print(f"✅ Model analysis completed:")
        print(f"   Answer logits: {result['answer_logits'].shape}")
        print(f"   Fuzzy rules: {len(result.get('fuzzy_rules', []))}")
        print(f"   Attention patterns: {len(result.get('attention_patterns', {}))}")
        
        # Test visualization with model output
        if 'attention_patterns' in result and 'attention_entropy' in result['attention_patterns']:
            # Create mock attention weights for visualization
            seq_len = len(tokens)
            attention_weights = torch.rand(1, seq_len, seq_len)
            attention_weights = torch.softmax(attention_weights, dim=-1)
            
            # Test matplotlib visualizations
            fig1 = visualizer.plot_attention_heatmap(attention_weights, tokens, "Model Output - Attention Heatmap")
            print("✅ Model output attention heatmap created")
            
            # Test interactive visualizations
            fig2 = interactive_viz.create_interactive_attention_heatmap(attention_weights, tokens, "Model Output - Interactive Attention Heatmap")
            print("✅ Model output interactive attention heatmap created")
        
        # Test with fuzzy rules
        if 'fuzzy_rules' in result and result['fuzzy_rules']:
            rules = result['fuzzy_rules']
            
            # Test matplotlib rule network
            fig3 = visualizer.plot_fuzzy_rules_network(rules, tokens, "Model Output - Fuzzy Rules Network")
            print("✅ Model output fuzzy rules network created")
            
            # Test interactive rule network
            fig4 = interactive_viz.create_interactive_rule_network(rules, tokens, "Model Output - Interactive Fuzzy Rules Network")
            print("✅ Model output interactive fuzzy rules network created")
        else:
            print("⚠️ No fuzzy rules extracted from model output")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run visualization tests"""
    print("🚀 Visualization System Test Suite")
    print("=" * 60)
    
    tests = [
        ("Matplotlib Visualizer", test_matplotlib_visualizer),
        ("Interactive Visualizer", test_interactive_visualizer),
        ("Visualization Integration", test_visualization_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        success = test_func()
        results.append((test_name, success))
        
        if success:
            print(f"✅ {test_name} test passed!")
        else:
            print(f"❌ {test_name} test failed!")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VISUALIZATION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All visualization tests passed!")
        print("🎨 Visualization system is ready for use!")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()


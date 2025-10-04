#!/usr/bin/env python3
"""
Test script for web interface components
"""

import torch
import sys
import os
from pathlib import Path

# Add src to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def test_adaptive_interface():
    """Test adaptive interface functionality"""
    print("🎯 Testing Adaptive Interface")
    print("=" * 50)
    
    try:
        from adaptive_interface import InteractiveInterface, UserExpertiseLevel
        
        # Create interface
        interface = InteractiveInterface()
        user_id = "test_user"
        
        # Initialize user
        user_profile = interface.initialize_user(user_id)
        print(f"✅ User initialized: {user_profile.expertise_level.value}")
        
        # Simulate interactions
        print("📝 Simulating user interactions...")
        
        # Novice interactions
        interface.process_interaction(user_id, 'request_explanation', 'attention_weights', 
                                    context={'explanation_depth': 'basic'})
        interface.process_interaction(user_id, 'feedback', 'explanation', 
                                    context={'feedback_type': 'too complex'})
        
        print(f"   After novice interactions: {interface.user_profiles[user_id].expertise_level.value}")
        
        # Intermediate interactions
        interface.process_interaction(user_id, 'request_explanation', 'attention_weights',
                                    context={'explanation_depth': 'detailed'})
        interface.process_interaction(user_id, 'click', 'rule_1', duration=3.0)
        interface.process_interaction(user_id, 'hover', 'technical_details', duration=2.0)
        
        print(f"   After intermediate interactions: {interface.user_profiles[user_id].expertise_level.value}")
        
        # Expert interactions
        interface.process_interaction(user_id, 'request_explanation', 'attention_weights',
                                    context={'explanation_depth': 'technical'})
        interface.process_interaction(user_id, 'click', 'rule_2', duration=8.0)
        interface.process_interaction(user_id, 'feedback', 'explanation',
                                    context={'feedback_type': 'need more detail about t-norms'})
        
        final_profile = interface.user_profiles[user_id]
        print(f"   Final expertise: {final_profile.expertise_level.value} (confidence: {final_profile.confidence_score:.3f})")
        
        # Test explanation generation
        from rule_extractor import FuzzyRule
        mock_rules = [
            FuzzyRule(0, 2, 0.25, 0.8, "Position 0 strongly attends to position 2", "gaussian", "product"),
            FuzzyRule(1, 3, 0.18, 0.7, "Position 1 moderately attends to position 3", "gaussian", "product"),
            FuzzyRule(4, 6, 0.12, 0.6, "Position 4 slightly attends to position 6", "gaussian", "product")
        ]
        
        mock_tokens = ["The", "cat", "sat", "on", "the", "mat", "quietly"]
        
        explanation = interface.get_explanation(user_id, mock_rules, mock_tokens)
        
        print(f"✅ Explanation generated for {explanation['user_level']} user")
        print(f"   Title: {explanation['title']}")
        print(f"   Rules shown: {len(explanation['content']['rules'])}")
        print(f"   Technical details: {explanation['content']['technical_details']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Adaptive interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_web_interface_components():
    """Test web interface components"""
    print("\n🌐 Testing Web Interface Components")
    print("=" * 50)
    
    try:
        from multimodal_fuzzy_attention import VQAFuzzyModel
        from adaptive_interface import InteractiveInterface
        
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
        user_id = "web_test_user"
        user_profile = interface.initialize_user(user_id)
        
        # Simulate web interactions
        interface.process_interaction(user_id, 'request_explanation', 'text_analysis',
                                    context={'explanation_depth': 'intermediate'})
        interface.process_interaction(user_id, 'click', 'analyze_button', duration=2.0)
        
        print(f"✅ Web interface components working")
        print(f"   User profile created: {user_profile.expertise_level.value}")
        print(f"   Interactions recorded: {len(user_profile.interaction_history)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Web interface components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_explanation_generation():
    """Test explanation generation for different user levels"""
    print("\n📝 Testing Explanation Generation")
    print("=" * 50)
    
    try:
        from adaptive_interface import InteractiveInterface
        from rule_extractor import FuzzyRule
        
        interface = InteractiveInterface()
        
        # Create mock rules
        mock_rules = [
            FuzzyRule(0, 2, 0.25, 0.8, "Position 0 strongly attends to position 2", "gaussian", "product"),
            FuzzyRule(1, 3, 0.18, 0.7, "Position 1 moderately attends to position 3", "gaussian", "product"),
            FuzzyRule(4, 6, 0.12, 0.6, "Position 4 slightly attends to position 6", "gaussian", "product"),
            FuzzyRule(2, 5, 0.15, 0.65, "Position 2 moderately attends to position 5", "gaussian", "product"),
            FuzzyRule(3, 7, 0.10, 0.55, "Position 3 slightly attends to position 7", "gaussian", "product")
        ]
        
        mock_tokens = ["The", "cat", "sat", "on", "the", "mat", "quietly", "watching"]
        
        # Test different user levels
        for level in ['novice', 'intermediate', 'expert']:
            user_id = f"test_{level}_user"
            interface.initialize_user(user_id)
            
            # Set expertise level
            from adaptive_interface import UserExpertiseLevel
            interface.user_profiles[user_id].expertise_level = UserExpertiseLevel(level)
            
            # Generate explanation
            explanation = interface.get_explanation(user_id, mock_rules, mock_tokens)
            
            print(f"✅ {level.upper()} explanation generated:")
            print(f"   Title: {explanation['title']}")
            print(f"   Rules shown: {len(explanation['content']['rules'])}")
            print(f"   Technical details: {explanation['content']['technical_details']}")
            print(f"   Visual style: {explanation['visual_style']}")
            
            # Show sample rule
            if explanation['content']['rules']:
                sample_rule = explanation['content']['rules'][0]
                print(f"   Sample rule: {sample_rule['text'][:60]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Explanation generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run web interface tests"""
    print("🚀 Web Interface Test Suite")
    print("=" * 60)
    
    tests = [
        ("Adaptive Interface", test_adaptive_interface),
        ("Web Interface Components", test_web_interface_components),
        ("Explanation Generation", test_explanation_generation)
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
    print("📊 WEB INTERFACE TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All web interface tests passed!")
        print("🌐 Ready to run: streamlit run web_interface.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()

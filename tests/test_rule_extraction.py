#!/usr/bin/env python3
"""
Test script for improved rule extraction
"""

import torch
import sys
import os
from pathlib import Path

# Add src to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def test_rule_extraction():
    """Test rule extraction functionality"""
    print("🔍 Testing Rule Extraction")
    print("=" * 50)
    
    try:
        from rule_extractor import RuleExtractor, AdaptiveRuleExplainer
        
        # Create sample attention weights with cross-modal structure
        batch_size, text_len, image_len = 1, 8, 16
        seq_len = text_len + image_len
        attention_weights = torch.rand(batch_size, seq_len, seq_len)
        
        # Add some strong cross-modal connections
        attention_weights[0, 2, text_len + 3] = 0.25  # Text to image
        attention_weights[0, 5, text_len + 7] = 0.22  # Text to image
        attention_weights[0, text_len + 1, 3] = 0.20  # Image to text
        attention_weights[0, text_len + 5, 6] = 0.18  # Image to text
        
        # Add some within-modal connections
        attention_weights[0, 1, 4] = 0.15  # Text to text
        attention_weights[0, text_len + 2, text_len + 8] = 0.12  # Image to image
        
        # Normalize to make it look like attention weights
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        # Create rule extractor
        extractor = RuleExtractor(attention_threshold=0.05, strong_threshold=0.1)
        
        # Extract rules
        rules = extractor.extract_rules(attention_weights)
        
        print(f"✅ Extracted {len(rules)} rules from attention weights")
        print(f"📈 Attention shape: {attention_weights.shape}")
        print(f"📊 Text length: {text_len}, Image length: {image_len}")
        
        # Show top rules
        print(f"\n🔍 Top Rules:")
        for i, rule in enumerate(rules[:5]):
            print(f"   {i+1}. {rule.linguistic_description}")
            print(f"      Strength: {rule.strength:.3f}, Confidence: {rule.confidence:.3f}")
            print(f"      From: {rule.from_position}, To: {rule.to_position}")
        
        # Test adaptive explanations
        explainer = AdaptiveRuleExplainer()
        
        for user_level in ['novice', 'intermediate', 'expert']:
            print(f"\n👤 {user_level.upper()} EXPLANATION:")
            print("-" * 40)
            explanation = explainer.generate_explanation(rules, user_level)
            print(explanation)
        
        # Extract attention patterns
        patterns = extractor.extract_attention_patterns(attention_weights)
        print(f"\n📊 ATTENTION PATTERNS:")
        print(f"   Average entropy: {patterns['attention_entropy'][0]:.3f}")
        print(f"   Sparsity: {patterns['attention_sparsity'][0]:.3f}")
        print(f"   Dominant connections: {patterns['dominant_connections'][0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Rule extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multimodal_rule_extraction():
    """Test rule extraction with multimodal model"""
    print("\n🖼️ Testing Multimodal Rule Extraction")
    print("=" * 50)
    
    try:
        from multimodal_fuzzy_attention import VQAFuzzyModel
        
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
        
        # Create test data
        batch_size, question_len, image_patches = 2, 8, 16
        question_tokens = torch.randint(0, 10000, (batch_size, question_len))
        image_features = torch.randn(batch_size, image_patches, 512)
        
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        print(f"📊 Input shapes: {question_tokens.shape}, {image_features.shape}")
        
        # Forward pass with rule extraction
        with torch.no_grad():
            result = model(question_tokens, image_features, return_explanations=True)
        
        print(f"✅ Forward pass successful:")
        print(f"   Answer logits: {result['answer_logits'].shape}")
        
        if 'fuzzy_rules' in result:
            rules = result['fuzzy_rules']
            print(f"   Fuzzy rules extracted: {len(rules)}")
            
            # Show top rules
            print(f"\n🔍 Top Rules:")
            for i, rule in enumerate(rules[:3]):
                print(f"   {i+1}. {rule.linguistic_description}")
                print(f"      Strength: {rule.strength:.3f}, Confidence: {rule.confidence:.3f}")
        
        if 'attention_patterns' in result:
            patterns = result['attention_patterns']
            print(f"\n📊 Attention Patterns:")
            if 'attention_entropy' in patterns:
                print(f"   Average entropy: {sum(patterns['attention_entropy'])/len(patterns['attention_entropy']):.3f}")
            if 'attention_sparsity' in patterns:
                print(f"   Average sparsity: {sum(patterns['attention_sparsity'])/len(patterns['attention_sparsity']):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multimodal rule extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run rule extraction tests"""
    print("🚀 Rule Extraction Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Rule Extraction", test_rule_extraction),
        ("Multimodal Rule Extraction", test_multimodal_rule_extraction)
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
    print("📊 RULE EXTRACTION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All rule extraction tests passed!")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()


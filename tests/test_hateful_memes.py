#!/usr/bin/env python3
"""
Test script for Hateful Memes dataset integration
"""

import torch
import sys
import os
from pathlib import Path

# Add src to Python path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Add experiments to path
experiments_path = os.path.join(os.path.dirname(__file__), '..', 'experiments')
if experiments_path not in sys.path:
    sys.path.insert(0, experiments_path)

def test_hateful_memes_dataset():
    """Test Hateful Memes dataset loading and processing"""
    print("🧪 Testing Hateful Memes Dataset")
    print("=" * 50)
    
    try:
        from evaluation_framework import HatefulMemesLocalDataset
        
        # Load dataset
        dataset = HatefulMemesLocalDataset(
            root_dir='./data/hateful_memes',
            split='train',
            max_samples=10,
            device='cpu'
        )
        
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
        # Test all samples
        for i in range(len(dataset)):
            sample = dataset[i]
            print(f"\n📝 Sample {i+1}:")
            print(f"   Text: {sample['question']}")
            print(f"   Label: {sample['label']} ({'Hateful' if sample['label'] == 1 else 'Non-hateful'})")
            print(f"   Question tokens: {sample['question_tokens'].shape}")
            print(f"   Image features: {sample['image_features'].shape}")
        
        # Statistics
        labels = [dataset[i]['label'] for i in range(len(dataset))]
        hateful_count = sum(labels)
        non_hateful_count = len(labels) - hateful_count
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(dataset)}")
        print(f"   Hateful memes: {hateful_count}")
        print(f"   Non-hateful memes: {non_hateful_count}")
        print(f"   Hateful ratio: {hateful_count/len(dataset):.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hateful Memes dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_with_hateful_memes():
    """Test model inference with Hateful Memes data"""
    print("\n🤖 Testing Model with Hateful Memes")
    print("=" * 50)
    
    try:
        from evaluation_framework import HatefulMemesLocalDataset
        from multimodal_fuzzy_attention import VQAFuzzyModel
        from adaptive_interface import InteractiveInterface
        
        # Load dataset
        dataset = HatefulMemesLocalDataset(
            root_dir='./data/hateful_memes',
            split='train',
            max_samples=5,
            device='cpu'
        )
        
        # Create model
        model = VQAFuzzyModel(
            vocab_size=10000,
            answer_vocab_size=1000,
            text_dim=256,
            image_dim=2048,  # ResNet features
            hidden_dim=256,
            n_heads=4,
            n_layers=2
        )
        
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test inference on each sample
        for i in range(len(dataset)):
            sample = dataset[i]
            question_tokens = sample['question_tokens'].unsqueeze(0)
            image_features = sample['image_features'].unsqueeze(0)
            
            with torch.no_grad():
                result = model(question_tokens, image_features, return_explanations=True)
            
            print(f"\n📝 Sample {i+1}: '{sample['question'][:50]}...'")
            print(f"   Label: {sample['label']} ({'Hateful' if sample['label'] == 1 else 'Non-hateful'})")
            print(f"   Answer logits shape: {result['answer_logits'].shape}")
            print(f"   Fuzzy rules: {len(result.get('fuzzy_rules', []))}")
            print(f"   Attention patterns: {list(result.get('attention_patterns', {}).keys())}")
            
            # Show top predictions
            probs = torch.softmax(result['answer_logits'], dim=-1)
            top_probs, top_indices = torch.topk(probs, 3)
            print(f"   Top predictions: {[(idx.item(), prob.item()) for idx, prob in zip(top_indices[0], top_probs[0])]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test with Hateful Memes failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_adaptive_explanations():
    """Test adaptive explanations with Hateful Memes data"""
    print("\n🎯 Testing Adaptive Explanations")
    print("=" * 50)
    
    try:
        from evaluation_framework import HatefulMemesLocalDataset
        from multimodal_fuzzy_attention import VQAFuzzyModel
        from adaptive_interface import InteractiveInterface
        
        # Load dataset
        dataset = HatefulMemesLocalDataset(
            root_dir='./data/hateful_memes',
            split='train',
            max_samples=3,
            device='cpu'
        )
        
        # Create model and interface
        model = VQAFuzzyModel(
            vocab_size=10000,
            answer_vocab_size=1000,
            text_dim=256,
            image_dim=2048,
            hidden_dim=256,
            n_heads=4,
            n_layers=2
        )
        
        interface = InteractiveInterface()
        
        # Test different user levels
        for user_level in ['novice', 'intermediate', 'expert']:
            print(f"\n👤 Testing {user_level.upper()} explanations:")
            
            user_id = f'hateful_memes_{user_level}'
            interface.initialize_user(user_id)
            
            # Set expertise level
            from adaptive_interface import UserExpertiseLevel
            interface.user_profiles[user_id].expertise_level = UserExpertiseLevel(user_level)
            
            # Test with first sample
            sample = dataset[0]
            question_tokens = sample['question_tokens'].unsqueeze(0)
            image_features = sample['image_features'].unsqueeze(0)
            
            with torch.no_grad():
                result = model(question_tokens, image_features, return_explanations=True)
            
            # Generate explanation
            rules = result.get('fuzzy_rules', [])
            tokens = sample['question'].split()
            attention_patterns = result.get('attention_patterns', {})
            
            explanation = interface.get_explanation(user_id, rules, tokens, attention_patterns)
            
            print(f"   Title: {explanation['title']}")
            print(f"   User level: {explanation['user_level']}")
            print(f"   Rules shown: {len(explanation['content']['rules'])}")
            print(f"   Technical details: {explanation['content']['technical_details']}")
            
            if explanation['content']['rules']:
                print(f"   Sample rule: {explanation['content']['rules'][0]['text'][:60]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Adaptive explanations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_with_hateful_memes():
    """Test evaluation framework with Hateful Memes data"""
    print("\n📊 Testing Evaluation with Hateful Memes")
    print("=" * 50)
    
    try:
        from evaluation_framework import FuzzyAttentionEvaluator, HatefulMemesLocalDataset
        from multimodal_fuzzy_attention import VQAFuzzyModel
        
        # Create evaluator
        evaluator = FuzzyAttentionEvaluator(device='cpu')
        
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
        
        # Load dataset
        dataset = HatefulMemesLocalDataset(
            root_dir='./data/hateful_memes',
            split='train',
            max_samples=5,
            device='cpu'
        )
        
        print(f"✅ Components created:")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Dataset size: {len(dataset)}")
        
        # Run evaluation
        result = evaluator.evaluate_model(model, dataset, 'HatefulMemes')
        
        print(f"\n📊 Evaluation Results:")
        print(f"   Task Accuracy: {result.metrics.task_accuracy:.3f}")
        print(f"   Attention Entropy: {result.metrics.attention_entropy:.3f}")
        print(f"   Attention Sparsity: {result.metrics.attention_sparsity:.3f}")
        print(f"   Rule Consistency: {result.metrics.rule_consistency:.3f}")
        print(f"   Explanation Quality: {result.metrics.explanation_quality:.3f}")
        print(f"   Computational Efficiency: {result.metrics.computational_efficiency:.1f} samples/sec")
        print(f"   Total Fuzzy Rules: {len(result.fuzzy_rules)}")
        print(f"   Memory Usage: {result.computational_stats['memory_usage']:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run Hateful Memes tests"""
    print("🚀 Hateful Memes Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Hateful Memes Dataset", test_hateful_memes_dataset),
        ("Model with Hateful Memes", test_model_with_hateful_memes),
        ("Adaptive Explanations", test_adaptive_explanations),
        ("Evaluation with Hateful Memes", test_evaluation_with_hateful_memes)
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
    print("📊 HATEFUL MEMES TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Hateful Memes tests passed!")
        print("📊 System is ready for real-world testing!")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()


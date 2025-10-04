#!/usr/bin/env python3
"""
Test script for Fuzzy Attention Networks system
"""

import torch
import sys
import os
from pathlib import Path

# Add src to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def test_basic_functionality():
    """Test basic fuzzy attention functionality"""
    print("🧪 Testing Basic Functionality")
    print("=" * 50)
    
    try:
        from fuzzy_attention import MultiHeadFuzzyAttention
        from utils import FuzzyOperators
        
        # Test fuzzy attention
        d_model, n_heads, seq_len = 128, 4, 10
        fuzzy_layer = MultiHeadFuzzyAttention(d_model, n_heads)
        
        # Create test data
        test_input = torch.randn(1, seq_len, d_model)
        
        # Forward pass
        with torch.no_grad():
            output, attention_info = fuzzy_layer(
                test_input, test_input, test_input, 
                return_attention=True
            )
        
        print(f"✅ Fuzzy Attention Test:")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Attention shape: {attention_info['avg_attention'].shape}")
        
        # Test fuzzy operators
        a, b = torch.tensor(0.7), torch.tensor(0.3)
        print(f"✅ Fuzzy Operators Test:")
        print(f"   Product t-norm: {FuzzyOperators.product_tnorm(a, b):.3f}")
        print(f"   Minimum t-norm: {FuzzyOperators.minimum_tnorm(a, b):.3f}")
        print(f"   Łukasiewicz t-norm: {FuzzyOperators.lukasiewicz_tnorm(a, b):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_multimodal_model():
    """Test multimodal fuzzy attention model"""
    print("\n🖼️ Testing Multimodal Model")
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
        
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create test data
        batch_size, question_len, image_patches = 2, 10, 49
        question_tokens = torch.randint(0, 10000, (batch_size, question_len))
        image_features = torch.randn(batch_size, image_patches, 512)
        
        print(f"✅ Test data created:")
        print(f"   Question tokens: {question_tokens.shape}")
        print(f"   Image features: {image_features.shape}")
        
        # Forward pass
        with torch.no_grad():
            result = model(question_tokens, image_features, return_explanations=True)
        
        print(f"✅ Forward pass successful:")
        print(f"   Answer logits: {result['answer_logits'].shape}")
        print(f"   Answer probabilities: {result['answer_probs'].shape}")
        
        if 'fuzzy_rules' in result:
            print(f"   Fuzzy rules extracted: {len(result['fuzzy_rules'])}")
        
        if 'explanations' in result:
            print(f"   Explanations generated: {len(result['explanations']['reasoning_steps'])} steps")
        
        return True
        
    except Exception as e:
        print(f"❌ Multimodal model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_loading():
    """Test dataset loading functionality"""
    print("\n📊 Testing Dataset Loading")
    print("=" * 50)
    
    try:
        from experiments.evaluation_framework import HatefulMemesLocalDataset, BeansHFDataset
        
        # Test HatefulMemes dataset
        print("Testing HatefulMemes dataset...")
        hateful_dataset = HatefulMemesLocalDataset(
            root_dir='./data/hateful_memes',
            split='train',
            max_samples=5,
            device='cpu'
        )
        
        print(f"✅ HatefulMemes dataset loaded: {len(hateful_dataset)} samples")
        
        if len(hateful_dataset) > 0:
            sample = hateful_dataset[0]
            print(f"   Sample keys: {list(sample.keys())}")
            print(f"   Question: {sample['question']}")
            print(f"   Label: {sample['label']}")
            print(f"   Question tokens shape: {sample['question_tokens'].shape}")
            print(f"   Image features shape: {sample['image_features'].shape}")
        
        # Test Beans dataset
        print("\nTesting Beans dataset...")
        beans_dataset = BeansHFDataset(
            split='train',
            max_samples=5,
            device='cpu'
        )
        
        print(f"✅ Beans dataset loaded: {len(beans_dataset)} samples")
        
        if len(beans_dataset) > 0:
            sample = beans_dataset[0]
            print(f"   Sample keys: {list(sample.keys())}")
            print(f"   Question: {sample['question']}")
            print(f"   Answer: {sample['answer']}")
            print(f"   Question tokens shape: {sample['question_tokens'].shape}")
            print(f"   Image features shape: {sample['image_features'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation():
    """Test evaluation framework"""
    print("\n📈 Testing Evaluation Framework")
    print("=" * 50)
    
    try:
        from experiments.evaluation_framework import FuzzyAttentionEvaluator, BeansHFDataset
        from multimodal_fuzzy_attention import VQAFuzzyModel
        
        # Create evaluator
        evaluator = FuzzyAttentionEvaluator(device='cpu')
        
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
        
        # Create dataset
        dataset = BeansHFDataset(split='train', max_samples=5, device='cpu')
        
        print(f"✅ Components created:")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"   Dataset size: {len(dataset)}")
        
        # Run evaluation
        result = evaluator.evaluate_model(model, dataset, 'Beans')
        
        print(f"✅ Evaluation completed:")
        print(f"   Task accuracy: {result.metrics.task_accuracy:.3f}")
        print(f"   Attention entropy: {result.metrics.attention_entropy:.3f}")
        print(f"   Computational efficiency: {result.metrics.computational_efficiency:.1f} samples/sec")
        print(f"   Fuzzy rules extracted: {len(result.fuzzy_rules)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Fuzzy Attention Networks System Test")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Multimodal Model", test_multimodal_model),
        ("Dataset Loading", test_dataset_loading),
        ("Evaluation Framework", test_evaluation)
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
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for development.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()


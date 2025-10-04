#!/usr/bin/env python3
"""
Final Project Check - Comprehensive validation of the entire system
"""

import os
import sys
import torch
import json
from pathlib import Path
import subprocess

def check_file_structure():
    """Check if all required files exist"""
    print("🔍 Checking file structure...")
    
    required_files = [
        'src/fuzzy_attention.py',
        'src/multimodal_fuzzy_attention.py',
        'src/rule_extractor.py',
        'src/adaptive_interface.py',
        'src/visualization_system.py',
        'src/utils.py',
        'src/config.py',
        'experiments/evaluation_framework.py',
        'demos/web_interface.py',
        'demos/demo_complete_system.py',
        'tests/test_hateful_memes.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def check_dataset():
    """Check if dataset is properly loaded"""
    print("\n📊 Checking dataset...")
    
    data_dir = Path('./data/hateful_memes')
    jsonl_file = data_dir / 'train.jsonl'
    img_dir = data_dir / 'img'
    
    if not jsonl_file.exists():
        print("❌ Dataset JSONL file not found")
        return False
    
    # Count samples
    with open(jsonl_file, 'r') as f:
        lines = f.readlines()
    
    print(f"   JSONL entries: {len(lines)}")
    
    # Count images
    img_files = list(img_dir.glob('*.png'))
    print(f"   Image files: {len(img_files)}")
    
    # Check distribution
    labels = []
    for line in lines:
        data = json.loads(line)
        labels.append(data['label'])
    
    hateful_count = sum(labels)
    non_hateful_count = len(labels) - hateful_count
    
    print(f"   Hateful memes: {hateful_count}")
    print(f"   Non-hateful memes: {non_hateful_count}")
    print(f"   Hateful ratio: {hateful_count/len(labels):.1%}")
    
    if len(lines) >= 40 and len(img_files) >= 40:
        print("✅ Dataset looks good")
        return True
    else:
        print("❌ Dataset insufficient")
        return False

def check_imports():
    """Check if all modules can be imported"""
    print("\n📦 Checking imports...")
    
    # Add src to path
    src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    try:
        from fuzzy_attention import MultiHeadFuzzyAttention
        print("✅ fuzzy_attention imported")
        
        from multimodal_fuzzy_attention import VQAFuzzyModel
        print("✅ multimodal_fuzzy_attention imported")
        
        from rule_extractor import RuleExtractor
        print("✅ rule_extractor imported")
        
        from adaptive_interface import InteractiveInterface
        print("✅ adaptive_interface imported")
        
        from visualization_system import AttentionVisualizer
        print("✅ visualization_system imported")
        
        from utils import FuzzyOperators
        print("✅ utils imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def check_model_creation():
    """Check if models can be created"""
    print("\n🤖 Checking model creation...")
    
    try:
        from multimodal_fuzzy_attention import VQAFuzzyModel
        
        model = VQAFuzzyModel(
            vocab_size=10000,
            answer_vocab_size=1000,
            text_dim=256,
            image_dim=2048,
            hidden_dim=256,
            n_heads=4,
            n_layers=2
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created with {param_count:,} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def check_dataset_loading():
    """Check if dataset can be loaded"""
    print("\n📊 Checking dataset loading...")
    
    try:
        # Add src to path
        src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Add experiments to path
        experiments_path = os.path.join(os.path.dirname(__file__), '..', 'experiments')
        if experiments_path not in sys.path:
            sys.path.insert(0, experiments_path)
        
        from evaluation_framework import HatefulMemesLocalDataset
        
        dataset = HatefulMemesLocalDataset(
            root_dir='./data/hateful_memes',
            split='train',
            max_samples=5,
            device='cpu'
        )
        
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
        # Test first sample
        sample = dataset[0]
        print(f"   Sample text: {sample['question'][:50]}...")
        print(f"   Label: {sample['label']}")
        print(f"   Question tokens: {sample['question_tokens'].shape}")
        print(f"   Image features: {sample['image_features'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading error: {e}")
        return False

def check_model_inference():
    """Check if model can run inference"""
    print("\n🔮 Checking model inference...")
    
    try:
        # Add src to path
        src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from multimodal_fuzzy_attention import VQAFuzzyModel
        
        # Add experiments to path
        experiments_path = os.path.join(os.path.dirname(__file__), '..', 'experiments')
        if experiments_path not in sys.path:
            sys.path.insert(0, experiments_path)
        
        from evaluation_framework import HatefulMemesLocalDataset
        
        # Load model and dataset
        model = VQAFuzzyModel(
            vocab_size=10000,
            answer_vocab_size=1000,
            text_dim=256,
            image_dim=2048,
            hidden_dim=256,
            n_heads=4,
            n_layers=2
        )
        
        dataset = HatefulMemesLocalDataset(
            root_dir='./data/hateful_memes',
            split='train',
            max_samples=3,
            device='cpu'
        )
        
        # Test inference
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            question_tokens = sample['question_tokens'].unsqueeze(0)
            image_features = sample['image_features'].unsqueeze(0)
            
            with torch.no_grad():
                result = model(question_tokens, image_features, return_explanations=True)
            
            print(f"   Sample {i+1}: {sample['question'][:40]}...")
            print(f"     Answer logits: {result['answer_logits'].shape}")
            print(f"     Fuzzy rules: {len(result.get('fuzzy_rules', []))}")
            print(f"     Attention patterns: {len(result.get('attention_patterns', {}))}")
        
        print("✅ Model inference working")
        return True
        
    except Exception as e:
        print(f"❌ Model inference error: {e}")
        return False

def check_web_interface():
    """Check if web interface can be imported"""
    print("\n🌐 Checking web interface...")
    
    try:
        import streamlit
        print("✅ Streamlit available")
        
        # Add src to path
        src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Add experiments to path
        experiments_path = os.path.join(os.path.dirname(__file__), '..', 'experiments')
        if experiments_path not in sys.path:
            sys.path.insert(0, experiments_path)
        
        # Check if web_interface.py can be imported
        import importlib.util
        spec = importlib.util.spec_from_file_location("web_interface", "demos/web_interface.py")
        web_interface = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(web_interface)
        
        print("✅ Web interface module loads")
        return True
        
    except Exception as e:
        print(f"❌ Web interface error: {e}")
        return False

def run_tests():
    """Run all test suites"""
    print("\n🧪 Running test suites...")
    
    test_scripts = [
        'tests/test_system.py',
        'tests/test_rule_extraction.py',
        'tests/test_visualization.py',
        'tests/test_web_interface.py',
        'tests/test_hateful_memes.py'
    ]
    
    results = []
    
    for test_script in test_scripts:
        if Path(test_script).exists():
            print(f"   Running {test_script}...")
            try:
                result = subprocess.run([sys.executable, test_script], 
                                      capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print(f"   ✅ {test_script} passed")
                    results.append(True)
                else:
                    print(f"   ❌ {test_script} failed")
                    print(f"      Error: {result.stderr[:100]}...")
                    results.append(False)
            except subprocess.TimeoutExpired:
                print(f"   ⏰ {test_script} timed out")
                results.append(False)
            except Exception as e:
                print(f"   ❌ {test_script} error: {e}")
                results.append(False)
        else:
            print(f"   ⚠️ {test_script} not found")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"   Test results: {passed}/{total} passed")
    return passed == total

def check_requirements():
    """Check if all requirements are met"""
    print("\n📋 Checking requirements...")
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.readlines()
        
        print(f"   Requirements file has {len(requirements)} packages")
        
        # Check key packages
        key_packages = ['torch', 'numpy', 'datasets', 'streamlit', 'matplotlib', 'plotly']
        
        for package in key_packages:
            try:
                __import__(package)
                print(f"   ✅ {package} available")
            except ImportError:
                print(f"   ❌ {package} not available")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Requirements check error: {e}")
        return False

def main():
    """Run complete project check"""
    print("🚀 FUZZY ATTENTION NETWORKS - FINAL PROJECT CHECK")
    print("=" * 60)
    
    checks = [
        ("File Structure", check_file_structure),
        ("Dataset", check_dataset),
        ("Imports", check_imports),
        ("Model Creation", check_model_creation),
        ("Dataset Loading", check_dataset_loading),
        ("Model Inference", check_model_inference),
        ("Web Interface", check_web_interface),
        ("Requirements", check_requirements),
        ("Test Suites", run_tests)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n{'='*60}")
        print(f"Running {check_name} Check...")
        print(f"{'='*60}")
        
        try:
            success = check_func()
            results.append((check_name, success))
            
            if success:
                print(f"✅ {check_name} check passed!")
            else:
                print(f"❌ {check_name} check failed!")
                
        except Exception as e:
            print(f"❌ {check_name} check failed with exception: {e}")
            results.append((check_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 FINAL PROJECT CHECK SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for check_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{check_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 ALL CHECKS PASSED!")
        print("🚀 Project is ready for:")
        print("   • User studies")
        print("   • Paper submission to IUI 2026")
        print("   • Production deployment")
        print("   • Further development")
    else:
        print(f"\n⚠️ {total - passed} checks failed.")
        print("Please fix the issues above before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    main()


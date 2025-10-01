"""
Evaluation Framework for Fuzzy Attention Networks
Benchmarks on VQA-X, e-SNLI-VE, and other interpretability datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import time
from pathlib import Path

from src.multimodal_fuzzy_attention import VQAFuzzyModel, MultimodalFuzzyTransformer
from src.rule_extractor import RuleExtractor, FuzzyRule
from src.adaptive_interface import InteractiveInterface, UserExpertiseLevel

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    task_accuracy: float
    attention_entropy: float
    attention_sparsity: float
    rule_consistency: float
    explanation_quality: float
    user_satisfaction: float
    computational_efficiency: float

@dataclass
class EvaluationResult:
    """Container for complete evaluation results"""
    model_name: str
    dataset_name: str
    metrics: EvaluationMetrics
    fuzzy_rules: List[FuzzyRule]
    attention_patterns: Dict[str, Any]
    user_study_results: Optional[Dict[str, Any]]
    computational_stats: Dict[str, float]

class VQAXDataset(Dataset):
    """VQA-X dataset for visual question answering with explanations"""
    
    def __init__(self, data_path: str, split: str = 'train', max_samples: int = 1000):
        self.data_path = data_path
        self.split = split
        self.max_samples = max_samples
        
        # Load data (mock implementation - replace with actual VQA-X loading)
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load VQA-X dataset (mock implementation)"""
        # In real implementation, load from VQA-X files
        mock_data = []
        for i in range(min(self.max_samples, 100)):
            mock_data.append({
                'question': f"What is the person doing in the image?",
                'answer': "running",
                'image_id': f"image_{i:04d}",
                'explanation': f"The person in the image is running because you can see their legs in motion.",
                'question_tokens': torch.randint(0, 1000, (10,)),  # Mock tokenized question
                'image_features': torch.randn(49, 2048)  # Mock image features (7x7 patches)
            })
        return mock_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class ESNLIVEDataset(Dataset):
    """e-SNLI-VE dataset for visual entailment with explanations"""
    
    def __init__(self, data_path: str, split: str = 'train', max_samples: int = 1000):
        self.data_path = data_path
        self.split = split
        self.max_samples = max_samples
        
        # Load data (mock implementation)
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load e-SNLI-VE dataset (mock implementation)"""
        mock_data = []
        labels = ['entailment', 'contradiction', 'neutral']
        
        for i in range(min(self.max_samples, 100)):
            mock_data.append({
                'premise': f"The person is running in the park.",
                'hypothesis': f"Someone is exercising outdoors.",
                'label': np.random.choice(labels),
                'explanation': f"The premise describes running in a park, which entails exercising outdoors.",
                'image_id': f"image_{i:04d}",
                'premise_tokens': torch.randint(0, 1000, (8,)),
                'hypothesis_tokens': torch.randint(0, 1000, (6,)),
                'image_features': torch.randn(49, 2048)
            })
        return mock_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class FuzzyAttentionEvaluator:
    """Comprehensive evaluator for fuzzy attention networks"""
    
    def __init__(self, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 rule_extractor: Optional[RuleExtractor] = None):
        self.device = device
        self.rule_extractor = rule_extractor or RuleExtractor()
        self.interface = InteractiveInterface()
        
    def evaluate_model(self, 
                      model: nn.Module,
                      dataset: Dataset,
                      dataset_name: str,
                      model_name: str = "FuzzyAttention",
                      return_detailed: bool = True) -> EvaluationResult:
        """Comprehensive model evaluation"""
        
        print(f"🔍 Evaluating {model_name} on {dataset_name}")
        print("=" * 50)
        
        model.eval()
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Initialize metrics
        total_correct = 0
        total_samples = 0
        all_attention_entropies = []
        all_attention_sparsities = []
        all_fuzzy_rules = []
        all_attention_patterns = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 10:  # Limit for demo
                    break
                
                # Get model predictions and attention
                if dataset_name == 'VQA-X':
                    result = self._evaluate_vqa_batch(model, batch)
                elif dataset_name == 'e-SNLI-VE':
                    result = self._evaluate_esnli_batch(model, batch)
                else:
                    result = self._evaluate_generic_batch(model, batch)
                
                # Accumulate metrics
                total_correct += result['correct_predictions']
                total_samples += result['batch_size']
                
                if 'attention_entropy' in result:
                    all_attention_entropies.append(result['attention_entropy'])
                if 'attention_sparsity' in result:
                    all_attention_sparsities.append(result['attention_sparsity'])
                if 'fuzzy_rules' in result:
                    all_fuzzy_rules.extend(result['fuzzy_rules'])
                if 'attention_patterns' in result:
                    all_attention_patterns.append(result['attention_patterns'])
        
        end_time = time.time()
        
        # Calculate final metrics
        task_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        attention_entropy = np.mean(all_attention_entropies) if all_attention_entropies else 0.0
        attention_sparsity = np.mean(all_attention_sparsities) if all_attention_sparsities else 0.0
        
        # Rule consistency (simplified metric)
        rule_consistency = self._calculate_rule_consistency(all_fuzzy_rules)
        
        # Explanation quality (simplified metric)
        explanation_quality = self._calculate_explanation_quality(all_fuzzy_rules, all_attention_patterns)
        
        # Computational efficiency
        total_time = end_time - start_time
        samples_per_second = total_samples / total_time if total_time > 0 else 0.0
        
        metrics = EvaluationMetrics(
            task_accuracy=task_accuracy,
            attention_entropy=attention_entropy,
            attention_sparsity=attention_sparsity,
            rule_consistency=rule_consistency,
            explanation_quality=explanation_quality,
            user_satisfaction=0.0,  # Would be filled by user study
            computational_efficiency=samples_per_second
        )
        
        # Aggregate attention patterns
        aggregated_patterns = self._aggregate_attention_patterns(all_attention_patterns)
        
        computational_stats = {
            'total_time': total_time,
            'samples_per_second': samples_per_second,
            'memory_usage': torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        }
        
        result = EvaluationResult(
            model_name=model_name,
            dataset_name=dataset_name,
            metrics=metrics,
            fuzzy_rules=all_fuzzy_rules,
            attention_patterns=aggregated_patterns,
            user_study_results=None,
            computational_stats=computational_stats
        )
        
        self._print_evaluation_summary(result)
        return result
    
    def _evaluate_vqa_batch(self, model: nn.Module, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate VQA batch"""
        question_tokens = batch['question_tokens'].to(self.device)
        image_features = batch['image_features'].to(self.device)
        answers = batch['answer']
        
        # Get model predictions
        result = model(question_tokens, image_features, return_explanations=True)
        
        # Calculate accuracy (simplified)
        predicted_answers = torch.argmax(result['answer_logits'], dim=-1)
        correct = 0  # Simplified - would need proper answer matching
        
        return {
            'correct_predictions': correct,
            'batch_size': len(answers),
            'attention_entropy': np.mean([p['entropy'] for p in result.get('attention_patterns', {}).get('attention_entropy', [0])]),
            'attention_sparsity': np.mean([p for p in result.get('attention_patterns', {}).get('attention_sparsity', [0])]),
            'fuzzy_rules': result.get('fuzzy_rules', []),
            'attention_patterns': result.get('attention_patterns', {})
        }
    
    def _evaluate_esnli_batch(self, model: nn.Module, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate e-SNLI-VE batch"""
        premise_tokens = batch['premise_tokens'].to(self.device)
        hypothesis_tokens = batch['hypothesis_tokens'].to(self.device)
        image_features = batch['image_features'].to(self.device)
        labels = batch['label']
        
        # Get model predictions (simplified)
        # In real implementation, would process premise + hypothesis + image
        result = model(premise_tokens, image_features, return_explanations=True)
        
        # Calculate accuracy (simplified)
        predicted_labels = torch.argmax(result['answer_logits'], dim=-1)
        correct = 0  # Simplified - would need proper label matching
        
        return {
            'correct_predictions': correct,
            'batch_size': len(labels),
            'attention_entropy': np.mean([p['entropy'] for p in result.get('attention_patterns', {}).get('attention_entropy', [0])]),
            'attention_sparsity': np.mean([p for p in result.get('attention_patterns', {}).get('attention_sparsity', [0])]),
            'fuzzy_rules': result.get('fuzzy_rules', []),
            'attention_patterns': result.get('attention_patterns', {})
        }
    
    def _evaluate_generic_batch(self, model: nn.Module, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Generic batch evaluation"""
        return {
            'correct_predictions': 0,
            'batch_size': 1,
            'attention_entropy': 0.0,
            'attention_sparsity': 0.0,
            'fuzzy_rules': [],
            'attention_patterns': {}
        }
    
    def _calculate_rule_consistency(self, rules: List[FuzzyRule]) -> float:
        """Calculate rule consistency metric"""
        if not rules:
            return 0.0
        
        # Simplified consistency: check if rules have reasonable strength distribution
        strengths = [rule.strength for rule in rules]
        strength_std = np.std(strengths)
        strength_mean = np.mean(strengths)
        
        # Consistency is higher when strength distribution is more uniform
        consistency = 1.0 / (1.0 + strength_std / (strength_mean + 1e-8))
        return min(consistency, 1.0)
    
    def _calculate_explanation_quality(self, 
                                     rules: List[FuzzyRule], 
                                     patterns: List[Dict[str, Any]]) -> float:
        """Calculate explanation quality metric"""
        if not rules:
            return 0.0
        
        # Quality based on rule diversity and attention patterns
        rule_diversity = len(set((r.from_position, r.to_position) for r in rules)) / len(rules)
        
        # Attention pattern quality (simplified)
        pattern_quality = 0.5  # Would be more sophisticated in real implementation
        
        return (rule_diversity + pattern_quality) / 2.0
    
    def _aggregate_attention_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate attention patterns across batches"""
        if not patterns:
            return {}
        
        aggregated = {}
        for key in patterns[0].keys():
            if isinstance(patterns[0][key], list):
                aggregated[key] = [item for pattern in patterns for item in pattern[key]]
            else:
                aggregated[key] = [pattern[key] for pattern in patterns]
        
        return aggregated
    
    def _print_evaluation_summary(self, result: EvaluationResult):
        """Print evaluation summary"""
        print(f"\n📊 EVALUATION SUMMARY")
        print(f"Model: {result.model_name}")
        print(f"Dataset: {result.dataset_name}")
        print(f"Task Accuracy: {result.metrics.task_accuracy:.3f}")
        print(f"Attention Entropy: {result.metrics.attention_entropy:.3f}")
        print(f"Attention Sparsity: {result.metrics.attention_sparsity:.3f}")
        print(f"Rule Consistency: {result.metrics.rule_consistency:.3f}")
        print(f"Explanation Quality: {result.metrics.explanation_quality:.3f}")
        print(f"Computational Efficiency: {result.metrics.computational_efficiency:.1f} samples/sec")
        print(f"Total Fuzzy Rules: {len(result.fuzzy_rules)}")
        print(f"Memory Usage: {result.computational_stats['memory_usage']:.2f} GB")

class ComparativeEvaluator:
    """Compare fuzzy attention with baseline methods"""
    
    def __init__(self, evaluator: FuzzyAttentionEvaluator):
        self.evaluator = evaluator
        
    def compare_with_baselines(self, 
                             fuzzy_model: nn.Module,
                             baseline_models: Dict[str, nn.Module],
                             dataset: Dataset,
                             dataset_name: str) -> Dict[str, EvaluationResult]:
        """Compare fuzzy attention with baseline methods"""
        
        print(f"🔄 Comparing Fuzzy Attention with Baselines on {dataset_name}")
        print("=" * 60)
        
        results = {}
        
        # Evaluate fuzzy model
        results['FuzzyAttention'] = self.evaluator.evaluate_model(
            fuzzy_model, dataset, dataset_name, "FuzzyAttention"
        )
        
        # Evaluate baseline models
        for baseline_name, baseline_model in baseline_models.items():
            print(f"\n📊 Evaluating {baseline_name}...")
            results[baseline_name] = self.evaluator.evaluate_model(
                baseline_model, dataset, dataset_name, baseline_name
            )
        
        # Print comparison
        self._print_comparison(results)
        
        return results
    
    def _print_comparison(self, results: Dict[str, EvaluationResult]):
        """Print comparison table"""
        print(f"\n📈 COMPARISON RESULTS")
        print("=" * 80)
        print(f"{'Model':<20} {'Accuracy':<10} {'Entropy':<10} {'Sparsity':<10} {'Rules':<8} {'Efficiency':<12}")
        print("-" * 80)
        
        for model_name, result in results.items():
            print(f"{model_name:<20} "
                  f"{result.metrics.task_accuracy:<10.3f} "
                  f"{result.metrics.attention_entropy:<10.3f} "
                  f"{result.metrics.attention_sparsity:<10.3f} "
                  f"{len(result.fuzzy_rules):<8} "
                  f"{result.metrics.computational_efficiency:<12.1f}")

def demo_evaluation_framework():
    """Demo function for evaluation framework"""
    print("🧪 Evaluation Framework Demo")
    print("=" * 50)
    
    # Create evaluator
    evaluator = FuzzyAttentionEvaluator()
    
    # Create demo datasets
    vqa_dataset = VQAXDataset("mock_path", split='train', max_samples=20)
    esnli_dataset = ESNLIVEDataset("mock_path", split='train', max_samples=20)
    
    print(f"📊 Created datasets:")
    print(f"   VQA-X: {len(vqa_dataset)} samples")
    print(f"   e-SNLI-VE: {len(esnli_dataset)} samples")
    
    # Create fuzzy attention model
    fuzzy_model = VQAFuzzyModel(
        vocab_size=1000,
        answer_vocab_size=100,
        text_dim=128,
        image_dim=256,
        hidden_dim=128,
        n_heads=4,
        n_layers=2
    )
    
    print(f"✅ Created fuzzy model with {sum(p.numel() for p in fuzzy_model.parameters())} parameters")
    
    # Evaluate on VQA-X
    print(f"\n🔍 Evaluating on VQA-X...")
    vqa_results = evaluator.evaluate_model(fuzzy_model, vqa_dataset, 'VQA-X')
    
    # Evaluate on e-SNLI-VE
    print(f"\n🔍 Evaluating on e-SNLI-VE...")
    esnli_results = evaluator.evaluate_model(fuzzy_model, esnli_dataset, 'e-SNLI-VE')
    
    # Create comparative evaluator
    comp_evaluator = ComparativeEvaluator(evaluator)
    
    # Mock baseline models for comparison
    baseline_models = {
        'StandardAttention': VQAFuzzyModel(vocab_size=1000, answer_vocab_size=100, text_dim=128, image_dim=256, hidden_dim=128, n_heads=4, n_layers=2),
        'LIME': VQAFuzzyModel(vocab_size=1000, answer_vocab_size=100, text_dim=128, image_dim=256, hidden_dim=128, n_heads=4, n_layers=2)
    }
    
    # Compare with baselines
    print(f"\n🔄 Comparing with baselines...")
    comparison_results = comp_evaluator.compare_with_baselines(
        fuzzy_model, baseline_models, vqa_dataset, 'VQA-X'
    )
    
    print(f"\n🎉 Evaluation demo completed!")
    print(f"   VQA-X results: {len(vqa_results.fuzzy_rules)} rules extracted")
    print(f"   e-SNLI-VE results: {len(esnli_results.fuzzy_rules)} rules extracted")
    
    return evaluator, vqa_results, esnli_results, comparison_results

if __name__ == "__main__":
    demo_evaluation_framework()


"""
Configuration file for Human-Centered Differentiable Neuro-Fuzzy Architectures
"""

import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    """Configuration for FAN model architecture"""
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 512

    # Fuzzy logic parameters
    fuzzy_type: str = 'product'  # 'product', 'minimum', 'lukasiewicz'
    fuzzy_temperature: float = 0.5
    rule_extraction_threshold: float = 0.1

    # Multimodal parameters
    vision_encoder: str = 'clip-vit-base-patch32'
    text_encoder: str = 'bert-base-uncased'
    cross_modal_fusion: str = 'attention'

@dataclass 
class TrainingConfig:
    """Configuration for training parameters"""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100

@dataclass
class AdaptiveUIConfig:
    """Configuration for adaptive user interface"""
    expertise_levels: Dict[str, Dict[str, Any]] = None

    def __post_init__(self):
        if self.expertise_levels is None:
            self.expertise_levels = {
                'novice': {
                    'max_rules': 3,
                    'detail_level': 'summary',
                    'technical_terms': False,
                    'visual_aids': True,
                    'explanation_length': 'short'
                },
                'intermediate': {
                    'max_rules': 7,
                    'detail_level': 'detailed', 
                    'technical_terms': True,
                    'visual_aids': True,
                    'explanation_length': 'medium'
                },
                'expert': {
                    'max_rules': -1,
                    'detail_level': 'complete',
                    'technical_terms': True,
                    'visual_aids': False,
                    'explanation_length': 'full'
                }
            }

@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics and datasets"""
    datasets: list = None
    metrics: list = None
    user_study_size: int = 30

    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ['vqa_x', 'e_snli_ve', 'hateful_memes']
        if self.metrics is None:
            self.metrics = [
                'accuracy', 'comprehension_score', 'trust_calibration',
                'cognitive_load', 'rule_quality', 'explanation_usefulness'
            ]

class Config:
    """Master configuration class"""
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.adaptive_ui = AdaptiveUIConfig()
        self.evaluation = EvaluationConfig()

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4
        self.seed = 42

        # Paths
        self.data_dir = './data'
        self.model_dir = './models'
        self.results_dir = './results'
        self.logs_dir = './temp_logs'

# Global config instance
config = Config()

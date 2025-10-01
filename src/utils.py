"""
Utility functions for the Fuzzy Attention Network project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import logging
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_dir: str = './temp_logs', level: int = logging.INFO):
    """Setup logging configuration"""
    import os
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"{log_dir}/fan_{timestamp}.log"

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class FuzzyOperators:
    """Differentiable fuzzy logic operators"""

    @staticmethod
    def product_tnorm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Product t-norm: T(a,b) = a * b"""
        return a * b

    @staticmethod
    def minimum_tnorm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Minimum t-norm: T(a,b) = min(a,b)"""
        return torch.min(a, b)

    @staticmethod
    def lukasiewicz_tnorm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Łukasiewicz t-norm: T(a,b) = max(0, a+b-1)"""
        return torch.clamp(a + b - 1, min=0)

    @staticmethod
    def probabilistic_sum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Probabilistic sum t-conorm: S(a,b) = a+b-a*b"""
        return a + b - a * b

    @staticmethod
    def get_tnorm(tnorm_type: str):
        """Get t-norm function by name"""
        tnorms = {
            'product': FuzzyOperators.product_tnorm,
            'minimum': FuzzyOperators.minimum_tnorm,
            'lukasiewicz': FuzzyOperators.lukasiewicz_tnorm
        }
        return tnorms.get(tnorm_type, FuzzyOperators.product_tnorm)

def gaussian_membership(x: torch.Tensor, center: torch.Tensor, 
                       sigma: torch.Tensor) -> torch.Tensor:
    """Gaussian membership function: μ(x) = exp(-||x-c||²/2σ²)"""
    # x: [batch_size, seq_len, d_k]
    # center: [n_functions, d_k]
    # sigma: [n_functions, d_k]
    
    batch_size, seq_len, d_k = x.shape
    n_functions = center.shape[0]
    
    # Reshape for broadcasting: x -> [batch, seq, 1, d_k], center -> [1, 1, n_func, d_k]
    x_expanded = x.unsqueeze(2)  # [batch_size, seq_len, 1, d_k]
    center_expanded = center.unsqueeze(0).unsqueeze(0)  # [1, 1, n_functions, d_k]
    sigma_expanded = sigma.unsqueeze(0).unsqueeze(0)    # [1, 1, n_functions, d_k]
    
    # Compute squared distance
    diff = x_expanded - center_expanded  # [batch, seq, n_func, d_k]
    squared_dist = torch.sum(diff ** 2, dim=-1)  # [batch, seq, n_func]
    
    # Compute Gaussian membership
    sigma_squared = torch.sum(sigma_expanded ** 2, dim=-1) + 1e-8  # [batch, seq, n_func]
    membership = torch.exp(-squared_dist / (2 * sigma_squared))
    
    return membership  # [batch_size, seq_len, n_functions]

def numeric_to_linguistic(value: float) -> str:
    """Convert numeric values to linguistic terms"""
    if value < 0.15:
        return 'very_low'
    elif value < 0.35:
        return 'low'
    elif value < 0.45:
        return 'medium_low'
    elif value < 0.55:
        return 'medium'
    elif value < 0.65:
        return 'medium_high'
    elif value < 0.85:
        return 'high'
    else:
        return 'very_high'

class ExplanationTemplates:
    """Templates for generating natural language explanations"""

    NOVICE_TEMPLATES = [
        "The model focuses mainly on {key_elements}.",
        "The most important connections are between {connections}.",
        "The decision is primarily based on {main_factors}."
    ]

    INTERMEDIATE_TEMPLATES = [
        "Analysis shows {strength} attention between {source} and {target} (weight: {weight:.3f})",
        "The fuzzy reasoning process identifies {rule_type} rules with {confidence} confidence",
        "Cross-modal connections: {modality1} '{element1}' relates to {modality2} '{element2}'"
    ]

    EXPERT_TEMPLATES = [
        "Fuzzy membership function μ(x) yields {membership:.4f} for input {input_desc}",
        "T-norm operation: T({val1:.3f}, {val2:.3f}) = {result:.3f} using {tnorm_type}",
        "Rule derivation: IF {antecedent} THEN {consequent} (strength: {strength:.4f})"
    ]

    @classmethod
    def get_template(cls, user_level: str, template_type: str = 'general') -> str:
        """Get appropriate template for user level"""
        if user_level == 'novice':
            return np.random.choice(cls.NOVICE_TEMPLATES)
        elif user_level == 'intermediate':
            return np.random.choice(cls.INTERMEDIATE_TEMPLATES)
        else:
            return np.random.choice(cls.EXPERT_TEMPLATES)

def save_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                         epoch: int, loss: float, path: str):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, path)

def load_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                         path: str) -> Dict[str, Any]:
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint

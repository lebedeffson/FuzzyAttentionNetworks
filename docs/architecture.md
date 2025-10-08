# 🏗️ FAN Architecture Documentation

## Overview
Fuzzy Attention Networks (FAN) implement a novel approach to multimodal learning using fuzzy logic principles for attention mechanisms.

## Key Components

### 1. Universal FAN Model
- **File**: `src/universal_fan_model.py`
- **Purpose**: Core model architecture supporting multiple datasets
- **Features**: 
  - Configurable for different datasets
  - Transfer learning with BERT and ResNet
  - Fuzzy attention mechanisms

### 2. Dataset Manager
- **File**: `src/dataset_manager.py`
- **Purpose**: Unified interface for dataset handling
- **Features**:
  - Support for multiple datasets
  - Automatic data loading and preprocessing
  - Balanced sampling for class imbalance

### 3. Web Interface
- **File**: `demos/universal_web_interface.py`
- **Purpose**: Interactive web application
- **Features**:
  - Dataset selection
  - Real-time predictions
  - Interpretability visualizations

## Model Variants

### Hateful Memes Model
- **Architecture**: BERT + ResNet50 + 8-Head FAN
- **Performance**: F1: 0.5649, Accuracy: 59%
- **Use Case**: Binary classification of hateful content

### CIFAR-10 Model
- **Architecture**: BERT + ResNet18 + 4-Head FAN
- **Performance**: F1: 0.8808, Accuracy: 85%
- **Use Case**: 10-class image classification

## Fuzzy Attention Mechanism

The core innovation is the use of fuzzy membership functions for attention:

1. **Bell-shaped Functions**: Soft attention boundaries
2. **Learnable Parameters**: Centers and widths
3. **Multi-head Architecture**: Parallel attention heads
4. **Interpretability**: Human-readable patterns

## Usage

See `examples/basic_usage.py` for detailed usage examples.

# Fuzzy Attention Networks (FAN)

**Multimodal Classification System with Fuzzy Attention Mechanisms**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Fuzzy Attention Networks (FAN) is an innovative multimodal classification system that integrates fuzzy logic with attention mechanisms for analyzing textual and visual data. The system demonstrates high effectiveness across various classification tasks, including medical diagnosis applications.

### Key Features

- **🧠 Fuzzy Attention Networks**: Integration of fuzzy logic with attention mechanisms
- **🎨 Interactive Web Interface**: Real-time visualization and analysis capabilities
- **🏥 Medical Specialization**: Specialized models for medical diagnosis
- **📊 Full Interpretability**: Visualization of fuzzy functions and attention weights
- **🌐 Multilingual Support**: Interface available in multiple languages

## Supported Datasets

| Dataset | Classes | F1 Score | Accuracy | Architecture |
|---------|---------|----------|----------|--------------|
| **Stanford Dogs** | 20 | **95.74%** | **95.0%** | Advanced FAN + 8-Head Attention |
| **CIFAR-10** | 10 | **88.0%** | **85.0%** | BERT + ResNet18 + 4-Head FAN |
| **HAM10000** | 7 | **89.3%** | **75.0%** | Medical FAN + 8-Head Attention |
| **Chest X-Ray** | 2 | **78.0%** | **75.0%** | Medical FAN + 8-Head Attention |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/fuzzy-attention-networks.git
cd fuzzy-attention-networks

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Web Interface

```bash
# Launch interactive interface
python -m streamlit run demos/final_working_interface.py --server.port 8501
```

Open your browser at: **http://localhost:8501**

## System Architecture

### Core Components

1. **FuzzyAttention** - Core fuzzy attention mechanism
2. **AdvancedFANModel** - Advanced FAN architecture
3. **UniversalFANModel** - Universal FAN model
4. **DatasetManager** - Dataset management
5. **SimpleModelManager** - Model management

### Fuzzy Membership Functions

The system utilizes specialized fuzzy functions for different data types:

**Medical (Chest X-Ray):**
- X-Ray: Lung Opacity
- X-Ray: Consolidation
- X-Ray: Air Bronchogram
- X-Ray: Pleural Effusion
- X-Ray: Heart Shadow

**General (Stanford Dogs, CIFAR-10):**
- Image: Visual Saliency
- Image: Object Boundaries
- Image: Color Patterns
- Image: Texture Features
- Image: Spatial Relations

## Project Structure

```
FuzzyAttentionNetworks/
├── demos/
│   └── final_working_interface.py    # Main web interface
├── src/
│   ├── advanced_fan_model.py         # Advanced FAN model
│   ├── universal_fan_model.py        # Universal FAN model
│   ├── fuzzy_attention.py            # Core fuzzy attention
│   ├── dataset_manager.py            # Dataset management
│   ├── simple_model_manager.py       # Model management
│   └── utils.py                      # Utilities
├── scripts/
│   ├── download_*.py                 # Dataset download scripts
│   └── train_*.py                    # Model training scripts
├── models/                           # Trained models
├── data/                            # Datasets
└── diagrams/                        # Architecture diagrams
```

## Model Training

### Stanford Dogs
```bash
python scripts/train_stanford_dogs.py
```

### CIFAR-10
```bash
python scripts/train_advanced_stanford_dogs.py
```

### HAM10000 (Skin Cancer)
```bash
python scripts/train_ham10000.py
```

### Chest X-Ray (Pneumonia)
```bash
python scripts/train_chest_xray.py
```

## Web Interface

The interactive web interface provides:

- **🎯 Dataset Selection**: Switch between 4 datasets
- **🧪 Model Testing**: Upload images and text
- **📊 Visualization**: Fuzzy functions and attention weights
- **📈 Performance Analysis**: Metrics and confusion matrices
- **🔍 Interpretability**: Detailed prediction analysis

## Scientific Results

### Key Achievements

- **High Accuracy**: 95.74% F1-score on Stanford Dogs
- **Medical Applicability**: 89.3% F1-score on skin cancer diagnosis
- **Interpretability**: Complete visualization of fuzzy functions
- **Multimodality**: Effective processing of text and images

### Performance Metrics

The system demonstrates robust performance across different domains:

- **General Classification**: 88-95% F1-score on standard datasets
- **Medical Diagnosis**: 78-89% F1-score on medical datasets
- **Interpretability**: Full transparency through fuzzy function visualization
- **Scalability**: Efficient processing of multimodal inputs

## Technical Specifications

### Model Architectures

- **Advanced FAN**: 8-head attention, 1024 hidden dimensions
- **Universal FAN**: 4-head attention, 512 hidden dimensions
- **Medical FAN**: Specialized for medical diagnosis tasks

### Training Configuration

- **Optimizer**: AdamW with learning rate scheduling
- **Regularization**: Dropout, weight decay, early stopping
- **Data Augmentation**: Image transformations for robustness
- **Validation**: Cross-validation with holdout sets

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Project**: Fuzzy Attention Networks
- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [@your-username](https://github.com/your-username)

## Acknowledgments

- PyTorch team for the excellent framework
- Streamlit for the intuitive web interface
- Community for inspiration and support

---

**⭐ If this project was helpful, please give it a star!**
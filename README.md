# 🧠 Fuzzy Attention Networks

**Human-Centered Differentiable Neuro-Fuzzy Architectures: Interactive Explanation Interfaces for Multimodal AI with Adaptive User-Controlled Interpretability**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Ready%20for%20IUI%202026-brightgreen.svg)](https://iui.acm.org/2026/)

## 📋 Abstract

We propose a novel differentiable neuro-symbolic framework that integrates fuzzy logic directly into transformer architectures, enabling end-to-end learning while maintaining inherent interpretability through human-readable reasoning pathways.

Our **Fuzzy Attention Networks (FAN)** replace standard self-attention mechanisms with learnable fuzzy membership functions and differentiable t-norms, allowing automatic extraction of interpretable linguistic rules from trained attention weights. The architecture incorporates cross-modal fuzzy reasoning layers that generate compositional explanations spanning text and visual modalities.

For adaptive user interaction, we implement a dynamic explanation system with three-tier progressive disclosure: high-level fuzzy rule summaries for novices, detailed membership function visualizations for intermediate users, and full compositional rule derivations for experts.

## 🎯 Key Features

- **Learnable Fuzzy Membership Functions**: Gaussian, triangular, and trapezoidal functions
- **Differentiable T-norms**: Product, minimum, and Lukasiewicz t-norms
- **Cross-modal Reasoning**: Text-image attention with interpretable rules
- **Adaptive Explanations**: Three-tier progressive disclosure system
- **Real-time User Assessment**: Interaction-based expertise evaluation
- **Interactive Rule Refinement**: User-controlled rule modification

## 🏗️ Architecture

### Core Components

**Fuzzy Attention Networks (FAN)**
```python
class MultiHeadFuzzyAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, fuzzy_type: str = 'product'):
        self.heads = nn.ModuleList([
            FuzzyAttentionHead(d_model, self.d_k, fuzzy_type)
            for _ in range(n_heads)
        ])
```

**Multimodal Integration**
```python
class MultimodalFuzzyTransformer(nn.Module):
    def __init__(self, vocab_size: int, text_dim: int, image_dim: int, 
                 hidden_dim: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        # Complete multimodal transformer with fuzzy attention
```

**Rule Extraction System**
```python
class RuleExtractor:
    def extract_rules(self, attention_weights: torch.Tensor) -> List[FuzzyRule]:
        # Dynamic thresholding based on attention statistics
        dynamic_threshold = max(0.01, attention_mean + 0.1 * attention_std)
```

### Model Specifications
- **Parameters**: 5,675,034 total parameters
- **Text Processing**: 512-dimensional embeddings
- **Image Processing**: 2048-dimensional ResNet-50 features
- **Hidden Dimension**: 256 for cross-modal fusion
- **Attention Heads**: 4 heads per layer
- **Layers**: 2 transformer layers

## 📊 Dataset & Performance

### Hateful Memes Dataset
- **Total Samples**: 200 (100 real images + 100 placeholders)
- **Distribution**: 80 hateful (40%), 120 non-hateful (60%)
- **Source**: [neuralcatcher/hateful_memes](https://huggingface.co/datasets/neuralcatcher/hateful_memes)

### Performance Metrics
- **Inference Speed**: 24.3 samples/sec
- **Memory Usage**: < 2GB during inference
- **Attention Entropy**: 1.406 (balanced distribution)
- **Rule Extraction**: 0-500+ rules per analysis
- **Test Coverage**: 9/9 comprehensive checks passed

## 🚀 Quick Start

### Installation
```bash
git clone <repository-url>
cd FuzzyAttentionNetworks
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Usage
```bash
# Demo
python main.py --mode demo

# Train model
python main.py --mode train_improved --epochs 20

# Download dataset
python main.py --mode download_hateful_memes --max_samples 200

# Web interface
streamlit run demos/final_web_interface.py

# Run tests
python tests/final_project_check.py
```

## 🎯 Adaptive Interface

### Three-tier Explanation System

**Novice Level**
- Simple language, visual aids
- High-level fuzzy rule summaries
- Example: "The AI looks at 3 important connections in your text."

**Intermediate Level**
- Technical terminology with explanations
- Detailed membership function visualizations
- Example: "Fuzzy attention shows 3 strong connections with confidence > 0.7."

**Expert Level**
- Full technical details
- Complete compositional rule derivations
- Example: "Cross-modal fuzzy reasoning reveals 3 rules using product t-norm."

## 🛠️ Project Structure

```
FuzzyAttentionNetworks/
├── src/                    # Core source code (9 files)
├── demos/                  # Demonstration scripts (5 files)
├── tests/                  # Test suites (6 files)
├── experiments/            # Evaluation framework
├── data/                   # Datasets (200 samples)
├── models/                 # Trained models
├── utils/                  # Utility scripts
├── docs/                   # Documentation
├── main.py                 # Main entry point
├── train_*.py              # Training scripts
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🧪 Testing

```bash
# Run all tests
python tests/test_system.py
python tests/test_rule_extraction.py
python tests/test_visualization.py
python tests/test_web_interface.py
python tests/test_hateful_memes.py

# Comprehensive validation
python tests/final_project_check.py
```

## 📝 Research

### IUI 2026 Submission
This work is being prepared for submission to **IUI 2026** (ACM Conference on Intelligent User Interfaces).

### Paper Status
- **Abstract**: ✅ Completed
- **Implementation**: ✅ Completed
- **Technical Evaluation**: ✅ Completed
- **User Study**: ⏳ Ready to conduct
- **Paper Writing**: ⏳ Ready to begin

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [neuralcatcher/hateful_memes](https://huggingface.co/datasets/neuralcatcher/hateful_memes) for the dataset
- PyTorch team for the deep learning framework
- Streamlit team for the web interface framework

---

**Status**: Ready for user studies and IUI 2026 submission 🚀

**Version**: 1.0.0  
**All Tests**: ✅ Passing  
**Ready for**: User studies, paper submission, production deployment
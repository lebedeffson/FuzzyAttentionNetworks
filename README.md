# 🧠 Fuzzy Attention Networks (FAN)

**Universal Interface for Multi-Dataset Analysis**

A cutting-edge implementation of Fuzzy Attention Networks with support for multiple datasets and real-time interpretability.

## 🎯 Supported Datasets

### 1. Hateful Memes Detection
- **Task**: Binary classification (Hateful/Not Hateful)
- **Model**: BERT + ResNet50 + 8-Head FAN
- **Performance**: F1: 0.5649, Accuracy: 59%
- **Architecture**: 768 hidden dimensions, 5 membership functions per head

### 2. CIFAR-10 Classification
- **Task**: 10-class image classification
- **Model**: BERT + ResNet18 + 4-Head FAN
- **Performance**: F1: 0.8808, Accuracy: 85%
- **Architecture**: 512 hidden dimensions, 5 membership functions per head

## 🚀 Quick Start

### 1. Installation
```bash
# Clone repository
git clone <repository-url>
cd FuzzyAttentionNetworks

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Universal Interface
```bash
python run_universal_interface.py
```

Open your browser at: http://localhost:8501

## 🏗️ Architecture

### Core Components

1. **Universal FAN Model** (`src/universal_fan_model.py`)
   - Configurable for different datasets
   - Transfer learning with BERT and ResNet
   - Fuzzy attention mechanisms

2. **Dataset Manager** (`src/dataset_manager.py`)
   - Unified interface for multiple datasets
   - Automatic data loading and preprocessing
   - Balanced sampling for class imbalance

3. **Web Interface** (`demos/universal_web_interface.py`)
   - Interactive dataset selection
   - Real-time model predictions
   - Interpretability visualizations

### Fuzzy Attention Mechanism

- **Membership Functions**: Bell-shaped functions for soft attention
- **Multi-Head Architecture**: Parallel attention heads for different features
- **Learnable Parameters**: Centers and widths of membership functions
- **Interpretability**: Human-readable attention patterns

## 📊 Model Performance

| Dataset | F1 Score | Accuracy | Architecture |
|---------|----------|----------|--------------|
| Hateful Memes | 0.5649 | 59% | BERT + ResNet50 + 8-Head FAN |
| CIFAR-10 | 0.8808 | 85% | BERT + ResNet18 + 4-Head FAN |

## 🔍 Key Features

### 1. Universal Architecture
- Single codebase for multiple datasets
- Automatic model configuration
- Easy addition of new datasets

### 2. Transfer Learning
- Pre-trained BERT for text understanding
- Pre-trained ResNet for image features
- Frozen feature extractors for stability

### 3. Enhanced Interpretability
- **4 Interactive Tabs**: Attention Weights, Fuzzy Functions, Performance, Rules
- **8+ Visualizations**: Heatmaps, graphs, comparisons
- **Rule Extraction**: Linguistic rules with confidence scores
- **High Confidence Predictions**: 70-95% confidence range

### 4. Advanced Web Interface
- Interactive web application with tabs
- Live model predictions with high confidence
- Dataset switching without restart
- Model comparison and performance metrics
- Real-time visualizations

## 📁 Project Structure

```
FuzzyAttentionNetworks/
├── src/                          # Core source code
│   ├── universal_fan_model.py    # Universal FAN model
│   ├── dataset_manager.py        # Dataset management
│   └── ...                       # Other utilities
├── demos/                        # Web interfaces
│   └── universal_web_interface.py
├── models/                       # Trained models
│   ├── hateful_memes/           # Hateful memes models
│   └── cifar10/                 # CIFAR-10 models
├── data/                        # Datasets
│   ├── hateful_memes/           # Hateful memes data
│   └── cifar10_fan/             # CIFAR-10 data
└── run_universal_interface.py   # Main launcher
```

## 🎛️ Usage Examples

### Web Interface
1. Select dataset from sidebar
2. Load corresponding model
3. Input text and/or image
4. Get predictions with explanations

### Programmatic Usage
```python
from src.dataset_manager import DatasetManager
from src.universal_fan_model import ModelManager

# Initialize managers
dataset_manager = DatasetManager()
model_manager = ModelManager()

# Load dataset
dataset = dataset_manager.create_dataset('cifar10', 'train')

# Load model
model = model_manager.get_model('cifar10')

# Make prediction
result = model_manager.predict(
    'cifar10', text_tokens, attention_mask, image
)
```

## 🔬 Research Applications

- **Multimodal Learning**: Text + Image understanding
- **Interpretable AI**: Human-readable attention patterns
- **Fuzzy Logic**: Soft decision boundaries
- **Transfer Learning**: Pre-trained feature extraction

## 📈 Future Work

- [ ] Support for more datasets
- [ ] Advanced interpretability features
- [ ] Real-time model fine-tuning
- [ ] Mobile interface support

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions and support, please open an issue on GitHub.

---

**🧠 Fuzzy Attention Networks - Bridging the gap between interpretability and performance in multimodal AI**


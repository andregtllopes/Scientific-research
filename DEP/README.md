# DEP: Dual-Path Embeddings for Protein Toxicity Classification

## 📋 Description

**DEP (Dual-Path Embeddings)** is an advanced deep learning model designed for protein toxicity classification. This work combines two complementary neural network pathways to process protein sequences and predict their toxicity potential with high accuracy. The model leverages both convolutional neural networks and transformer-based architectures to capture local and global patterns in protein sequences simultaneously.

## 🎯 Key Features

- **Dual-Path Architecture**: Combines local pathway (convolutional-transformer hybrid) and global pathway (full transformer attention)
- **Advanced Protein Embeddings**: Processes protein sequences with sophisticated embedding representations
- **Windowed Attention Mechanism**: Efficient attention computation using sliding windows over sequence data
- **Multi-Scale Feature Extraction**: Captures patterns at different scales through dilated convolutions
- **Binary Classification**: Predicts protein toxicity with confidence scores

## 🏗️ Architecture Overview

The DEP model implements a sophisticated dual-pathway neural network:

### Local Pathway (ConvTransformer)
- **Dilated Stem**: Multi-scale convolutional feature extraction with dilated convolutions
- **Transformer Blocks**: Windowed attention mechanisms for efficient sequence processing
- **Summary Aggregator**: Aggregates local features using transformer attention with summary tokens

### Global Pathway (Global Attention)
- **Full Transformer Encoder**: Processes entire sequence with global self-attention
- **Multi-Head Attention**: Captures different semantic aspects of protein sequences
- **Layer Normalization**: Stabilizes training and improves convergence

### Classification Head
- **Feature Fusion**: Concatenates local and global pathway outputs
- **Binary Classifier**: Linear classification layer for toxicity prediction

## 📦 Project Structure

```
DEP/
├── model.py                                              # DEP model implementation
├── README.md                                            # This file
└── DEP Dual-Path Embeddings for Protein Toxicity Classification.pdf  # Full research paper
```

## 🔧 Requirements

- Python 3.8 or higher
- PyTorch 1.9 or higher
- NumPy
- Pandas (for data handling)

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/andregtllopes/Scientific-research.git
cd Scientific-research/DEP
pip install torch numpy pandas
```

## 🚀 Usage

### Model Configuration

The model uses a configuration dataclass (`ConfigBuzelin`) to set architecture parameters:

```python
from model import ConfigBuzelin, ConvTransformerCached_v7

config = ConfigBuzelin(
    seq_length=512,          # Maximum protein sequence length
    hidden_dim=256,          # Hidden dimension size
    mlp_dim=512,             # MLP dimension in transformer blocks
    window_size=16,          # Attention window size
    stride=8,                # Stride for windowed attention
    dropout_rate=0.1         # Dropout rate for regularization
)

model = ConvTransformerCached_v7(config)
```

### Making Predictions

```python
import torch
from model import ConvTransformerCached_v7, ConfigBuzelin

# Initialize model
config = ConfigBuzelin()
model = ConvTransformerCached_v7(config)
model.eval()

# Example: protein sequence embedding (batch_size=1, seq_length=512, embedding_dim=128)
protein_embedding = torch.randn(1, 512, 128)

# Forward pass
with torch.no_grad():
    toxicity_score = model(protein_embedding)
    prediction = (toxicity_score > 0.5).float()  # Binary classification

print(f"Toxicity Score: {toxicity_score.item():.4f}")
print(f"Classification: {'Toxic' if prediction.item() == 1 else 'Non-Toxic'}")
```

## 📊 Model Components

### DilatedStem
Multi-scale convolutional feature extraction using dilated convolutions for capturing features at different receptive field sizes.

### WindowedAttention
Efficient attention mechanism that computes attention within sliding windows, reducing computational complexity from O(n²) to O(n) for long sequences.

### TransformerBlock
Combines windowed attention with residual connections and MLP layers for deep feature learning.

### GlobalAttentionPath
Full sequence transformer encoder for capturing long-range dependencies across the entire protein sequence.

### SummaryAggregator
Aggregates local features using attention with learnable summary tokens.

## 🔬 Methodology

1. **Sequence Encoding**: Protein sequences are converted into numerical embeddings
2. **Local Feature Extraction**: DilatedStem and local transformer blocks extract patterns at different scales
3. **Global Context**: Global attention pathway processes the full sequence
4. **Feature Fusion**: Local and global representations are concatenated
5. **Classification**: Binary classification head predicts toxicity

## 📈 Performance

The model achieves competitive performance on protein toxicity classification benchmarks:

- **Input Flexibility**: Handles variable-length sequences with configurable maximum length
- **Computational Efficiency**: Windowed attention reduces memory usage for long sequences
- **Multi-Scale Analysis**: Captures patterns at different scales through dilated convolutions

For detailed performance metrics and comparisons, refer to the full research paper.

## 👨‍🔬 Author

**André Gambogi** - Researcher and Deep Learning Practitioner

## 📝 Citation

If you use this model in your research, please cite it as:

```bibtex
@article{Gambogi2024DEP,
  author={Gambogi, André},
  title={DEP: Dual-Path Embeddings for Protein Toxicity Classification},
  year={2025}
}
```

---

**Last Updated**: May 2024  
**Model Version**: 1.0  
**PyTorch Version**: 1.9+

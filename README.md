# Causal Discovery Transformer for Stock Price Prediction

A research project implementing Causal Discovery Transformer models for stock price prediction and discovering causal relationships between financial variables using deep learning and causal inference techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Components](#model-components)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## 🎯 Overview

This project explores the intersection of causal inference and deep learning for financial time series prediction. The main objective is to not only predict stock prices but also discover and leverage causal relationships between financial indicators to improve prediction accuracy and interpretability.

### What Makes This Different?

Unlike traditional time series models that treat all features equally, our Causal Discovery Transformer:
- **Learns causal structures** between financial variables automatically
- **Adapts to temporal dynamics** using sliding window training
- **Provides interpretability** through discovered causal graphs
- **Tests robustness** via intervention analysis

## ✨ Key Features

### 🔍 Causal Discovery
- Automatic discovery of causal relationships between financial indicators
- Differentiable causal graph learning with sparsity constraints
- DAG (Directed Acyclic Graph) enforcement to ensure valid causal structures
- Temporal causal masking to prevent information leakage

### 🧠 Advanced Architecture
- Multi-head causal attention mechanism
- Positional encoding for temporal awareness
- Intervention adapter for robustness testing
- Sliding window training for temporal stability

### 📊 Comprehensive Analysis
- Causal relationship visualization
- Intervention robustness testing
- Multi-window aggregation for stable causal graphs
- Comparison with vanilla Transformer baseline

## 🏗️ Architecture

### Causal Discovery Transformer

```
Input Features (T, V)
    ↓
CausalDiscoveryLayer
    ├── Causal Logits (learnable)
    ├── Gumbel-Softmax Sampling
    └── Causal Adjacency Matrix
    ↓
Causal Feature Enhancement
    ↓
Input Projection → d_model
    ↓
Positional Encoding
    ↓
Transformer Blocks (×N)
    ├── Causal Multi-Head Attention
    ├── Layer Normalization
    ├── Feed-Forward Network
    └── Residual Connections
    ↓
Output Projection
    ↓
Predictions
```

### Loss Function

```
Total Loss = α·Prediction_Loss 
           + β·DAG_Loss 
           + γ·Sparsity_Loss 
           + δ·Consistency_Loss 
           + ε·Intervention_Loss
```

## 📁 Project Structure

```
Casual-Discovery-Transformer/
├── CD_Transformer.ipynb
│   └── Main implementation with sliding windows training
│
├── CD_Transfomer_MODIFY_WITH_ATTENTION_GUIDED_CAUSUAL..ipynb
│   └── Enhanced version with attention-guided causal discovery
│
├── vanilla_transformer.ipynb
│   └── Baseline Vanilla Transformer for comparison
│
├── Casual_discovery_transformer.docx
│   └── Detailed documentation and theoretical background
│
└── README.md
    └── This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy pandas
pip install yfinance
pip install scikit-learn
pip install matplotlib seaborn
pip install jupyter notebook
```

Or install all at once:

```bash
pip install torch numpy pandas yfinance scikit-learn matplotlib seaborn jupyter
```

## 💻 Usage

### 1. Quick Start with Causal Discovery Transformer

```python
import numpy as np
from CD_Transformer import StockPriceCausalPredictor, train_sliding_windows

# Define your features
feature_names = ['Close', 'Volume', 'Returns', 'RSI', 'MACD', 
                 'BB_Upper', 'BB_Lower', 'Volatility', 'MA20', 'MA50']

# Prepare your data (X: [N, T, V], y: [N, 1])
# N = number of samples, T = sequence length, V = number of variables

# Train with sliding windows
all_adj, edge_mean, edge_freq, stable_graph_mean, stable_graph_freq = train_sliding_windows(
    X=X_train,
    y=y_train,
    feature_names=feature_names,
    win_len_months=24,        # 24-month windows
    step_months=6,            # 6-month step
    samples_per_month=21,     # ~21 trading days per month
    num_epochs=20,
    batch_size=32,
    d_model=64,
    num_heads=4,
    num_layers=2,
    prediction_horizon=1
)

# Analyze discovered causal relationships
print("Stable Causal Graph (by mean):")
print(stable_graph_mean)
print(f"\nNumber of stable edges: {int(stable_graph_mean.sum())}")
```

### 2. Making Predictions

```python
# Initialize predictor
predictor = StockPriceCausalPredictor(
    feature_names=feature_names,
    d_model=64,
    num_heads=4,
    num_layers=2,
    prediction_horizon=1,
    device='cuda'
)

# Make predictions
results = predictor.predict(X_test)

predictions = results['predictions']
causal_structure = results['causal_structure']
attention_weights = results['attention_weights']
```

### 3. Analyzing Causal Relationships

```python
# Get causal relationships with threshold
relationships = predictor.get_causal_relationships(threshold=0.3)

for rel in relationships[:10]:  # Top 10 relationships
    print(f"{rel['cause']} → {rel['effect']}: {rel['strength']:.3f}")
```

### 4. Testing Intervention Robustness

```python
# Test how interventions on each variable affect predictions
intervention_effects = predictor.test_intervention_robustness(
    X_test, 
    num_samples=20
)

for feature, effect in intervention_effects.items():
    print(f"{feature}: {effect:.4f}")
```

### 5. Vanilla Transformer Baseline

```python
from vanilla_transformer import VanillaTransformer

# Initialize model
model = VanillaTransformer(
    num_features=len(feature_names),
    d_model=64,
    nhead=4,
    num_layers=2,
    dropout=0.1
).to(device)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
```

## 🔧 Model Components

### CausalDiscoveryLayer

Learns the causal adjacency matrix between variables:

```python
class CausalDiscoveryLayer(nn.Module):
    def __init__(self, num_variables: int, hidden_dim: int = 32):
        # Learnable causal logits
        self.causal_logits = nn.Parameter(torch.randn(num_variables, num_variables))
        # Transform network
        self.causal_transform = nn.Sequential(...)
```

**Key Methods:**
- `get_causal_adjacency()`: Returns causal adjacency matrix with sigmoid activation
- `apply_causal_mask()`: Applies temporal causal masking to features

### CausalAttention

Multi-head attention with temporal causal masking:

```python
class CausalAttention(nn.Module):
    def forward(self, x, causal_adj=None, mask=None):
        # Compute Q, K, V
        # Apply temporal mask (lower triangular)
        # Apply causal structure mask (optional)
        # Return attention output and weights
```

### SparsityScheduler

Dynamically adjusts sparsity weight during training:

```python
scheduler = SparsityScheduler(
    model_wrapper=predictor,
    start=0.03,  # Initial sparsity weight
    end=0.08,    # Final sparsity weight
    T=25         # Number of epochs
)
```

## 📚 Training Strategy

### Sliding Window Approach

To ensure temporal stability of discovered causal relationships:

1. **Divide data** into overlapping time windows (e.g., 24 months with 6-month steps)
2. **Train separate models** on each window
3. **Aggregate causal graphs** across windows
4. **Identify stable edges** that appear consistently

### Sparsity Weight Suggestion

Automatically suggests sparsity weight based on data length:

```python
sparsity_weight = suggest_sparsity_weight(
    n_months=24,      # Window length in months
    base=0.10,        # Base sparsity
    ref_months=24,    # Reference period
    floor=0.03,       # Minimum sparsity
    ceil=0.12         # Maximum sparsity
)
```

**Rule of thumb:**
- 2 years (24 months): ~0.10
- 5 years (60 months): ~0.04-0.05
- 10 years (120 months): ~0.03-0.04

### Loss Components

| Component | Purpose | Weight |
|-----------|---------|--------|
| Prediction Loss | Forecast accuracy | 1.0 |
| DAG Loss | Enforce acyclicity | 0.05 |
| Sparsity Loss | Encourage sparse graphs | 0.03-0.12 |
| Consistency Loss | Stabilize learning | 0.01 |
| Intervention Loss | Robustness | 0.02 |

## 📊 Results

### Discovered Causal Relationships

Example output from training:

```
Number of windows: 4
Edge mean shape: (6, 6)
Stable edges by mean (>= 0.35): 30
Edge freq shape: (6, 6)
Stable edges by freq (bin>0.40, freq>=0.50): 30
```

### Performance Metrics

**Vanilla Transformer (Baseline):**
- MSE: 0.074221
- RMSE: 0.272435
- MAE: 0.212962

**Causal Discovery Transformer:**
- Improved interpretability through causal graphs
- Comparable or better prediction accuracy
- Enhanced robustness to interventions

## 🔬 Technical Details

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 64 | Model dimension |
| `num_heads` | 4 | Number of attention heads |
| `num_layers` | 2 | Number of transformer blocks |
| `ff_dim` | 128 | Feed-forward dimension |
| `dropout` | 0.1 | Dropout rate |
| `max_seq_len` | 50 | Maximum sequence length |
| `learning_rate` | 1e-3 | Initial learning rate |
| `weight_decay` | 1e-2 | L2 regularization |

### Data Requirements

- **Input shape**: `[batch_size, sequence_length, num_features]`
- **Output shape**: `[batch_size, prediction_horizon]`
- **Recommended sequence length**: 30-50 time steps
- **Minimum training samples**: 500+ for stable causal discovery

### Computational Requirements

- **GPU Memory**: ~2-4 GB for typical configurations
- **Training Time**: ~5-10 minutes per window (GPU)
- **Inference**: Real-time capable

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add more financial indicators
- [ ] Implement attention visualization
- [ ] Add hyperparameter tuning
- [ ] Create web interface for predictions
- [ ] Add more baseline models
- [ ] Implement cross-validation
- [ ] Add unit tests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@misc{causal_discovery_transformer_2024,
  author = {Dthai2103},
  title = {Causal Discovery Transformer for Stock Price Prediction},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Dthai2103/Casual-Discovery-Transformer}}
}
```

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

---

**⭐ If you find this project useful, please consider giving it a star!**

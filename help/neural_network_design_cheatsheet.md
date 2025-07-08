# PyTorch Neural Network Design Cheatsheet

## 🎯 Step 1: Problem Definition & Data Analysis

### Essential Questions to Ask First:
- **Target Variable Type**: Continuous (regression) or categorical (classification)?
- **Dataset Size**: < 1K (small), 1K-100K (medium), > 100K (large)
- **Feature Count**: < 10 (few), 10-100 (moderate), > 100 (many)
- **Data Quality**: Missing values? Outliers? Imbalanced classes?
- **Feature Types**: Numerical, categorical, text, images?

### Quick Data Inspection Checklist:
```python
# Basic info
print(f"Dataset shape: {df.shape}")
print(f"Target distribution: {df['target'].value_counts()}")
print(f"Missing values: {df.isnull().sum()}")
print(f"Data types: {df.dtypes}")

# For numerical features
df.describe()  # Check ranges, means, std

# For categorical features
df['categorical_col'].nunique()  # High vs low cardinality
```

---

## 🏗️ Step 2: Architecture Design Framework

### Problem Type → Architecture Mapping

| Problem Type | Output Layer | Loss Function | Metrics |
|-------------|-------------|---------------|---------|
| **Binary Classification** | `nn.Linear(hidden, 1)` | `BCEWithLogitsLoss()` | Accuracy, F1, AUC |
| **Multi-class Classification** | `nn.Linear(hidden, num_classes)` | `CrossEntropyLoss()` | Accuracy, F1-macro |
| **Regression** | `nn.Linear(hidden, 1)` | `MSELoss()` or `L1Loss()` | MAE, RMSE, R² |
| **Multi-output Regression** | `nn.Linear(hidden, num_outputs)` | `MSELoss()` | MAE per output |

### Dataset Size → Model Complexity

| Dataset Size | Hidden Layers | Neurons per Layer | Regularization |
|-------------|---------------|-------------------|----------------|
| **< 1K samples** | 1-2 layers | 32-128 neurons | Heavy dropout (0.5-0.7) |
| **1K-10K samples** | 2-3 layers | 64-256 neurons | Moderate dropout (0.3-0.5) |
| **10K-100K samples** | 3-5 layers | 128-512 neurons | Light dropout (0.1-0.3) |
| **> 100K samples** | 5+ layers | 256-1024 neurons | BatchNorm + light dropout |

---

## ⚙️ Step 3: Component Selection Guide

### Activation Functions
```python
# When to use each:
activations = {
    'ReLU': nn.ReLU(),           # Default choice, fast, works well
    'LeakyReLU': nn.LeakyReLU(), # When ReLU causes dead neurons
    'ELU': nn.ELU(),             # Smooth, good for deep networks
    'GELU': nn.GELU(),           # State-of-the-art, computationally expensive
    'Tanh': nn.Tanh(),           # For outputs between -1 and 1
    'Sigmoid': nn.Sigmoid(),     # Only for binary classification output
}
```

### Optimizers
```python
# Optimizer selection guide:
optimizers = {
    'Adam': optim.Adam(lr=0.001),     # Default choice, adaptive learning
    'AdamW': optim.AdamW(lr=0.001),   # Adam with better weight decay
    'SGD': optim.SGD(lr=0.01),        # Simple, good with momentum
    'RMSprop': optim.RMSprop(lr=0.001) # Good for RNNs
}

# Learning rate by dataset size:
# Small dataset: 0.01-0.1
# Medium dataset: 0.001-0.01
# Large dataset: 0.0001-0.001
```

### Regularization Techniques
```python
# When to use each:
regularization = {
    'Dropout': nn.Dropout(0.5),           # Overfitting, any layer
    'BatchNorm': nn.BatchNorm1d(64),      # Training stability, after linear
    'Weight Decay': {'weight_decay': 1e-4}, # L2 regularization in optimizer
    'Early Stopping': {'patience': 10},   # Prevent overfitting
}
```

---

## 🔧 Step 4: Hyperparameter Tuning Priority

### Tune in This Order:
1. **Learning Rate** (most important!)
   - Start with: 0.001
   - Try: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

2. **Architecture**
   - Hidden layer sizes: [64, 128, 256, 512]
   - Number of layers: [1, 2, 3, 4]

3. **Regularization**
   - Dropout rates: [0.1, 0.3, 0.5, 0.7]
   - Weight decay: [1e-5, 1e-4, 1e-3]

4. **Training Parameters**
   - Batch size: [16, 32, 64, 128]
   - Optimizer choice

---

## 📊 Step 5: Data Preprocessing Decision Tree

### Feature Scaling (Always Important!)
```python
# Decision tree for scaling:
if "features have different ranges (e.g., age: 0-100, income: 0-100000)":
    if "outliers present":
        scaler = RobustScaler()  # Uses median/IQR
    else:
        scaler = StandardScaler()  # Z-score normalization

elif "need features in [0,1] range":
    scaler = MinMaxScaler()

elif "sparse data with many zeros":
    scaler = MaxAbsScaler()
```

### Categorical Features
```python
# High cardinality (>50 unique values):
# - Use embedding layers or target encoding
# - Consider dropping if not informative

# Low cardinality (<50 unique values):
# - One-hot encoding with pd.get_dummies()
# - Label encoding for ordinal data
```

---

## 🚀 Step 6: Implementation Template

### Basic Model Structure
```python
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        super().__init__()

        # Build layers dynamically
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # Output layer (no activation for regression, sigmoid/softmax handled by loss)
        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Example usage:
model = NeuralNetwork(
    input_size=20,           # Number of features
    hidden_sizes=[128, 64],  # Hidden layer sizes
    output_size=1,           # 1 for regression/binary, num_classes for multi-class
    dropout_rate=0.3
)
```

---

## 🔍 Step 7: Training Monitoring & Debugging

### Essential Metrics to Track
```python
# Training loop essentials:
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0

    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass
        loss.backward()

        # Gradient clipping (if needed)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

# What to monitor:
# - Training loss decreasing
# - Validation loss not increasing (overfitting check)
# - Gradient norms (exploding/vanishing gradients)
# - Learning rate changes (if using scheduler)
```

### Red Flags & Solutions
| Problem | Symptoms | Solution |
|---------|----------|----------|
| **Underfitting** | High train & val loss | Increase model complexity |
| **Overfitting** | Low train loss, high val loss | Add regularization |
| **Exploding Gradients** | Loss becomes NaN/Inf | Gradient clipping, lower LR |
| **Vanishing Gradients** | Loss doesn't decrease | Higher LR, different activation |
| **Dead ReLU** | Many zero activations | Use LeakyReLU or ELU |

---

## 📋 Step 8: Quick Decision Checklist

### Before You Start Coding:
- [ ] **Data preprocessed** (scaled, encoded, split)
- [ ] **Problem type identified** (regression/classification)
- [ ] **Architecture planned** (layers, neurons, activation)
- [ ] **Loss function chosen** (MSE, CrossEntropy, etc.)
- [ ] **Metrics defined** (what success looks like)
- [ ] **Baseline established** (simple model to beat)

### During Development:
- [ ] **Start simple** (2-3 layers, basic hyperparameters)
- [ ] **Monitor training** (loss curves, overfitting)
- [ ] **Validate assumptions** (data distribution, feature importance)
- [ ] **Iterate systematically** (change one thing at a time)

### Final Model:
- [ ] **Cross-validation** (ensure robustness)
- [ ] **Test set evaluation** (final unbiased performance)
- [ ] **Error analysis** (where does the model fail?)
- [ ] **Documentation** (hyperparameters, preprocessing steps)

---

## 🎯 Quick Start Examples

### Regression Example
```python
# For predicting house prices
model = NeuralNetwork(
    input_size=10,          # 10 features
    hidden_sizes=[64, 32],  # 2 hidden layers
    output_size=1,          # 1 continuous output
    dropout_rate=0.2
)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### Binary Classification Example
```python
# For spam detection
model = NeuralNetwork(
    input_size=50,          # 50 features
    hidden_sizes=[128, 64], # 2 hidden layers
    output_size=1,          # 1 binary output
    dropout_rate=0.5
)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---

## 🔧 Advanced Tips

### When to Use Advanced Features:
- **Batch Normalization**: Deep networks (>5 layers) or unstable training
- **Learning Rate Scheduling**: Long training (>50 epochs) or plateauing loss
- **Gradient Clipping**: RNNs or very deep networks
- **Custom Weight Initialization**: Very deep networks or specific activations
- **Multiple Optimizers**: Different learning rates for different layers

### Performance Optimization:
```python
# For larger datasets
DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)

# For GPU training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

Remember: Start simple, make it work, then make it better!

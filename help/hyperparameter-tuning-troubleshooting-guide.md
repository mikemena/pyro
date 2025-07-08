# Hyperparameter Tuning Guide for Poor Model Performance

Here's a comprehensive guide on hyperparameters to tune when your model isn't performing well:

## 1. Architecture Hyperparameters

### Layer Depth and Width

```python
# Too shallow - underfitting
model = nn.Sequential(
    nn.Linear(input_size, 32),  # Try: 128, 256, 512
    nn.ReLU(),
    nn.Linear(32, output_size)   # Add more layers?
)

# Better - more capacity
model = nn.Sequential(
    nn.Linear(input_size, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, output_size)
)
```

### Activation Functions

```python
# Try different activations
activations = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(0.1),
    'elu': nn.ELU(),
    'gelu': nn.GELU(),  # Often good for deep networks
    'swish': nn.SiLU(), # Smooth version of ReLU
}
```

## 2. Optimization Hyperparameters

### Learning Rate (Most Important!)

```python
# Common ranges to try
learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

# Quick LR finder
def find_lr(model, train_loader, init_lr=1e-6, end_lr=1):
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.1)

    losses = []
    lrs = []

    for batch in train_loader:
        # Forward pass
        loss = train_step(model, batch)
        losses.append(loss.item())
        lrs.append(optimizer.param_groups[0]['lr'])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if optimizer.param_groups[0]['lr'] > end_lr:
            break

    # Plot losses vs learning rates to find optimal LR
    plt.plot(lrs, losses)
    plt.xscale('log')
```

### Batch Size

```python
# Larger batch = more stable gradients, less noise
# Smaller batch = more updates, better generalization
batch_sizes = [16, 32, 64, 128, 256]

# Note: Adjust learning rate with batch size
# Rule of thumb: Double batch size → increase LR by √2
```

### Optimizer Choice

```python
optimizers = {
    'adam': optim.Adam(model.parameters(), lr=0.001),
    'adamw': optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01),
    'sgd': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    'rmsprop': optim.RMSprop(model.parameters(), lr=0.001),
}
```

## 3. Regularization Hyperparameters

### Dropout

```python
class RegularizedModel(nn.Module):
    def __init__(self, dropout_rate=0.5):  # Try: 0.1, 0.3, 0.5, 0.7
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Prevents overfitting
        return x
```

### Weight Decay (L2 Regularization)

```python
# Try different weight decay values
weight_decays = [0, 1e-5, 1e-4, 1e-3, 1e-2]
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
```

### Batch Normalization

```python
# Can stabilize training
model = nn.Sequential(
    nn.Linear(input_size, 256),
    nn.BatchNorm1d(256),  # Add after linear layers
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(128, output_size)
)
```

## 4. Data-Related Hyperparameters

### Data Augmentation

```python
# For images
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomCrop(224, padding=4),
])

# For tabular data - add noise
def add_gaussian_noise(x, std=0.1):
    return x + torch.randn_like(x) * std
```

### Train/Validation Split

```python
# Try different splits
split_ratios = [0.7, 0.8, 0.9]  # 70/30, 80/20, 90/10
```

## 5. Training Strategy Hyperparameters

### Number of Epochs

```python
# Use early stopping with patience
early_stopping = EarlyStopping(patience=20)  # Try: 10, 20, 30
```

### Learning Rate Schedule

```python
# Different scheduling strategies
schedulers = {
    'step': optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1),
    'plateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10),
    'cosine': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100),
    'exponential': optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
}
```

## 6. Loss Function Alternatives

```python
# For regression
losses = {
    'mse': nn.MSELoss(),
    'mae': nn.L1Loss(),
    'huber': nn.HuberLoss(),  # Robust to outliers
    'smooth_l1': nn.SmoothL1Loss(),
}

# For classification
losses = {
    'ce': nn.CrossEntropyLoss(),
    'focal': FocalLoss(),  # For imbalanced classes
    'label_smoothing': nn.CrossEntropyLoss(label_smoothing=0.1),
}
```

## Systematic Hyperparameter Search

### Grid Search

```python
from itertools import product

# Define search space
param_grid = {
    'lr': [1e-4, 1e-3, 1e-2],
    'hidden_size': [64, 128, 256],
    'dropout': [0.2, 0.4, 0.6],
    'batch_size': [32, 64, 128]
}

# Try all combinations
results = []
for params in product(*param_grid.values()):
    config = dict(zip(param_grid.keys(), params))
    val_loss = train_model_with_config(config)
    results.append((config, val_loss))
```

### Random Search (Often Better!)

```python
import random

# Random search - often finds good params faster
best_loss = float('inf')
for trial in range(50):  # 50 random trials
    config = {
        'lr': 10**random.uniform(-5, -2),  # Log scale
        'hidden_size': random.choice([64, 128, 256, 512]),
        'dropout': random.uniform(0.1, 0.7),
        'batch_size': random.choice([16, 32, 64, 128]),
        'activation': random.choice(['relu', 'elu', 'gelu'])
    }

    val_loss = train_model_with_config(config)
    if val_loss < best_loss:
        best_loss = val_loss
        best_config = config
```

## Debugging Poor Performance

```python
# 1. Check if model is learning at all
print(f"Initial loss: {initial_loss}")
print(f"Final loss: {final_loss}")
# If no change → learning rate too low or model broken

# 2. Check for overfitting
print(f"Train loss: {train_loss}")
print(f"Val loss: {val_loss}")
# If val >> train → add regularization

# 3. Monitor gradient flow
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm()}")
# If all ~0 → vanishing gradients
```

## Quick Improvement Checklist

1. **First try**: Increase learning rate (most common issue)
2. **If underfitting**: Add more layers/neurons
3. **If overfitting**: Add dropout, weight decay, or data augmentation
4. **If unstable**: Reduce learning rate, add batch norm, clip gradients
5. **If slow convergence**: Try different optimizer (AdamW often good)
6. **Always**: Ensure data is properly normalized/scaled

The key is to change one thing at a time and track results systematically!

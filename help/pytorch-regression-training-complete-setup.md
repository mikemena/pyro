# PyTorch Regression Training Setup Guide

To have code that is designed for regression, using MSE as the loss function, and includes features like early stopping, learning rate scheduling, means the code is set up to predict continuous numerical values (like prices, temperatures, scores) rather than categories (like yes/no, cat/dog).

## Regression with MSE

```python
# Regression: Predicting continuous values
# Examples:
# - House price: $245,000
# - Temperature: 72.5°F
# - Stock price: $156.23
# - Age: 25.7 years

# MSE (Mean Squared Error) Loss
criterion = nn.MSELoss()

# How MSE works:
# Prediction: 25.5
# Actual:     23.0
# Error:      2.5
# Squared:    6.25  ← MSE penalizes larger errors more
```

## Complete Regression Setup

```python
import torch
import torch.nn as nn
import torch.optim as optim

class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # No activation! Output any value
        )

    def forward(self, x):
        return self.network(x)

# Setup for regression
model = RegressionModel(input_size=10)
criterion = nn.MSELoss()  # For continuous values
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## Early Stopping

Prevents overfitting by stopping training when validation performance stops improving:

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Usage
early_stopping = EarlyStopping(patience=10)

for epoch in range(1000):  # Max epochs
    # ... training code ...

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break
```

## Learning Rate Scheduling

Adjusts learning rate during training for better convergence:

```python
# Setup scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # Minimize loss
    factor=0.5,      # Reduce lr by half
    patience=5       # Wait 5 epochs before reducing
)
```

## Complete Training Loop with All Features

```python
# Initialize components
model = RegressionModel(input_size=10)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
early_stopping = EarlyStopping(patience=15)

# Training loop
train_losses = []
val_losses = []

for epoch in range(200):
    # Training phase
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        # Forward pass
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)  # MSE loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            val_loss += loss.item()

    # Calculate average losses
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    # Learning rate scheduling
    scheduler.step(avg_val_loss)

    # Early stopping check
    early_stopping(avg_val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch}")
        break

    # Print progress
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, "
          f"Val Loss={avg_val_loss:.4f}, LR={current_lr}")
```

## What This Achieves

- **MSE for Regression**: Optimizes for accurate continuous predictions
- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **LR Scheduling**: Fine-tunes learning when progress slows

## Visual of Training Progress

```
Loss
│
│ Train loss ↘
│         ↘   ↘
│           ↘   ← LR reduced here
│             ↘
│ Val loss    ↘_____ ← Early stop here
│                  ↗ (starts overfitting)
└─────────────────────────► Epochs
```

## Key Differences from Classification

| Feature | Regression | Classification |
|---------|------------|----------------|
| **Output** | Any continuous value | Probabilities/Classes |
| **Loss** | MSE, MAE, Huber | CrossEntropy, BCE |
| **Final activation** | None | Softmax/Sigmoid |
| **Metrics** | R², RMSE, MAE | Accuracy, F1, Precision |
| **Example** | Price: $523.67 | Category: "House" |

This setup is ideal for problems where you need precise numerical predictions rather than category labels!

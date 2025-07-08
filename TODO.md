[X] Replace pkl files with other readable format outputted by the data_preprocessing.py

[X] In data_analyzer in line 103, update def determine_column_type to add
   a elif for high cardinality categorical, and change the other one for
   low cardinlaity categorical.

[X] In data_preprocessor in line 160 finish Process high-cardinality categorical feature

Research Needed about train_model.py

[X] Tools for regression, like Mean Squared Error (MSE) as the loss function, and includes early stopping to avoid overfitting

[X] Training process also adjusts the learning rate and saves the best model based on validation performance

[X] Measures performance with metrics like MSE, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score, which are standard for checking how well the model predicts

- This practice involves monitoring validation metrics during training and saving model checkpoints when performance improves. It's a form of model selection that ensures you keep the best version of your model.

- Metrics vary significantly between regression and classification. Here's a comprehensive guide:

# Regression Metrics

- For regression (predicting continuous values), you typically use:

## Mean Squared Error (MSE)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)


# Or in PyTorch:
mse = torch.nn.functional.mse_loss(predictions, targets)

# Interpretation:
# - Penalizes large errors more (squared)
# - Units: squared units of target (e.g., dollars²)
# - Lower is better

## Root Mean Squared Error (RMSE)

import numpy as np

rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# Interpretation:
# - Same units as target variable
# - More interpretable than MSE
# - If predicting house prices: RMSE = $10,000 means typical error

## Mean Absolute Error (MAE)
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)

# Interpretation:
# - Average absolute difference
# - Less sensitive to outliers than MSE
# - Same units as target

## R² Score (Coefficient of Determination)
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)

# Interpretation:
# - 1.0 = perfect predictions
# - 0.0 = no better than predicting mean
# - <0 = worse than predicting mean
# - 0.8 = model explains 80% of variance

## Mean Absolute Percentage Error (MAPE)
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Interpretation:
# - Percentage error (scale-independent)
# - 10% MAPE = predictions off by 10% on average
# - Bad for values near zero

# Classification Metrics
- For classification, completely different metrics:

## Binary Classification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# 1. Accuracy - Overall correctness
accuracy = accuracy_score(y_true, y_pred)
# Problem: Misleading for imbalanced datasets

# 2. Precision - When model says "positive", how often correct?
precision = precision_score(y_true, y_pred)

# 3. Recall (Sensitivity) - Of all actual positives, how many found?
recall = recall_score(y_true, y_pred)

# 4. F1 Score - Harmonic mean of precision & recall
f1 = f1_score(y_true, y_pred)

# 5. ROC-AUC - Area under ROC curve
# Needs probabilities, not just predictions
roc_auc = roc_auc_score(y_true, y_proba)

# 6. Confusion Matrix - See all prediction types
cm = confusion_matrix(y_true, y_pred)
# [[True Negatives, False Positives],
#  [False Negatives, True Positives]]

## Multi-class Classification

# Accuracy still works
accuracy = accuracy_score(y_true, y_pred)

# For precision/recall/F1, need averaging strategy
precision = precision_score(y_true, y_pred, average='weighted')
# average options: 'micro', 'macro', 'weighted'

# Confusion matrix for detailed view
cm = confusion_matrix(y_true, y_pred)
# n×n matrix for n classes

## Choosing Metrics Based on Problem
- When to Use Each Metric
Regression:

MSE/RMSE: When large errors are particularly bad
MAE: When all errors matter equally, robust to outliers
R²: When you want to know proportion of variance explained
MAPE: When you need scale-independent comparison

Classification:

Accuracy: Only when classes are balanced
Precision: When false positives are costly (spam detection)
Recall: When false negatives are costly (disease detection)
F1: Balanced view of precision and recall
ROC-AUC: Overall model quality, threshold-independent

[X] Tune hyperparameters like the number of layers or learning rate to see if you can improve performance, especially if the current results aren’t great.

- Here's a comprehensive guide on hyperparameters to tune when your model isn't performing well:

# Architecture Hyperparameters

## Layer Depth and Width

- Too shallow - underfitting
model = nn.Sequential(
    nn.Linear(input_size, 32),  # Try: 128, 256, 512
    nn.ReLU(),
    nn.Linear(32, output_size)   # Add more layers?
)

- Better - more capacity
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

## Activation Functions

- Try different activations
activations = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(0.1),
    'elu': nn.ELU(),
    'gelu': nn.GELU(),  # Often good for deep networks
    'swish': nn.SiLU(), # Smooth version of ReLU
}

## Optimization Hyperparameters

- Learning Rate (Most Important!)
-- Common ranges to try
learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

## Quick LR finder
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

## Batch Size
- Larger batch = more stable gradients, less noise
- Smaller batch = more updates, better generalization
  batch_sizes = [16, 32, 64, 128, 256]

- Note: Adjust learning rate with batch size
- Rule of thumb: Double batch size → increase LR by √2

## Optimizer Choice

optimizers = {
    'adam': optim.Adam(model.parameters(), lr=0.001),
    'adamw': optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01),
    'sgd': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    'rmsprop': optim.RMSprop(model.parameters(), lr=0.001),
}

# Regularization Hyperparameters

## Dropout
class RegularizedModel(nn.Module):
    def __init__(self, dropout_rate=0.5):  # Try: 0.1, 0.3, 0.5, 0.7
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Prevents overfitting
        return x

## Weight Decay (L2 Regularization)
- Try different weight decay values
weight_decays = [0, 1e-5, 1e-4, 1e-3, 1e-2]
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

## Batch Normalization
- Can stabilize training
model = nn.Sequential(
    nn.Linear(input_size, 256),
    nn.BatchNorm1d(256),  # Add after linear layers
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(128, output_size)
)

# Data-Related Hyperparameters

## Data Augmentation
# For images
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomCrop(224, padding=4),
])

- For tabular data - add noise

## Train/Validation Split
- Try different splits
split_ratios = [0.7, 0.8, 0.9]  # 70/30, 80/20, 90/10
def add_gaussian_noise(x, std=0.1):
    return x + torch.randn_like(x) * std

# Training Strategy Hyperparameters
## Number of Epochs
- Use early stopping with patience
early_stopping = EarlyStopping(patience=20)  # Try: 10, 20, 30

## Learning Rate Schedule
- Different scheduling strategies
schedulers = {
    'step': optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1),
    'plateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10),
    'cosine': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100),
    'exponential': optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
}

## Loss Function Alternatives
- For regression
losses = {
    'mse': nn.MSELoss(),
    'mae': nn.L1Loss(),
    'huber': nn.HuberLoss(),  # Robust to outliers
    'smooth_l1': nn.SmoothL1Loss(),
}

- For classification
losses = {
    'ce': nn.CrossEntropyLoss(),
    'focal': FocalLoss(),  # For imbalanced classes
    'label_smoothing': nn.CrossEntropyLoss(label_smoothing=0.1),
}

# Systematic Hyperparameter Search
- Grid Search

from itertools import product

- Define search space
param_grid = {
    'lr': [1e-4, 1e-3, 1e-2],
    'hidden_size': [64, 128, 256],
    'dropout': [0.2, 0.4, 0.6],
    'batch_size': [32, 64, 128]
}

- Try all combinations
results = []
for params in product(*param_grid.values()):
    config = dict(zip(param_grid.keys(), params))
    val_loss = train_model_with_config(config)
    results.append((config, val_loss))

## Random Search (Often Better!)
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

## Debugging Poor Performance
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

## -- Quick Improvement Checklist --

First try: Increase learning rate (most common issue)
If underfitting: Add more layers/neurons
If overfitting: Add dropout, weight decay, or data augmentation
If unstable: Reduce learning rate, add batch norm, clip gradients
If slow convergence: Try different optimizer (AdamW often good)
Always: Ensure data is properly normalized/scaled

The key is to change one thing at a time and track results systematically!

[X] For larger datasets, consider setting num_workers in the DataLoader to speed up data loading

- num_workers controls how many subprocesses are used for data loading in parallel. It speeds up training by loading the next batch while the GPU is processing the current one.

# Platform specific
if platform == "Windows":
    num_workers = 0  # Windows has multiprocessing issues
else:
    num_workers = 4  # Linux/Mac work well

# What num_workers Does
- num_workers controls how many subprocesses are used for data loading in parallel. It speeds up training by loading the next batch while the GPU is processing the current one.

[X] Code is designed for regression, using MSE as the loss function, and includes features like early stopping, learning rate scheduling

# Regression with MSE
- Is used for continuous numerical values (like prices, temperatures, scores) rather than categories (like yes/no, cat/dog).

MSE (Mean Squared Error) Loss
criterion = nn.MSELoss()

# How MSE works:
Prediction: 25.5
Actual:     23.0
Error:      2.5
Squared:    6.25  ← MSE penalizes larger errors more

# Early Stopping
-Prevents overfitting by stopping training when validation performance stops improving:

# Learning Rate Scheduling
- Adjusts learning rate during training for better convergence:
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # Minimize loss
    factor=0.5,      # Reduce lr by half
    patience=5       # Wait 5 epochs before reducing
)

[X] An output layer with a single neuron, suitable for regression, without an activation function to allow for continuous predictions.What if i dont want a continuous predictions, but a prediction like yes/no?

- For yes/no (binary classification), you need to change your output layer

# Here's an approach with no activation + BCEWWithLogitsLoss

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        # No sigmoid here!

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Raw logits
        return x

# Usage
model = BinaryClassifier(input_size=10, hidden_size=64)
criterion = nn.BCEWithLogitsLoss()  # Combines sigmoid + BCE

# Training
logits = model(input)  # Raw values (no sigmoid)
loss = criterion(logits, target)  # More numerically stable

[X] Weight initialization is handled by the _initialize_weights method, using Xavier uniform initialization for linear layers, which is a standard practice to ensure stable training.

- Xavier (Glorot) initialization sets weights to specific random values that help signals flow properly through the network. It prevents the "vanishing/exploding activations" problem at the start of training.

- Start without manual initialization. PyTorch's defaults are quite good. Only add custom initialization if you see problems or have specific requirements!

# Skip manual initialization when:

- Using standard layers (Linear, Conv2d)
- Using ReLU activations
- Model trains fine without it
- Building simple architectures

# Consider initialization when:

- Very deep networks (>10 layers)
- Using tanh/sigmoid activations
- Training doesn't converge
- Gradients vanish/explode at start
- Matching specific research papers

# Signs of bad initialization:
1. Loss is NaN from epoch 1
2. Loss doesn't decrease at all
3. Gradients are all zeros
4. Activations die immediately

# Quick fix to try:
model.apply(lambda m: nn.init.xavier_uniform_(m.weight)
           if hasattr(m, 'weight') else None)

[X] Training on a batch-by-batch basis with gradient clipping to prevent exploding gradients

# What is Gradient Clipping
- Gradient clipping prevents gradients from becoming too large during backpropagation by "clipping" them to a maximum value. This prevents the "exploding gradient" problem where huge gradients cause unstable training.

# When Gradient Clipping is Needed

- RNNs/LSTMs/GRUs
- Default approach: Start without, add if needed

# Networks with many layers
if num_layers > 50:
    # Gradients can compound through layers
    use_gradient_clipping = True

# Unstable Training Signs
Loss during training:
Epoch 1: 2.3
Epoch 2: 1.8
Epoch 3: NaN  ← Gradient explosion!

# Almost always use clipping with RNNs
model = nn.LSTM(input_size, hidden_size, num_layers)
# RNNs suffer from exploding gradients due to repeated multiplication
clip_grad_norm_(model.parameters(), max_norm=1.0)  # Essential!

- Very Deep Networks

# When You DON'T Need It

# Simple models rarely need clipping:
- Basic CNNs (ResNet, VGG)
- Standard feedforward networks
- Shallow networks (< 10 layers)
- When using BatchNorm (helps stabilize gradients)

# These are usually fine without clipping:
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
# No clipping needed!

# How to Know If You Need It
1. Loss becomes NaN or Inf
   if torch.isnan(loss) or torch.isinf(loss):
      print("You need gradient clipping!")

2. Loss explodes suddenly
      0.5 → 0.4 → 0.3 → 9999999 → NaN

3. Gradient magnitude monitoring
     total_norm = 0
     for p in model.parameters():
       if p.grad is not None:
           total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Gradient norm: {total_norm}")

# If this is frequently > 100, consider clipping

[X] learning rate scheduler (ReduceLROnPlateau) that halves the learning rate when validation loss plateaus, with a patience of 5 epochs.

- A learning rate scheduler automatically adjusts the learning rate during training.
-ReduceLROnPlateau specifically monitors a metric (like validation loss) and reduces the learning rate when improvement stalls.

-   Common Configuration Options

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',           # 'min' for loss, 'max' for metrics like accuracy
    factor=0.5,           # New_lr = old_lr * factor
    patience=5,           # Epochs to wait before reducing
    threshold=0.0001,     # Minimum change to qualify as improvement
    threshold_mode='rel', # 'rel'ative or 'abs'olute
    cooldown=0,           # Epochs to wait before resuming normal operation
    min_lr=1e-6,         # Don't reduce below this
    eps=1e-08            # Minimal decay applied to lr
)

# When should i use it?

# Simple/small models often work fine without schedulers:
- Small datasets (< 10k samples)
- Simple architectures (few layers)
- Quick experiments/prototyping
- When training for < 20 epochs
- When loss decreases smoothly

# Just optimizer is fine:
optimizer = optim.Adam(model.parameters(), lr=0.001)
# No scheduler needed!

[X] Adam optimizer with a learning rate of 0.001 and weight decay (1e-5) for additional regularization

- An optimizer always needed
- Adam is often the default choice for neural networks because it works well out-of-the-box
- Learning Rate = 0.001 is default and works well for most cases
  -   Controls the step size for weight updates
  -   Too high: training unstable, might overshoot
  -   Too low: training very slow
- weight decay adds L2 regularization (penalty on large weights)
  -   Helps prevent overfitting

Research Needed about data_preprocessor
[X] What exactly is scaling features? Lik scaled inputs. If features have different ranges, this could affect results

- sklearn.preprocessing offers several scaling methods, each suited for different situations:

from sklearn.preprocessing import (
    StandardScaler,    # Z-score normalization
    MinMaxScaler,      # Scale to range [0, 1]
    RobustScaler,      # Uses median/IQR, handles outliers
    Normalizer,        # Scales each sample to unit norm
    MaxAbsScaler       # Scale to [-1, 1] by max absolute value
)

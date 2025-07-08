# Machine Learning Metrics Guide: Regression vs Classification

Yes, metrics vary significantly between regression and classification! Here's a comprehensive guide:

## Regression Metrics

For regression (predicting continuous values), you typically use:

### 1. Mean Squared Error (MSE)

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
# Or in PyTorch:
mse = torch.nn.functional.mse_loss(predictions, targets)

# Interpretation:
# - Penalizes large errors more (squared)
# - Units: squared units of target (e.g., dollars²)
# - Lower is better
```

### 2. Root Mean Squared Error (RMSE)

```python
import numpy as np

rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# Interpretation:
# - Same units as target variable
# - More interpretable than MSE
# - If predicting house prices: RMSE = $10,000 means typical error
```

### 3. Mean Absolute Error (MAE)

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)

# Interpretation:
# - Average absolute difference
# - Less sensitive to outliers than MSE
# - Same units as target
```

### 4. R² Score (Coefficient of Determination)

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)

# Interpretation:
# - 1.0 = perfect predictions
# - 0.0 = no better than predicting mean
# - <0 = worse than predicting mean
# - 0.8 = model explains 80% of variance
```

### 5. Mean Absolute Percentage Error (MAPE)

```python
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Interpretation:
# - Percentage error (scale-independent)
# - 10% MAPE = predictions off by 10% on average
# - Bad for values near zero
```

## Classification Metrics

For classification, completely different metrics:

### Binary Classification

```python
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
```

### Multi-class Classification

```python
# Accuracy still works
accuracy = accuracy_score(y_true, y_pred)

# For precision/recall/F1, need averaging strategy
precision = precision_score(y_true, y_pred, average='weighted')
# average options: 'micro', 'macro', 'weighted'

# Confusion matrix for detailed view
cm = confusion_matrix(y_true, y_pred)
# n×n matrix for n classes
```

## Complete Example: Regression vs Classification

### Regression Example

```python
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# During validation
model.eval()
predictions = []
targets = []

with torch.no_grad():
    for batch_x, batch_y in val_loader:
        pred = model(batch_x)
        predictions.append(pred.cpu().numpy())
        targets.append(batch_y.cpu().numpy())

predictions = np.concatenate(predictions)
targets = np.concatenate(targets)

# Calculate all metrics
mse = mean_squared_error(targets, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(targets, predictions)
r2 = r2_score(targets, predictions)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
```

### Classification Example

```python
from sklearn.metrics import classification_report

# Binary classification
model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for batch_x, batch_y in val_loader:
        logits = model(batch_x)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Comprehensive report
print(classification_report(all_labels, all_preds))
# Shows precision, recall, F1 for each class

# ROC-AUC (needs probabilities)
auc = roc_auc_score(all_labels, all_probs)
print(f"ROC-AUC: {auc:.4f}")
```

## Choosing Metrics Based on Problem

### When to Use Each Metric

**Regression:**
- **MSE/RMSE**: When large errors are particularly bad
- **MAE**: When all errors matter equally, robust to outliers
- **R²**: When you want to know proportion of variance explained
- **MAPE**: When you need scale-independent comparison

**Classification:**
- **Accuracy**: Only when classes are balanced
- **Precision**: When false positives are costly (spam detection)
- **Recall**: When false negatives are costly (disease detection)
- **F1**: Balanced view of precision and recall
- **ROC-AUC**: Overall model quality, threshold-independent

## Custom Metrics for Specific Problems

```python
# Financial prediction - Directional accuracy
def directional_accuracy(y_true, y_pred):
    """Percentage of times we predict the right direction"""
    return np.mean(np.sign(y_true) == np.sign(y_pred))

# Time series - Mean Absolute Scaled Error
def mase(y_true, y_pred, y_train):
    """Compares to naive forecast"""
    naive_mae = np.mean(np.abs(np.diff(y_train)))
    mae = np.mean(np.abs(y_true - y_pred))
    return mae / naive_mae

# Ranking - Spearman correlation
from scipy.stats import spearmanr
correlation, _ = spearmanr(y_true, y_pred)
```

## Monitoring During Training

```python
class MetricTracker:
    def __init__(self, task='regression'):
        self.task = task
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []

    def update(self, preds, targets):
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())

    def compute(self):
        preds = np.array(self.predictions)
        targets = np.array(self.targets)

        if self.task == 'regression':
            return {
                'mse': mean_squared_error(targets, preds),
                'mae': mean_absolute_error(targets, preds),
                'r2': r2_score(targets, preds)
            }
        else:  # classification
            return {
                'accuracy': accuracy_score(targets, preds),
                'f1': f1_score(targets, preds, average='weighted')
            }
```

## Key Takeaway

The key is choosing metrics that align with your business/research goals and problem constraints!

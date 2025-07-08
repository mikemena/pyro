# Saving Best Model Based on Validation Performance

## What is "Saving Best Model Based on Validation Performance"?

This practice involves monitoring validation metrics during training and saving model checkpoints when performance improves. It's a form of model selection that ensures you keep the best version of your model.

## How It Works

```python
class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', mode='min'):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')

    def __call__(self, current_score, model):
        if self.mode == 'min' and current_score < self.best_score:
            self.best_score = current_score
            torch.save(model.state_dict(), self.filepath)
            print(f"Model saved! {self.monitor}: {current_score:.4f}")
            return True
        elif self.mode == 'max' and current_score > self.best_score:
            self.best_score = current_score
            torch.save(model.state_dict(), self.filepath)
            print(f"Model saved! {self.monitor}: {current_score:.4f}")
            return True
        return False
```

## Complete Training Loop with Best Practices

```python
import torch
import copy
from pathlib import Path

def train_model(model, train_loader, val_loader, num_epochs=100):
    # Setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )

    # Tracking
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'val_loss': []}

    # Early stopping
    patience = 20
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, 'best_model.pth')
            patience_counter = 0
            print(f"✓ New best model saved! Val loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        print(f"Epoch {epoch}: Train={avg_train_loss:.4f}, "
              f"Val={avg_val_loss:.4f}, LR={current_lr:.6f}")

    # Load best model weights
    model.load_state_dict(best_model_weights)
    return model, history
```

## When is This a Best Practice?

### ✅ ALWAYS Use When:

**Training for Production**
```python
# Production models need the best possible performance
checkpoint = ModelCheckpoint('production_model.pth')
# Don't want the final epoch, want the best epoch!
```

**Long Training Runs**
```python
# Training for hours/days? Save progress!
if num_epochs > 50:
    save_best_model = True  # Protection against crashes
```

**Limited Data**
```python
# Small datasets are prone to overfitting
if len(dataset) < 10000:
    # Best model prevents using overfit version
    use_checkpointing = True
```

**Research/Experiments**
```python
# Comparing different architectures fairly
for architecture in ['modelA', 'modelB', 'modelC']:
    # Each should be evaluated at its best performance
    train_with_checkpoint(architecture)
```

### ⚠️ Optional When:

**Quick Prototyping**
```python
# Just testing if idea works
if experimenting:
    # Skip the complexity, just train
    model.fit(train_data, epochs=5)
```

**Very Large Datasets**
```python
# ImageNet-scale: Less likely to overfit
if len(dataset) > 1_000_000:
    # Model usually improves monotonically
    save_every_n_epochs = 10  # Instead of best
```

**Transfer Learning**
```python
# Fine-tuning pre-trained model
if using_pretrained:
    # Often trains for just a few epochs
    # Final model is usually best
    save_best = False
```

## Different Saving Strategies

### 1. Save Best Only
```python
# Most common - keeps only the best model
if val_loss < best_val_loss:
    torch.save(model.state_dict(), 'best_model.pth')
```

### 2. Save Best + Regular Checkpoints
```python
# Save best AND periodic checkpoints
if val_loss < best_val_loss:
    torch.save(model.state_dict(), 'best_model.pth')

if epoch % 10 == 0:
    torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pth')
```

### 3. Save Last N Best Models
```python
# Keep top 3 models
from collections import deque

best_models = deque(maxlen=3)
if val_loss < worst_of_best:
    best_models.append((val_loss, model.state_dict()))
    # Save all top models
```

## What to Save?

```python
# Minimal - just weights
torch.save(model.state_dict(), 'model.pth')

# Complete - everything needed to resume
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_loss': best_val_loss,
    'history': history,
    'config': config,  # Hyperparameters
}, 'checkpoint.pth')

# Loading complete checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## Why This Prevents Overfitting

```
Training Progress:
Val Loss
│
│     Best model here
│          ↓
│    ╱────╲        ← Saves this
│   ╱      ╲___╱╲
│  ╱              ╲  ← But training continues
│ ╱                ╲___ ← Final model (worse!)
└──────────────────────── Epochs
   Underfitting  Best  Overfitting
```

## Practical Implementation Tips

```python
# 1. Use descriptive filenames
filename = f"model_val{val_loss:.4f}_epoch{epoch}.pth"

# 2. Clean up old checkpoints
def save_checkpoint(model, val_loss, epoch, keep_last=3):
    # Save new checkpoint
    filename = f"checkpoint_{epoch}_{val_loss:.4f}.pth"
    torch.save(model.state_dict(), filename)

    # Remove old checkpoints
    checkpoints = sorted(Path('.').glob('checkpoint_*.pth'))
    for old_checkpoint in checkpoints[:-keep_last]:
        old_checkpoint.unlink()

# 3. Validate the saved model
def validate_checkpoint(checkpoint_path, val_loader):
    model = create_model()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    # Run validation to ensure it works
```

## Summary

**Best practice level:**

🟢 **Production/Research**: Always save best model
🟡 **Experimentation**: Recommended but optional
🟠 **Quick tests**: Usually skip

The overhead is minimal, and the benefits (preventing overfitting, disaster recovery, fair comparison) far outweigh the costs. It's one of those practices that seems unnecessary until you need it - then you're very glad you did it!

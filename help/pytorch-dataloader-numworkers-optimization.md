# PyTorch DataLoader num_workers Optimization Guide

## What num_workers Does

`num_workers` controls how many subprocesses are used for data loading in parallel. It speeds up training by loading the next batch while the GPU is processing the current one.

## Visual Explanation

### num_workers=0 (default):
```
GPU: |--Train Batch 1--|--Wait--|--Train Batch 2--|--Wait--|
CPU: |--Load Batch 1---|        |--Load Batch 2---|        |
     ↑                          ↑
     GPU waits for data!        GPU waits again!
```

### num_workers=4:
```
GPU: |--Train Batch 1--|--Train Batch 2--|--Train Batch 3--|
CPU: |--Load B1--|--Load B2--|--Load B3--|--Load B4--|
     ↑ Workers load ahead while GPU trains!
```

## How It Works

```python
# Without workers - Sequential loading
train_loader = DataLoader(dataset, batch_size=32, num_workers=0)
# Main process: Load data → Send to GPU → Train → Repeat

# With workers - Parallel loading
train_loader = DataLoader(dataset, batch_size=32, num_workers=4)
# Main process: Train on GPU
# Worker 1: Load batch 1
# Worker 2: Load batch 2
# Worker 3: Load batch 3
# Worker 4: Load batch 4
```

## When You Need Multiple Workers

### 1. Heavy Data Preprocessing

```python
class MyDataset(Dataset):
    def __getitem__(self, idx):
        # Heavy operations benefit from workers:
        image = Image.open(self.image_paths[idx])  # Disk I/O
        image = self.transform(image)              # CPU intensive
        image = self.augment(image)                # More CPU work
        return image, label

# Multiple workers can process different images simultaneously
```

### 2. Large Datasets from Disk

```python
# Reading from disk is slow
# Workers can prefetch while GPU computes
train_loader = DataLoader(
    ImageFolder('path/to/millions/of/images'),
    batch_size=64,
    num_workers=8  # Critical for performance!
)
```

## How to Choose num_workers

```python
# Quick benchmark to find optimal value
import time

for num_workers in [0, 2, 4, 8, 16]:
    train_loader = DataLoader(dataset,
                            batch_size=32,
                            num_workers=num_workers)

    start = time.time()
    for i, (data, target) in enumerate(train_loader):
        if i == 100:  # Test 100 batches
            break
    end = time.time()

    print(f"num_workers={num_workers}, time={end-start:.2f}s")
```

## Common Guidelines

```python
# Rule of thumb
num_workers = 4 * num_gpus  # Often good starting point

# Or based on CPU cores
import multiprocessing
num_workers = multiprocessing.cpu_count() // 2

# Platform specific
if platform == "Windows":
    num_workers = 0  # Windows has multiprocessing issues
else:
    num_workers = 4  # Linux/Mac work well
```

## Potential Issues and Solutions

### 1. Too Many Workers

```python
# Problem: Excessive RAM usage, CPU thrashing
# Each worker loads a full copy of the dataset in memory!

# Solution: Find the sweet spot
if dataset_fits_in_ram:
    num_workers = 4-8
else:
    num_workers = 2-4  # Less to avoid memory issues
```

### 2. Windows Compatibility

```python
# Windows requires special handling
if __name__ == '__main__':  # Required on Windows!
    train_loader = DataLoader(dataset, num_workers=4)
    # Training code here
```

### 3. Pin Memory for GPU

```python
# Combine with pin_memory for extra speed
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True  # Faster GPU transfer
)
```

## Complete Example

```python
import torch
from torch.utils.data import DataLoader, Dataset
import time

class MyDataset(Dataset):
    def __init__(self, size=10000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Simulate some processing time
        import numpy as np
        data = np.random.randn(3, 224, 224)  # Fake image
        label = idx % 10
        return torch.tensor(data, dtype=torch.float32), label

# Compare performance
dataset = MyDataset()

# Slow version
slow_loader = DataLoader(dataset, batch_size=32, num_workers=0)

# Fast version
fast_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True  # Keep workers alive between epochs
)

# Benchmark
for loader, name in [(slow_loader, "Single"), (fast_loader, "Multi")]:
    start = time.time()
    for i, (data, label) in enumerate(loader):
        if i == 50:
            break
        # Simulate GPU work
        _ = data.mean()
    print(f"{name} worker: {time.time() - start:.2f}s")
```

## When NOT to Use Workers

```python
# Don't use workers when:
# 1. Dataset is already in memory
if isinstance(dataset, TensorDataset):
    num_workers = 0  # Already fast

# 2. Very small datasets
if len(dataset) < 1000:
    num_workers = 0  # Overhead not worth it

# 3. Simple data (no preprocessing)
if no_augmentation and data_in_memory:
    num_workers = 0
```

## Key Takeaway

- **Default (num_workers=0)**: Fine for small datasets or when data is in memory
- **num_workers=4-8**: Significant speedup for image datasets, heavy preprocessing
- **Monitor GPU utilization**: If GPU usage is low (<90%), you might need more workers
- **Start with 4, benchmark, and adjust** based on your specific setup!

**The goal**: Keep the GPU fed with data constantly instead of waiting for the CPU to load the next batch!

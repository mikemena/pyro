
# 🧠 PyTorch Optimizer Selection Cheat Sheet

## ✅ Optimizers Overview

| Optimizer | Best For | Avoid If | Key Features |
|----------|----------|----------|---------------|
| **SGD** | Image classification (CNNs), large datasets, strong generalization | Small data, sparse features, quick prototyping | Simple, fast, tunable, often best generalization |
| **SGD + Momentum** | Most deep learning tasks (esp. computer vision) | Unstable gradients, sparse data | Momentum helps escape local minima |
| **Adam** | Transformers, NLP, GANs, quick convergence | Poor generalization, small batches | Adaptive learning rate, fast convergence |
| **AdamW** | Transformers, LLMs, fine-tuning pretrained models | Small models without need for weight decay | Decouples weight decay for better regularization |
| **RMSProp** | RNNs, noisy or non-stationary data | Tasks needing generalization | Adapts learning rate per weight; good for non-convex problems |
| **Adagrad** | Sparse data (e.g., NLP embeddings) | Long training runs (learning rate shrinks) | High adaptivity, especially useful in early training |
| **Adadelta** | Sparse data, low memory | When precise control of learning rate needed | Learning rate-free (relies on moving average) |
| **NAdam** | Same as Adam + momentum tweaks | Same as Adam | Combines Adam and Nesterov momentum |
| **LAMB** | Large-batch training (e.g., BERT pretraining) | Small models or standard batch sizes | Layer-wise adaptive moments; needs large batch sizes |
| **Lion** | Vision/Transformer models (faster with similar/better results) | Not yet well-established | Momentum-based but simpler than Adam |

## 🛠️ What Info Do You Need?

| Question | Why It Matters |
|----------|----------------|
| 🔢 Data Size | Large data works better with SGD; small favors Adam |
| 🔄 Sparse Gradients? | Use Adam or Adagrad if yes |
| ⌛ Training Time? | Use Adam/AdamW for faster convergence |
| 🧠 Architecture Type? | CNN = SGD, Transformer = AdamW, RNN = RMSProp |
| 🏋️ Weight Decay? | AdamW handles it best |
| 💥 Gradient Stability? | RMSProp/Adam preferred for unstable gradients |
| 🧪 Large Batches? | Consider LAMB or Lion |

## 🎯 Optimizers by Task

| Task | Optimizer |
|------|-----------|
| Image Classification | SGD + Momentum, AdamW |
| NLP (Transformers) | AdamW, LAMB, Lion |
| NLP (Sparse Embeddings) | Adagrad, Adam |
| RNNs | RMSProp, Adam |
| GANs | Adam (β1=0.5, β2=0.999) |
| Tabular Data | Adam, SGD |
| Reinforcement Learning | Adam, RMSProp |
| Pretraining | LAMB, AdamW, Lion |

## 🧪 Defaults

| Goal | Try This First |
|------|----------------|
| Quick Prototype | Adam (lr=1e-3) |
| Best Generalization | SGD + Momentum |
| Transformers/LLMs | AdamW |
| Sparse NLP | Adagrad |
| Large Scale | LAMB, Lion |

## 📊 Flowchart
(See attached PDF)

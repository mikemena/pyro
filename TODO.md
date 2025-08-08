1. Model Architecture & Training

[] Add train_model_v3 --> previous models are quite basic - just a simple feedforward network with ReLU and dropout. v3 adds the following:

- Batch normalization to stabilize training
- Learning rate scheduling (learning rate stays constant)
- Limited regularization techniques

2. Model Evaluation

[] Create evaluate.py with enhanced evaluation

[] Move the evaluate method from train_model_v3 to this new evaluate.py

3. Changed date format in logger.py

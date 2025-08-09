1. Model Architecture & Training

[] Add train_model_v3 --> previous models are quite basic - just a simple feedforward network with ReLU and dropout. v3 adds the following:

- Batch normalization to stabilize training
- Learning rate scheduling (learning rate stays constant)
- Limited regularization techniques

2. Model Evaluation

[] Create evaluate.py with enhanced evaluation

[] Move the evaluate method from train_model_v3 to this new evaluate.py

3. Understand these components - what are they, when to use it, when not to use it:

## Batch Normalization
 -- if your dataset is small or highly imbalanced, you might want to monitor the training stability or experiment with smaller/larger batch sizes

## Learning Rate Scheduling
-- CosineAnnealingLR
-- StepLR
-- ReduceLROnPlateau : use if if validation loss plateaus

## Gradient clipping

gradient clipping is good for stabilizing training, especially with deep networks or imbalanced data

The clipping (torch.nn.utils.clip_grad_norm_) occurs after loss.backward() and before optimizer.step()

## Regularization (ReLU, LeakyReLU, GELU, Dropout, and Weight Decay)

## Weight Decay

weight decay of 1e-05 is small, which is fine for subtle regularization, but you might want to test larger values (e.g., 1e-04 or 1e-03) to see if stronger regularization improves generalization.

## dropout_rate

high dropout rate, which can be effective for preventing overfitting but may lead to underfitting if the model is too small or the data is limited. Since you’re using a single hidden layer with 128 units, this high dropout rate might be aggressive. You could experiment with lower values e.g., 0.3 or 0.5

## hidden dims

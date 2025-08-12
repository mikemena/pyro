1. Model Architecture & Training

[] Add train_model_v3 --> previous models are quite basic - just a simple feedforward network with ReLU and dropout. v3 adds the following:

- Batch normalization to stabilize training
- Learning rate scheduling (learning rate stays constant)
- Limited regularization techniques

2. Model Evaluation

[] Create evaluate.py with enhanced evaluation

[] Move the evaluate method from train_model_v3 to this new evaluate.py

3. Understand these components - what are they, when to use it, when not to use it:

## Batch Normalization (BN)
Batch Normalization (BN) is a technique used in AI modeling—especially deep neural networks—to stabilize and speed up training by normalizing the inputs to each layer.

When to Use Batch Normalization (BN)
✅ Use BN:
✅ Large batch sizes
✅ CNNs or deep MLPs
✅ Stable training speed desired

When to Avoid It
🚫 Batch size < 8–16
🚫 RNNs/sequence models (use Layer Norm instead)
🚫 Highly variable input distributions per batch

## Learning Rate Scheduling
Learning Rate Scheduling is a technique in AI modeling where the learning rate—the size of each step the optimizer takes during training—is changed over time instead of staying constant.

Think of the learning rate like the stride of a hiker on a mountain:

Early in the hike (training), you want big steps to cover ground quickly.

Near the peak (convergence), you want small steps to avoid overshooting.

A learning rate schedule automates that adjustment.

-- CosineAnnealingLR : Learning rate follows a cosine curve, starting   high and gradually reaching near zero

-- StepLR : Reduce the learning rate by a factor every fixed number of epochs.

-- ReduceLROnPlateau : Monitor validation loss; when it stops improving, lower the learning rate.

When to Use Learning Rate Scheduling
✅ Long training runs — scheduling helps avoid wasting time at a suboptimal rate.
✅ Deep networks — especially CNNs, Transformers, or deep MLPs.
✅ When loss plateaus — good sign the LR might be too high to fine-tune further.
✅ Transfer learning / fine-tuning — often start with lower LR and decay further.
✅ When experimenting with optimizers — especially SGD, which benefits heavily from LR decay.

When to Avoid It
🚫 Small datasets or very short training — the schedule may not have enough time to make a difference.
🚫 Already using an adaptive optimizer like Adam, AdamW, or RMSprop — they adjust effective learning rates internally (though combining with schedules can still help).
🚫 When the model is underfitting badly — lowering LR too early can freeze progress.
🚫 If your initial LR is already optimal — a bad schedule can harm training instead of helping.

## Gradient clipping

Gradient Clipping is a technique used in AI modeling to prevent gradients from becoming too large during backpropagation.

When to Use Gradient Clipping
✅ Recurrent Neural Networks (RNNs, LSTMs, GRUs) — they are especially prone to exploding gradients because of long-term dependencies.
✅ Very deep networks — as depth increases, gradients can accumulate and blow up.
✅ High learning rates — which can amplify unstable gradient updates.
✅ When loss suddenly spikes — often a sign gradients are exploding.
✅ Training GANs — where generator/discriminator competition can cause unstable updates.

When to Avoid It
🚫 If you’re not experiencing exploding gradients — clipping might unnecessarily slow learning.
🚫 When the problem is vanishing gradients — clipping won’t help; you need different architectures or activation functions.
🚫 With well-tuned learning rates and batch normalization — sometimes clipping becomes redundant.
🚫 If the threshold is set too low — gradients will be constantly clipped, leading to underfitting.

## Regularization (ReLU, LeakyReLU, GELU)
Regularization in AI modeling is any technique that reduces overfitting by controlling model complexity or improving generalization.
It doesn’t always mean “making the model smaller” — sometimes it means “forcing the model to learn in a more robust way.”

ReLU (Rectified Linear Unit)

- Zeroes out negative values, keeps positives unchanged.
- Introduces non-linearity so networks can model complex relationships.
- Simple, efficient, and helps avoid the vanishing gradient problem.

When to use ReLU (Rectified Linear Unit)
✅ Training deep feedforward or convolutional networks.
✅ You want fast convergence with stable gradients.

When to Avoid It
🚫 Inputs have a lot of negative values that still carry useful information — ReLU might “kill” too many neurons (“dying ReLU” problem).
🚫 Instead, try LeakyReLU or GELU.

LeakyReLU

- Like ReLU, but negative values are scaled by a small slope (
𝛼, e.g., 0.01) instead of being zeroed.
- Prevents neurons from dying completely.

When to use LeakyReLU
✅ You’re seeing dead neurons with plain ReLU.
✅ Tasks where negative activations still carry meaningful patterns.

When to Avoid It
🚫 The added computation cost outweighs benefits (rarely a big issue).
🚫 You're working with activations that should be strictly non-negative.

GELU (Gaussian Error Linear Unit)
- Smooth, probabilistic version of ReLU.
- Retains small negative values in a smooth way
- Popular in Transformer architectures (e.g., BERT).

When to use GELU
✅ You want smoother gradients than ReLU.
✅ Using Transformer-like models or NLP tasks.

When to Avoid It
🚫 Training speed is critical (GELU is slower than ReLU).
🚫 You don’t see noticeable performance improvement over ReLU.

## Weight Decay
Weight Decay is a form of regularization in AI modeling that discourages the model’s weights from growing too large.
It works by adding a penalty term to the loss function so that, during training, the optimizer favors smaller weights.

When to Use Weight Decay
✅ Overfitting is an issue — especially with high-capacity models like large MLPs, CNNs, or Transformers.
✅ When using SGD — SGD benefits a lot from weight decay to stabilize training.
✅ For image and text models — it’s a common default in vision and NLP.
✅ With AdamW optimizer — AdamW decouples weight decay from the gradient step, making it more effective.

When to Avoid Weight Decay
🚫 When underfitting — shrinking weights further can make it harder for the model to learn.
🚫 If you already have strong regularization — heavy dropout, strong data augmentation, or early stopping might make weight decay redundant.
🚫 For bias terms or batch norm parameters — penalizing these can hurt performance (common practice is to exclude them from weight decay).
🚫 In sparse models where L1 regularization is preferred — e.g., when you want many weights to become exactly zero.

## dropout_rate

high dropout rate, which can be effective for preventing overfitting but may lead to underfitting if the model is too small or the data is limited. Since you’re using a single hidden layer with 128 units, this high dropout rate might be aggressive. You could experiment with lower values e.g., 0.3 or 0.5

## hidden dims

## grid search or random search

import torch
import matplotlib.pyplot as plt
from models.model_v2 import NeuralNetwork
from data.dataloaders import test_dataloader

# Label for fashion MNIST
class_labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Load the training model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device)

# Load te saved model weights - make sure you save them first
# If not saved yet, run this in train.py after training:
    # torch.save(model.state_dict(), "model_weights.pth")
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

# Get a batch of test images
detaiter = iter(test_dataloader)
images, labels = next(detaiter)

# Make predictions
with torch.no_grad():
    images = images.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# Convert imaged for display
images = images.cpu().numpy()

# Plot the images and predictions
plt.figure(figsize=(16, 8))
for i in range(min(16, len(images))):  # Display up to 16 images
    plt.subplot(4, 4, i+1)

    # Display image (need to reshape from [1,28,28] to [28,28])
    plt.imshow(images[i][0], cmap='gray')

    # Get ground truth and prediction
    true_label = class_labels[labels[i]]
    pred_label = class_labels[predicted[i]]

    # Set title with true and predicted labels
    title = f"True: {true_label}\nPred: {pred_label}"

    # Color the title based on correct/incorrect prediction
    color = 'green' if pred_label == true_label else 'red'
    plt.title(title, color=color)

    plt.axis('off')

plt.tight_layout()
plt.savefig('fashion_mnist_predictions.png')
plt.show()

# Calculate overall accuracy for this batch
correct = (predicted == labels.to(device)).sum().item()
total = labels.size(0)
accuracy = 100 * correct / total
print(f"Batch Accuracy: {accuracy:.2f}%")

# Show confidence scores for a single example
print("\nDetailed prediction for first image:")
image_idx = 0
probs = torch.nn.functional.softmax(outputs[image_idx], dim=0)
for i, prob in enumerate(probs):
    print(f"{class_labels[i]}: {prob:.4f} ({prob*100:.2f}%)")

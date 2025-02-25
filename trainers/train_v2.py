import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from data.dataloaders import train_dataloader, test_dataloader
from models.model_v2 import NeuralNetwork

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device)

# Use Cross Entropy Loss
loss_fn = nn.CrossEntropyLoss()

# Use Adam optimizer instead of SGD for faster convergence
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,  # Higher initial learning rate for Adam
    weight_decay=1e-5  # L2 regularization to prevent overfitting
)

# Add learning rate scheduler to reduce learning rate when validation loss plateaus
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,  # Reduce learning rate by half when plateau is detected
    patience=2,   # Wait for 2 epochs before reducing LR
    verbose="True"
)

# Store metrics for plotting
train_losses = []
test_losses = []
test_accuracies = []

def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:5d}]")

    # Calculate average training loss for this epoch
    avg_loss = total_loss / len(dataloader)
    train_losses.append(avg_loss)
    print(f"Train Avg loss: {avg_loss:>8f}")

    return avg_loss

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    test_losses.append(test_loss)

    accuracy = 100 * correct / size
    test_accuracies.append(accuracy)

    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, accuracy

def plot_metrics(train_losses, test_losses, test_accuracies, epochs):
    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs+1), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), test_accuracies, label='Test Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

# Main training loop
epochs = 15
best_accuracy = 0

print(f"Training for {epochs} epochs...")
for t in range(epochs):
    print(f"Epoch {t+1}/{epochs}\n-------------------------------")
    train_loss = train(train_dataloader, model, loss_fn, optimizer, t+1)
    test_loss, accuracy = test(test_dataloader, model, loss_fn)

    # Update learning rate based on validation loss
    scheduler.step(test_loss)

    # Save the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "best_model_weights.pth")
        print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

# Save the final model
torch.save(model.state_dict(), "final_model_weights.pth")

# Plot and save training metrics
plot_metrics(train_losses, test_losses, test_accuracies, epochs)

print(f"Training complete! Best accuracy: {best_accuracy:.2f}%")
print("Final model saved to 'final_model_weights.pth'")
print("Best model saved to 'best_model_weights.pth'")
print("Training metrics plot saved to 'training_metrics.png'")

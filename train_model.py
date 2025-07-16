import torch
import torch.nn as nn
import torch.optim as optim
from data_preprocessor import DataPreprocessor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import json
import os
from datetime import datetime

class Predictor(nn.Module):
    """Feedforward Neural Network for predicting credentialing non responders"""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super(Predictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # Build the network layers
        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Output layer (single neuron for regression)
        layers.append(nn.Linear(prev_dim, 1)) # No activation
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x).squeeze()

    def get_model_info(self):
        """Return model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }

class ModelTrainer:
    """Handles model training, validation, and evaluation"""
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, train_loader, val_loader, epochs=100, lr=0.001,
              patience=15, min_delta=1e-4, save_path='best_model.pt'):
        """Train the model with early stopping"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"Training on {self.device}")
        print(f"Model Info: {self.model.get_model_info()}")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_loss = self.validate(val_loader, criterion)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(current_lr)

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'model_config': self.model.get_model_info()
                }, save_path)
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or patience_counter == 0:
                print(f'Epoch [{epoch+1}/{epochs}] - '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, '
                      f'LR: {current_lr:.6f}')

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

        return {
            'best_val_loss': best_val_loss,
            'final_epoch': epoch + 1,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch_y.cpu().numpy())

        predictions = np.array(predictions)
        targets = np.array(targets)

        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)

        metrics = {'mse': mse, 'mae': mae, 'rmse': rmse, 'r2_score': r2}

        print("Test Set Evaluation:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R² Score: {r2:.4f}")

        return metrics, predictions, targets

    def plot_training_history(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.learning_rates)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_predictions(self, predictions, targets):
        """Plot predictions vs actual values"""
        plt.figure(figsize=(10, 8))

        plt.subplot(2, 2, 1)
        plt.scatter(targets, predictions, alpha=0.6)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        plt.xlabel('Actual Grades')
        plt.ylabel('Predicted Grades')
        plt.title('Predictions vs Actual')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        residuals = predictions - targets
        plt.scatter(targets, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Actual Grades')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=20, alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.hist(targets, bins=15, alpha=0.7, label='Actual', color='blue')
        plt.hist(predictions, bins=15, alpha=0.7, label='Predicted', color='red')
        plt.xlabel('Grades')
        plt.ylabel('Frequency')
        plt.title('Grade Distribution Comparison')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    """Create DataLoaders for training, validation, and testing"""
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def load_dataset(file_path):
    # Load the preprocessor state
    preprocessor = DataPreprocessor()
    preprocessor.load_state()

    df = pd.read_excel(file_path)
    X = df.drop('non_responder', axis=1).values
    y = df['non_responder'].values

<<<<<<< HEAD
    #Encode y if its a binary/categorical target
    if preprocessor.target_type in ['binary','categorical']:
        if preprocessor.target_label_encoder is None:
            raise ValueError("target LabelEncoder not loaded.Ensure preprocessor state is saved correctly.")
=======
    # Encode if binary or categorical target
    if preprocessor.target_type in ['binary', 'categorical']:
        if preprocessor.target_label_encoder is None:
            raise ValueError("Target LabelEncoder not loaded. Ensure preprocessor state is saved.")
>>>>>>> onehotencoder
        y = preprocessor.target_label_encoder.transform(y)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1,1)
    return X_tensor, y_tensor

def main():
    # Get datasets and state file
    preprocessing_artifacts_dir = 'preprocessing_artifacts'
    state_file = os.path.join(preprocessing_artifacts_dir, 'preprocessor_state.json')
    train_file = os.path.join(preprocessing_artifacts_dir, 'cred_data_train_processed.xlsx')
    val_file = os.path.join(preprocessing_artifacts_dir, 'cred_data_val_processed.xlsx')
    test_file = os.path.join(preprocessing_artifacts_dir, 'cred_data_test_processed.xlsx')

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Check if preprocessing artifacts exist
    print("\n1. CHECKING PREPROCESSING ARTIFACTS...")
    if not all(os.path.exists(f) for f in [state_file,train_file, val_file,test_file]):
        print("One or more preprocessing artifacts are missing. Run the preprocessing step first")

    # Load processed data from Excel
    print("\n2. LOADING PROCESSED DATA FROM EXCEL...")
    X_train, y_train = load_dataset(train_file)
    X_val, y_val = load_dataset(val_file)
    X_test, y_test = load_dataset(test_file)

    # Create data loaders
    print("\n3. CREATING DATA LOADERS...")
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32
    )
    print("Data loaders created successfully!")

    # Create model
    print("\n4. CREATING MODEL...")
    input_dim = X_train.shape[1]
    model = Predictor(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        dropout_rate=0.3
    )
    print(f"Model created with {input_dim} input features!")

    # Training
    print("\n5. TRAINING MODEL...")
    os.makedirs('models', exist_ok=True)
    trainer = ModelTrainer(model)

    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        lr=0.001,
        patience=15,
        save_path='models/best_student_grade_model.pt'
    )

    # Evaluation
    print("\n6. EVALUATING MODEL...")
    metrics, predictions, targets = trainer.evaluate(test_loader)

    # Visualization
    print("\n7. GENERATING PLOTS...")
    try:
        trainer.plot_training_history()
        trainer.plot_predictions(predictions, targets)
    except Exception as e:
        print(f"Plotting skipped (display issues): {e}")

    # Save results
    print("\n8. SAVING RESULTS...")
    results = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'model_config': model.get_model_info(),
        'training_results': training_results,
        'test_metrics': metrics,
        # 'data_source': processed_excel,
        'preprocessing_artifacts': state_file
    }

    with open('models/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Generated files:")
    print(f"   🤖 Model: models/best_student_grade_model.pt")
    print(f"   📊 Results: models/training_results.json")
    print(f"\nModel Performance:")
    print(f"   📈 R² Score: {metrics['r2_score']:.4f}")
    print(f"   📉 RMSE: {metrics['rmse']:.4f}")
    print(f"   📊 MAE: {metrics['mae']:.4f}")

    return model, trainer, results

if __name__ == "__main__":
    result = main()
    if result:
        model, trainer, results = result
        print("\n✨ Training pipeline completed successfully!")
        print("Your model is ready for inference using the same preprocessing artifacts.")
    else:
        print("\n❌ Training failed. Please fix the issues above and try again.")

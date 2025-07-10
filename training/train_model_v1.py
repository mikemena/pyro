import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import json
import os
from datetime import datetime

class Predictor(nn.Module):
    """Simple Feedforward Neural Network for predicting student grades"""
    def __init__(self, input_dim, hidden_dims=[64], dropout_rate=0.0):
        super(Predictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # Build the network layers (simplified: no batch norm for starting point)
        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Output layer (single neuron for regression)
        layers.append(nn.Linear(prev_dim, 1))
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

    def train(self, train_loader, val_loader, epochs=50, lr=0.001, weight_decay=0.0,
              patience=10, min_delta=1e-4, save_path='best_model.pt'):
        """Train the model with early stopping (simplified: no scheduler, no clipping)"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"Training on {self.device}")
        print(f"Model Info: {self.model.get_model_info()}")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_loss = self.validate(val_loader, criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

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

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

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

    # (Optional: Keep or remove plotting functions as needed)
    def plot_training_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_predictions(self, predictions, targets):
        plt.figure(figsize=(8, 6))
        plt.scatter(targets, predictions, alpha=0.6)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        plt.xlabel('Actual Grades')
        plt.ylabel('Predicted Grades')
        plt.title('Predictions vs Actual')
        plt.grid(True)
        plt.show()

def load_dataset(file_path):
    df = pd.read_excel(file_path)
    X = df.drop('non_responder', axis=1).values
    y = df['non_responder'].values
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1,1)
    #y_tensor = torch.tensor(y, dtype=torch.float32)  # Fixed: Keep 1D [N] for scalar regression
    return X_tensor, y_tensor

def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    """Create DataLoaders for training, validation, and testing"""
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def main():
    # Make paths relative to the script's location (robust to cwd changes)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessing_artifacts_dir = os.path.join(script_dir, '../preprocessing_artifacts')  # Go up one level as per your note

    # Debug prints to verify paths
    print("\nDebug Info:")
    print("Script directory:", script_dir)
    print("Current working directory:", os.getcwd())
    print("Preprocessing artifacts directory:", preprocessing_artifacts_dir)
    if os.path.exists(preprocessing_artifacts_dir):
        print("Files in artifacts dir:", os.listdir(preprocessing_artifacts_dir))
    else:
        print("Artifacts dir does not exist at the path above.")

    state_file = os.path.join(preprocessing_artifacts_dir, 'preprocessor_state.json')
    train_file = os.path.join(preprocessing_artifacts_dir, 'cred_data_train_processed.xlsx')
    val_file = os.path.join(preprocessing_artifacts_dir, 'cred_data_val_processed.xlsx')
    test_file = os.path.join(preprocessing_artifacts_dir, 'cred_data_test_processed.xlsx')

    # Check each file explicitly
    files_to_check = [state_file, train_file, val_file, test_file]
    missing_files = [f for f in files_to_check if not os.path.exists(f)]
    if missing_files:
        print(f"Missing files: {missing_files}")
        print("Run the preprocessing step (e.g., prepare.py or similar) to generate them, or adjust the paths if incorrect.")
        return None

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load processed data from Excel
    print("\n2. LOADING PROCESSED DATA FROM EXCEL...")
    X_train, y_train = load_dataset(train_file)
    X_val, y_val = load_dataset(val_file)
    X_test, y_test = load_dataset(test_file)

    input_dim = X_train.shape[1]

    # Hyperparameter Tuning Setup (uncomment one section at a time, run sequentially)
    # Start with simple defaults
    default_hidden_dims = [64]
    default_dropout = 0.0
    default_weight_decay = 0.0
    default_batch_size = 32
    default_optimizer = 'Adam'

    # 1. Tune Learning Rate (fix others to defaults)
    print("\nTUNING LEARNING RATE...")
    lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    best_lr = None
    best_val_loss = float('inf')
    for lr in lrs:
        print(f"  Trying LR: {lr}")
        model = Predictor(input_dim=input_dim, hidden_dims=default_hidden_dims, dropout_rate=default_dropout)
        trainer = ModelTrainer(model)
        save_path = f'models/trial_lr_{lr}.pt'
        os.makedirs('models', exist_ok=True)
        results = trainer.train(train_loader=create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=default_batch_size)[0],
                                val_loader=create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=default_batch_size)[1],
                                epochs=50, lr=lr, weight_decay=default_weight_decay, save_path=save_path)
        if results['best_val_loss'] < best_val_loss:
            best_val_loss = results['best_val_loss']
            best_lr = lr
    print(f"Best LR: {best_lr} (val loss: {best_val_loss:.4f})")
    # After running, hardcode best_lr below (e.g., lr = 0.001) and proceed to next tuning.

    # # 2. Tune Architecture (fix best LR from above, others default)
    # print("\nTUNING ARCHITECTURE...")
    # lr = best_lr  # Replace with your best from step 1
    # hidden_sizes = [64, 128, 256, 512]
    # num_layers_list = [1, 2, 3, 4]
    # best_hidden_dims = None
    # best_val_loss = float('inf')
    # for num_layers in num_layers_list:
    #     for size in hidden_sizes:
    #         hidden_dims = [size] * num_layers  # Uniform size per layer for simplicity
    #         print(f"  Trying hidden_dims: {hidden_dims}")
    #         model = Predictor(input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=default_dropout)
    #         trainer = ModelTrainer(model)
    #         save_path = f'models/trial_arch_{num_layers}_{size}.pt'
    #         results = trainer.train(train_loader=create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=default_batch_size)[0],
    #                                 val_loader=create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=default_batch_size)[1],
    #                                 epochs=50, lr=lr, weight_decay=default_weight_decay, save_path=save_path)
    #         if results['best_val_loss'] < best_val_loss:
    #             best_val_loss = results['best_val_loss']
    #             best_hidden_dims = hidden_dims
    # print(f"Best hidden_dims: {best_hidden_dims} (val loss: {best_val_loss:.4f})")
    # # Hardcode best_hidden_dims for next step.

    # # 3. Tune Regularization (fix best LR + architecture from above)
    # print("\nTUNING REGULARIZATION...")
    # lr = best_lr  # From step 1
    # hidden_dims = best_hidden_dims  # From step 2
    # dropouts = [0.1, 0.3, 0.5, 0.7]
    # weight_decays = [1e-5, 1e-4, 1e-3]
    # best_dropout = None
    # best_weight_decay = None
    # best_val_loss = float('inf')
    # for dropout in dropouts:
    #     for wd in weight_decays:
    #         print(f"  Trying dropout: {dropout}, weight_decay: {wd}")
    #         model = Predictor(input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=dropout)
    #         trainer = ModelTrainer(model)
    #         save_path = f'models/trial_reg_{dropout}_{wd}.pt'
    #         results = trainer.train(train_loader=create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=default_batch_size)[0],
    #                                 val_loader=create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=default_batch_size)[1],
    #                                 epochs=50, lr=lr, weight_decay=wd, save_path=save_path)
    #         if results['best_val_loss'] < best_val_loss:
    #             best_val_loss = results['best_val_loss']
    #             best_dropout = dropout
    #             best_weight_decay = wd
    # print(f"Best dropout: {best_dropout}, weight_decay: {best_weight_decay} (val loss: {best_val_loss:.4f})")
    # # Hardcode for next step.

    # # 4. Tune Training Parameters (fix all best from above)
    # print("\nTUNING TRAINING PARAMETERS...")
    # lr = best_lr  # From step 1
    # hidden_dims = best_hidden_dims  # From step 2
    # dropout = best_dropout  # From step 3
    # weight_decay = best_weight_decay  # From step 3
    # batch_sizes = [16, 32, 64, 128]
    # optimizers = ['Adam', 'SGD', 'RMSprop']  # Assumed common choices
    # best_batch_size = None
    # best_optimizer = None
    # best_val_loss = float('inf')
    # for bs in batch_sizes:
    #     for opt_name in optimizers:
    #         print(f"  Trying batch_size: {bs}, optimizer: {opt_name}")
    #         model = Predictor(input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=dropout)
    #         trainer = ModelTrainer(model)
    #         # Override optimizer in train method if needed, but for simplicity, assume Adam always (or modify train to take opt_name)
    #         save_path = f'models/trial_train_{bs}_{opt_name}.pt'
    #         results = trainer.train(train_loader=create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=bs)[0],
    #                                 val_loader=create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=bs)[1],
    #                                 epochs=50, lr=lr, weight_decay=weight_decay, save_path=save_path)
    #         if results['best_val_loss'] < best_val_loss:
    #             best_val_loss = results['best_val_loss']
    #             best_batch_size = bs
    #             best_optimizer = opt_name
    # print(f"Best batch_size: {best_batch_size}, optimizer: {best_optimizer} (val loss: {best_val_loss:.4f})")

    # Final training with best params (uncomment after all tuning)
    # print("\nFINAL TRAINING WITH BEST PARAMS...")
    # model = Predictor(input_dim=input_dim, hidden_dims=best_hidden_dims, dropout_rate=best_dropout)
    # trainer = ModelTrainer(model)
    # train_loader, val_loader, test_loader = create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=best_batch_size)
    # training_results = trainer.train(train_loader, val_loader, epochs=50, lr=best_lr, weight_decay=best_weight_decay, save_path='models/best_final_model.pt')
    # metrics, predictions, targets = trainer.evaluate(test_loader)
    # trainer.plot_training_history()
    # trainer.plot_predictions(predictions, targets)

    # # Save final results
    # results = {
    #     'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
    #     'model_config': model.get_model_info(),
    #     'training_results': training_results,
    #     'test_metrics': metrics,
    #     'preprocessing_artifacts': state_file,
    #     'best_params': {'lr': best_lr, 'hidden_dims': best_hidden_dims, 'dropout': best_dropout, 'weight_decay': best_weight_decay, 'batch_size': best_batch_size, 'optimizer': best_optimizer}
    # }
    # with open('models/final_results.json', 'w') as f:
    #     json.dump(results, f, indent=2)
    # print("\n🎉 FINAL TRAINING COMPLETED!")

    return model, trainer, results

if __name__ == "__main__":
    result = main()
    if result:
        print("\n✨ Pipeline completed!")
    else:
        print("\n❌ Failed. Fix issues and retry.")

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessor import DataPreprocessor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, recall_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from datetime import datetime
import json
from visualize import ModelVisualizer

class Predictor(nn.Module):
    """Simple Feedforward Neural Network for predicting classification target"""
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
    def __init__(self, model, device=None, class_weights=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.train_losses = []
        self.val_losses = []
        self.class_weights = class_weights
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(batch_x)
            if self.class_weights is not None:
                # Apply per-sample weights
                weight = self.class_weights[batch_y.long()]
                loss = criterion(outputs, batch_y)
                loss = (loss * weight).mean()  # Apply weights and compute mean loss
            else:
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
                if self.class_weights is not None:
                    # Apply per-sample weights
                    weight = self.class_weights[batch_y.long()]
                    loss = criterion(outputs, batch_y)
                    loss = (loss * weight).mean()  # Apply weights and compute mean loss
                else:
                    loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, train_loader, val_loader, epochs=50, lr=0.001, weight_decay=0.0,
              patience=10, min_delta=1e-4, save_path='best_model.pt', optimizer_name='Adam'):
        """Train the model with early stopping (simplified: no scheduler, no clipping)"""
        # Regression target
        # criterion = nn.MSELoss()
        # binary classification
        # Set reduction='none' for class weights to allow per-sample weighting
        criterion = nn.BCEWithLogitsLoss(reduction='none' if self.class_weights is not None else 'mean')

        if optimizer_name == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"Training on {self.device}")
        print(f"Model Info: {self.model.get_model_info()}")
        if self.class_weights is not None:
            print(f"Using class weights: {self.class_weights.tolist()}")

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
                    'model_config': self.model.get_model_info(),
                    'class_weights': self.class_weights.tolist() if self.class_weights is not None else None
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
                logits = self.model(batch_x)
                probabilities = torch.sigmoid(logits)
                predictions.extend(probabilities.cpu().numpy())
                targets.extend(batch_y.cpu().numpy())

        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()

        # Apply threshold to get binary predictions
        binary_preds = (predictions >= 0.5).astype(int)

        # Classification metrics
        acc = accuracy_score(targets, binary_preds)
        f1 = f1_score(targets, binary_preds)
        try:
            auc = roc_auc_score(targets, predictions) # Use raw scores for AUC
        except ValueError:
            auc = None # Handle case where only one class is present in y_true

        cm = confusion_matrix(targets, binary_preds)

        # Percentage of negatives caught - True Negative Rate
        tn, fp, fn, tp = cm.ravel() #Extract TN, FP, FN, TP
        tnr = tn / (tn + fp)

        # Percentage of positives caught - True Positive Rate
        recall = recall_score(targets, binary_preds)

        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)

        metrics = {
            'accuracy': acc,
            'f1_score': f1,
            'roc_auc': auc,
            'confusion_matrix': cm,
            'recall': recall,
            'tnr': tnr
            # 'mse': mse,
            # 'mae': mae,
            # 'rmse': rmse,
            # 'r2_score': r2
        }

        print("Test Set Evaluation:")
        print(f"  TNR: {tnr:.2%}")
        print(f"  Recall: {recall:.2%}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  f1 Score: {f1:.4f}")
        print(f"  roc auc: {auc:.4f}")
        print(f"  Confusion Matrix: {cm}")

        return metrics, predictions, targets

def load_dataset(file_path):
    # Load the preprocessor state
    preprocessor = DataPreprocessor()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    state_file = os.path.join(root_dir, 'preprocessing_artifacts', 'preprocessor_state.json')
    preprocessor.load_state(state_file)

    df = pd.read_excel(file_path)
    X = df.drop(['Status', 'temp_index'], axis=1, errors='ignore').values
    y = df['Status'].values

    # Encode y if binary or categorical target
    if preprocessor.target_type in ['binary', 'categorical']:
        if preprocessor.target_label_encoder is None:
            raise ValueError("Target LabelEncoder not loaded. Ensure preprocessor state is saved.")
        y = preprocessor.target_label_encoder.transform(y)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor, y

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
    # Make paths relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessing_artifacts_dir = os.path.join(script_dir, '../preprocessing_artifacts')

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
    train_file = os.path.join(preprocessing_artifacts_dir, 'loan_default_train_processed.xlsx')
    val_file = os.path.join(preprocessing_artifacts_dir, 'loan_default_val_processed.xlsx')
    test_file = os.path.join(preprocessing_artifacts_dir, 'loan_default_test_processed.xlsx')

    # Check each file explicitly
    files_to_check = [state_file, train_file, val_file, test_file]
    missing_files = [f for f in files_to_check if not os.path.exists(f)]
    if missing_files:
        print(f"Missing files: {missing_files}")
        print("Run the preprocessing step (e.g., prepare.py or similar) to generate them, or adjust the paths if incorrect.")
        return None

    # Load preprocessor state to get feature names for plotting
    with open(state_file, 'r') as f:
        preprocessor_state = json.load(f)
    feature_names = preprocessor_state['feature_columns']

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load processed data from Excel
    print("\n2. LOADING PROCESSED DATA FROM EXCEL...")
    X_train, y_train, y_train_raw = load_dataset(train_file)
    X_val, y_val, _ = load_dataset(val_file)
    X_test, y_test, _ = load_dataset(test_file)

    # Compute class weights for handling class imbalance
    print("\nComputing class weights...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_raw), y=y_train_raw)
    print(f"Class weights: {class_weights}")

    input_dim = X_train.shape[1]

    # Hyperparameter Tuning Setup (uncomment one section at a time, run sequentially)
    # Start with simple defaults
    default_hidden_dims = [64]
    default_dropout = 0.0
    default_weight_decay = 0.0
    default_batch_size = 32
    default_optimizer = 'Adam'

    # 1. Tune Learning Rate (fix others to defaults)

    # print("\nTUNING LEARNING RATE...")
    # lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-5]
    # best_lr = None # Replace with your best from step 1
    # best_val_loss = float('inf')
    # for lr in lrs:
    #     print(f"  Trying LR: {lr}")
    #     model = Predictor(input_dim=input_dim, hidden_dims=default_hidden_dims, dropout_rate=default_dropout)
    #     trainer = ModelTrainer(model)
    #     save_path = f'models/trial_lr_{lr}.pt'
    #     os.makedirs('models', exist_ok=True)
    #     results = trainer.train(train_loader=create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=default_batch_size)[0],
    #                             val_loader=create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=default_batch_size)[1],
    #                             epochs=50, lr=lr, weight_decay=default_weight_decay, save_path=save_path, optimizer_name=default_optimizer)
    #     if results['best_val_loss'] < best_val_loss:
    #         best_val_loss = results['best_val_loss']
    #         best_lr = lr
    # print(f"Best LR: {best_lr} (val loss: {best_val_loss:.4f})")

    # After running, hardcode best_lr below (e.g., lr = 0.001) and proceed to next tuning.

    # 2. Tune Architecture (fix best LR from above, others default)

    # print("\nTUNING ARCHITECTURE...")
    # lr = 0.0005 # Replace with your best from step 1
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
    #                                 epochs=50, lr=lr, weight_decay=default_weight_decay, save_path=save_path, optimizer_name=default_optimizer)
    #         if results['best_val_loss'] < best_val_loss:
    #             best_val_loss = results['best_val_loss']
    #             best_hidden_dims = hidden_dims
    # print(f"Best hidden_dims: {best_hidden_dims} (val loss: {best_val_loss:.4f})")

    # Hardcode best_hidden_dims for next step.

    # 3. Tune Regularization (fix best LR + architecture from above)

    # print("\nTUNING REGULARIZATION...")
    # lr = 0.0005 # From step 1
    # hidden_dims = [128]  # From step 2
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
    #                                 epochs=50, lr=lr, weight_decay=wd, save_path=save_path, optimizer_name=default_optimizer)
    #         if results['best_val_loss'] < best_val_loss:
    #             best_val_loss = results['best_val_loss']
    #             best_dropout = dropout
    #             best_weight_decay = wd
    # print(f"Best dropout: {best_dropout}, weight_decay: {best_weight_decay} (val loss: {best_val_loss:.4f})")

    # Hardcode for next step.

    # 4. Tune Training Parameters (fix all best from above)

    # print("\nTUNING TRAINING PARAMETERS...")
    # lr = 0.0005 # From step 1
    # hidden_dims = [128]  # From step 2
    # dropout = 0.7  # From step 3
    # weight_decay = 1e-05 # From step 3
    # batch_sizes = [16, 32, 64, 128]
    # optimizers = ['Adam', 'AdamW']
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
    #                                 epochs=50, lr=lr, weight_decay=weight_decay, save_path=save_path, optimizer_name=opt_name)
    #         if results['best_val_loss'] < best_val_loss:
    #             best_val_loss = results['best_val_loss']
    #             best_batch_size = bs
    #             best_optimizer = opt_name
    # print(f"Best batch_size: {best_batch_size}, Best optimizer: {best_optimizer} (val loss: {best_val_loss:.4f})")

    # Final training with best params (uncomment after all tuning)

    print("\nFINAL TRAINING WITH BEST PARAMS...")
    model = Predictor(input_dim=input_dim, hidden_dims=[128], dropout_rate=0.7)
    trainer = ModelTrainer(model, class_weights=class_weights)
    train_loader, val_loader, test_loader = create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64)
    training_results = trainer.train(train_loader, val_loader, epochs=50, lr=0.0005, weight_decay=1e-05, save_path='models/best_final_model.pt', optimizer_name='AdamW')
    metrics, predictions, targets = trainer.evaluate(test_loader)

    # Initialize visualizer
    visualizer = ModelVisualizer(save_dir='plots')

    # Generate visualizations
    visualizer.plot_training_history(trainer.train_losses, trainer.val_losses, display=False, save=True)
    visualizer.plot_confusion_matrix(targets, predictions, display=False, save=True)
    visualizer.plot_data_distribution(X_train.numpy(), feature_names, display=False, save=True)
    visualizer.plot_prediction_distribution(predictions, targets, display=False, save=True)
    visualizer.plot_roc_curve(targets, predictions, display=False, save=True)
    visualizer.plot_precision_recall_curve(targets, predictions, display=False, save=True)
    visualizer.plot_metrics_bar(metrics, display=False, save=True)

    # Save final results

    results = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'model_config': model.get_model_info(),
        'training_results': training_results,
        'test_metrics': {
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc'],
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
            'mse': metrics['mse'],
            'mae': metrics['mae'],
            'rmse': float(metrics['rmse']),
            'r2_score': metrics['r2_score']
        },
        'preprocessing_artifacts': state_file,
        'best_params': {'lr': 0.0005, 'hidden_dims': [128], 'dropout': 0.7, 'weight_decay': 1e-05, 'batch_size': 64, 'optimizer': 'AdamW'}
    }
    print(f"Results: {results}")

    with open('models/final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n🎉 FINAL TRAINING COMPLETED!")

    return model, trainer, results

if __name__ == "__main__":
    result = main()
    if result:
        print("\n✨ Pipeline completed!")
    else:
        print("\n❌ Failed. Fix issues and retry.")

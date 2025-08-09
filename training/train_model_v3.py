import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import setup_logger
from data_preprocessor import DataPreprocessor
from evaluate import ModelEvaluator
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from datetime import datetime
import json

logger = setup_logger(__name__, include_location=True)


class Predictor(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dims=[128, 64],
        dropout_rate=0.3,
        use_batch_norm=True,
        activation="relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        self.use_batch_norm = use_batch_norm
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())

            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()

    def get_model_info(self):
        """Return model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }


class ModelTrainer:

    def __init__(self, model, device=None, class_weights=None):
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.train_losses = []
        self.val_losses = []
        self.class_weights = class_weights
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(
                self.device
            )
        self.scheduler = None
        self.best_model_state = None

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

    def train(
        self,
        train_loader,
        val_loader,
        epochs=100,
        lr=0.001,
        use_scheduler=True,
        scheduler_type="cosine",
        weight_decay=0.0,  # L2 regularization to prevent overfitting
        patience=10,  # stop if validation loss plateaus
        min_delta=1e-4,  # define what counts as 'improvements'
        save_path="best_model.pt",
        optimizer_name="Adam",
    ):
        # Optimizer

        if optimizer_name == "Adam":
            optimizer = optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            logger.error(f"Unsupported optimizer: {optimizer_name}")
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Learning rate scheduler

        if use_scheduler:
            if scheduler_type == "cosine":
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=lr * 0.01
                )
            if scheduler_type == "step":
                self.scheduler = optim.lr_scheduler.StepLR(
                    optimizer, step_size=20, gamma=0.5
                )
            if scheduler_type == "plateau":
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=5, factor=0.5
                )

        criterion = nn.BCEWithLogitsLoss(
            reduction="none" if self.class_weights is not None else "mean"
        )

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Training on {self.device}")
        logger.info(f"Model Info: {self.model.get_model_info()}")

        if self.class_weights is not None:
            logger.info(f"Using class weights: {self.class_weights.tolist()}")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_loss = self.validate(val_loader, criterion)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            if self.scheduler:
                if scheduler_type == "plateau":
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "model_config": self.model.get_model_info(),
                        "class_weights": (
                            self.class_weights.tolist()
                            if self.class_weights is not None
                            else None
                        ),
                    },
                    save_path,
                )
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")

        return {
            "best_val_loss": best_val_loss,
            "final_epoch": epoch + 1,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }


def load_dataset(file_path):
    # Load the preprocessor state
    preprocessor = DataPreprocessor()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    state_file = os.path.join(
        root_dir, "preprocessing_artifacts", "preprocessor_state.json"
    )
    preprocessor.load_state(state_file)

    df = pd.read_excel(file_path)
    X = df.drop(["personal_loan", "temp_index"], axis=1, errors="ignore").values
    y = df["personal_loan"].values

    # Encode y if binary or categorical target
    if preprocessor.target_type in ["binary", "categorical"]:
        if preprocessor.target_label_encoder is None:
            logger.error(
                "Target LabelEncoder not loaded. Ensure preprocessor state is saved."
            )
            raise ValueError(
                "Target LabelEncoder not loaded. Ensure preprocessor state is saved."
            )
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
    preprocessing_artifacts_dir = os.path.join(script_dir, "../preprocessing_artifacts")

    # Debug prints to verify paths
    logger.info("\nDebug Info:")
    logger.info(f"Script directory: {script_dir}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Preprocessing artifacts directory: {preprocessing_artifacts_dir}")

    if os.path.exists(preprocessing_artifacts_dir):
        logger.info(
            f"Files in artifacts dir: {os.listdir(preprocessing_artifacts_dir)}"
        )
    else:
        logger.error("Artifacts dir does not exist at the path above.")

    state_file = os.path.join(preprocessing_artifacts_dir, "preprocessor_state.json")
    train_file = os.path.join(
        preprocessing_artifacts_dir, "bank_loans_train_processed.xlsx"
    )
    val_file = os.path.join(
        preprocessing_artifacts_dir, "bank_loans_val_processed.xlsx"
    )
    test_file = os.path.join(
        preprocessing_artifacts_dir, "bank_loans_test_processed.xlsx"
    )

    # Check each file explicitly
    files_to_check = [state_file, train_file, val_file, test_file]
    missing_files = [f for f in files_to_check if not os.path.exists(f)]
    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        logger.error(
            "Run the preprocessing step (e.g., prepare.py or similar) to generate them, or adjust the paths if incorrect."
        )
        return None

    # Load preprocessor state to get feature names for plotting
    with open(state_file, "r") as f:
        preprocessor_state = json.load(f)
    feature_names = preprocessor_state["feature_columns"]

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load processed data from Excel
    logger.info("\n2. LOADING PROCESSED DATA FROM EXCEL...")
    X_train, y_train, y_train_raw = load_dataset(train_file)
    X_val, y_val, _ = load_dataset(val_file)
    X_test, y_test, _ = load_dataset(test_file)

    # Compute class weights for handling class imbalance
    logger.info("\nComputing class weights...")
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train_raw), y=y_train_raw
    )
    logger.info(f"Class weights: {class_weights}")

    input_dim = X_train.shape[1]

    # network starts here...

    logger.info("\nFINAL TRAINING WITH BEST PARAMS...")
    model = Predictor(
        input_dim=input_dim,
        hidden_dims=[128],
        dropout_rate=0.3,
        use_batch_norm=True,
        activation="gelu",
    )
    trainer = ModelTrainer(model, class_weights=class_weights)
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64
    )
    training_results = trainer.train(
        train_loader,
        val_loader,
        epochs=50,
        lr=0.0005,
        weight_decay=1e-05,
        save_path="models/best_final_model.pt",
        optimizer_name="AdamW",
        use_scheduler=True,
        scheduler_type="cosine",
    )

    evaluator = ModelEvaluator(
        model=model, device=trainer.device, save_dir="evaluation_results"
    )
    metrics, predictions, probabilities, targets = evaluator.evaluate(
        test_loader, feature_names=feature_names
    )

    # Generate visualizations
    evaluator.plot_training_history(
        trainer.train_losses, trainer.val_losses, display=False, save=True
    )

    evaluator.plot_data_distribution(
        X_train.numpy(), feature_names, display=False, save=True
    )
    evaluator.plot_prediction_distribution(
        predictions, targets, display=False, save=True
    )

    # Save final results

    results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_config": model.get_model_info(),
        "training_results": training_results,
        "test_metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "f1_score": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "recall": metrics["recall"],
            "true_positives": metrics["true_positives"],
        },
        "preprocessing_artifacts": state_file,
        "best_params": {
            "lr": 0.0005,
            "hidden_dims": [128],
            "dropout": 0.3,
            "weight_decay": 1e-05,
            "batch_size": 64,
            "optimizer": "AdamW",
            "scheduler_type": "cosine",
            "activation": "gelu",
        },
    }
    logger.info(f"Results: {results}")

    with open("models/final_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\n🎉 FINAL TRAINING COMPLETED!")

    return model, trainer, results


if __name__ == "__main__":
    result = main()
    if result:
        logger.info("\n✨ Pipeline completed!")
    else:
        logger.info("\n❌ Failed. Fix issues and retry.")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import pandas as pd
import json
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch.nn.functional as F

# Add path for imports (same as v3)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import setup_logger
from data_preprocessor import DataPreprocessor
from evaluate import ModelEvaluator

logger = setup_logger(__name__, include_location=True)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ImprovedPredictor(nn.Module):
    """Enhanced neural network for imbalanced classification"""

    def __init__(
        self,
        input_dim,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.4,
        use_batch_norm=True,
        activation="swish",
        use_residual=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual

        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)

        # Build hidden layers with residual connections
        self.layers = nn.ModuleList()
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layer_block = nn.ModuleList()

            # Linear layer
            layer_block.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if use_batch_norm:
                layer_block.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            if activation == "relu":
                layer_block.append(nn.ReLU())
            elif activation == "leaky_relu":
                layer_block.append(nn.LeakyReLU(0.01))
            elif activation == "gelu":
                layer_block.append(nn.GELU())
            elif activation == "swish":
                layer_block.append(nn.SiLU())

            # Dropout
            layer_block.append(nn.Dropout(dropout_rate))

            self.layers.append(layer_block)

            # Residual connection projection if dimensions don't match
            if use_residual and prev_dim != hidden_dim:
                setattr(self, f"residual_proj_{i}", nn.Linear(prev_dim, hidden_dim))

            prev_dim = hidden_dim

        # Output layer with additional regularization
        self.output_layer = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(prev_dim // 2, 1),
        )

    def forward(self, x):
        # Input normalization
        x = self.input_norm(x)

        # Pass through hidden layers with residual connections
        for i, layer_block in enumerate(self.layers):
            residual = x

            # Apply layer operations
            for module in layer_block:
                x = module(x)

            # Add residual connection if enabled and dimensions match
            if self.use_residual:
                if hasattr(self, f"residual_proj_{i}"):
                    residual = getattr(self, f"residual_proj_{i}")(residual)
                if x.shape == residual.shape:
                    x = x + residual

        # Output layer
        x = self.output_layer(x)
        return x.squeeze()

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


class AdvancedModelTrainer:
    """Enhanced trainer with multiple strategies for imbalanced data"""

    def __init__(self, model, device=None, loss_type="focal", alpha=0.25, gamma=2.0):
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Choose loss function
        if loss_type == "focal":
            self.criterion = FocalLoss(alpha=alpha, gamma=gamma)
        elif loss_type == "weighted_bce":
            self.criterion = None
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.loss_type = loss_type
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.best_threshold = 0.5

    def find_optimal_threshold(self, val_loader):
        """Find optimal threshold using validation set"""
        self.model.eval()
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(batch_y.cpu().numpy())

        # Convert to numpy arrays
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)

        # Find threshold that maximizes F1 score
        precision, recall, thresholds = precision_recall_curve(all_targets, all_probs)

        # Remove the last threshold (which is always problematic)
        if len(thresholds) > 0:
            precision = precision[:-1]
            recall = recall[:-1]

        # Calculate F1 scores, avoiding division by zero
        f1_scores = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) != 0,
        )

        if len(f1_scores) > 0 and np.max(f1_scores) > 0:
            best_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[best_idx]
            # Ensure threshold is reasonable (between 0.01 and 0.99)
            optimal_threshold = np.clip(optimal_threshold, 0.01, 0.99)
            logger.info(
                f"Found optimal threshold: {optimal_threshold:.3f} with F1: {f1_scores[best_idx]:.3f}"
            )
            return optimal_threshold
        else:
            logger.warning("Could not find optimal threshold, using 0.5")
            return 0.5

    def train_epoch(self, train_loader, optimizer, class_weights=None):
        """Train for one epoch with enhanced techniques"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(batch_x)

            # Choose loss calculation based on type
            if self.loss_type == "focal":
                loss = self.criterion(outputs, batch_y)
            elif self.loss_type == "weighted_bce" and class_weights is not None:
                weight = class_weights[batch_y.long()]
                loss = F.binary_cross_entropy_with_logits(
                    outputs, batch_y, reduction="none"
                )
                loss = (loss * weight).mean()
            else:
                loss = F.binary_cross_entropy_with_logits(outputs, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, val_loader, class_weights=None):
        """Validate with multiple metrics"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)

                # Loss calculation
                if self.loss_type == "focal":
                    loss = self.criterion(outputs, batch_y)
                elif self.loss_type == "weighted_bce" and class_weights is not None:
                    weight = class_weights[batch_y.long()]
                    loss = F.binary_cross_entropy_with_logits(
                        outputs, batch_y, reduction="none"
                    )
                    loss = (loss * weight).mean()
                else:
                    loss = F.binary_cross_entropy_with_logits(outputs, batch_y)

                total_loss += loss.item()

                # Predictions
                probs = torch.sigmoid(outputs)
                preds = (probs > self.best_threshold).float()

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        # Calculate F1 score for minority class
        f1 = f1_score(all_targets, all_preds, pos_label=1, zero_division=0)

        return total_loss / len(val_loader), f1

    def train(
        self,
        train_loader,
        val_loader,
        epochs=100,
        lr=0.001,
        weight_decay=1e-4,
        patience=15,
        class_weights=None,
        save_path="models/best_improved_model.pt",
    ):
        """Enhanced training with multiple techniques"""

        # Setup optimizer with different learning rates for different parts
        optimizer = optim.AdamW(
            [
                {"params": self.model.input_norm.parameters(), "lr": lr * 0.1},
                {"params": self.model.layers.parameters(), "lr": lr},
                {"params": self.model.output_layer.parameters(), "lr": lr * 2},
            ],
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
        )

        # Convert class weights to tensor if provided
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(
                self.device
            )

        best_f1 = 0
        patience_counter = 0

        logger.info(f"Training with {self.loss_type} loss on {self.device}")

        for epoch in range(epochs):
            # Update threshold more conservatively and less frequently
            if epoch % 20 == 0 and epoch > 0:
                new_threshold = self.find_optimal_threshold(val_loader)
                # Don't allow dramatic threshold changes
                if abs(new_threshold - self.best_threshold) < 0.3:
                    self.best_threshold = new_threshold

            train_loss = self.train_epoch(train_loader, optimizer, class_weights)
            val_loss, val_f1 = self.validate(val_loader, class_weights)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)

            scheduler.step()

            # Early stopping based on F1 score
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                # Save best model
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "best_f1": best_f1,
                        "best_threshold": self.best_threshold,
                        "epoch": epoch,
                        "model_config": self.model.get_model_info(),
                    },
                    save_path,
                )
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, "
                    f"Threshold: {self.best_threshold:.3f}"
                )

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load best model with weights_only=False for backward compatibility
        checkpoint = torch.load(save_path, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_threshold = checkpoint["best_threshold"]

        return {
            "best_f1": best_f1,
            "best_threshold": self.best_threshold,
            "train_losses": self.train_losses,
            "val_f1_scores": self.val_f1_scores,
        }


def create_balanced_data_loaders(
    X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64, use_sampler=True
):
    """Create data loaders with balanced sampling option"""
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    if use_sampler:
        # Create balanced sampler
        class_counts = np.bincount(y_train.numpy().astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train.numpy().astype(int)]

        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_dataset(file_path):
    """Load dataset (same as v3)"""
    preprocessor = DataPreprocessor()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    state_file = os.path.join(
        root_dir, "preprocessing_artifacts", "preprocessor_state.json"
    )
    preprocessor.load_state(state_file)

    df = pd.read_excel(file_path)
    X = df.drop(["non_responder_in", "temp_index"], axis=1, errors="ignore").values
    y = df["non_responder_in"].values

    # Encode y if binary or categorical target
    if preprocessor.target_type in ["binary", "categorical"]:
        if preprocessor.target_label_encoder is None:
            logger.error("Target LabelEncoder not loaded.")
            raise ValueError("Target LabelEncoder not loaded.")
        y = preprocessor.target_label_encoder.transform(y)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor, y


def main():
    """Main function - THIS WAS MISSING IN YOUR v4!"""
    # Setup paths (same as v3)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessing_artifacts_dir = os.path.join(script_dir, "../preprocessing_artifacts")

    logger.info("\nDebug Info:")
    logger.info(f"Script directory: {script_dir}")
    logger.info(f"Preprocessing artifacts directory: {preprocessing_artifacts_dir}")

    # File paths
    state_file = os.path.join(preprocessing_artifacts_dir, "preprocessor_state.json")
    train_file = os.path.join(preprocessing_artifacts_dir, "specs_train_processed.xlsx")
    val_file = os.path.join(preprocessing_artifacts_dir, "specs_val_processed.xlsx")
    test_file = os.path.join(preprocessing_artifacts_dir, "specs_test_processed.xlsx")

    # Check files exist
    files_to_check = [state_file, train_file, val_file, test_file]
    missing_files = [f for f in files_to_check if not os.path.exists(f)]
    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        return None

    # Load preprocessor state
    with open(state_file, "r") as f:
        preprocessor_state = json.load(f)
    feature_names = preprocessor_state["feature_columns"]

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    logger.info("\n=== LOADING PROCESSED DATA ===")
    X_train, y_train, y_train_raw = load_dataset(train_file)
    X_val, y_val, _ = load_dataset(val_file)
    X_test, y_test, _ = load_dataset(test_file)

    input_dim = X_train.shape[1]
    logger.info(f"Input dimension: {input_dim}")
    logger.info(f"Training samples: {len(X_train)}")

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Method 1: Focal Loss with optimized parameters
    logger.info("\n=== TRAINING WITH FOCAL LOSS ===")
    model_focal = ImprovedPredictor(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],  # Larger network
        dropout_rate=0.3,  # Reduced dropout
        activation="swish",
        use_residual=True,
    )

    trainer_focal = AdvancedModelTrainer(
        model_focal,
        loss_type="focal",
        alpha=0.1,  # Reduced alpha for less aggressive weighting
        gamma=3.0,  # Increased gamma to focus more on hard examples
    )

    train_loader, val_loader, test_loader = create_balanced_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64, use_sampler=True
    )

    results_focal = trainer_focal.train(
        train_loader,
        val_loader,
        epochs=150,
        lr=0.002,
        weight_decay=5e-5,
        patience=25,  # More epochs, lower LR, longer patience
        save_path="models/best_focal_model.pt",
    )

    # Method 2: Weighted BCE
    logger.info("\n=== TRAINING WITH WEIGHTED BCE ===")
    model_weighted = ImprovedPredictor(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.4,
        activation="swish",
        use_residual=True,
    )

    # Compute REASONABLE class weights (your current weights are too extreme)
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train_raw), y=y_train_raw
    )
    # MUCH more conservative weighting
    class_weights[1] *= 1.5  # Only 1.5x boost, not 3x
    class_weights = np.clip(class_weights, 0.1, 10)  # Much more reasonable range
    logger.info(f"Conservative class weights: {class_weights}")

    trainer_weighted = AdvancedModelTrainer(model_weighted, loss_type="weighted_bce")

    results_weighted = trainer_weighted.train(
        train_loader,
        val_loader,
        epochs=100,
        lr=0.003,
        weight_decay=1e-4,
        patience=20,
        class_weights=class_weights,
        save_path="models/best_weighted_model.pt",
    )

    # Compare results
    logger.info(f"\n=== RESULTS COMPARISON ===")
    logger.info(f"Focal Loss - Best F1: {results_focal['best_f1']:.4f}")
    logger.info(f"Weighted BCE - Best F1: {results_weighted['best_f1']:.4f}")

    # Use the better model for evaluation
    if results_focal["best_f1"] > results_weighted["best_f1"]:
        best_model = model_focal
        best_trainer = trainer_focal
        best_results = results_focal
        best_method = "Focal Loss"
    else:
        best_model = model_weighted
        best_trainer = trainer_weighted
        best_results = results_weighted
        best_method = "Weighted BCE"

    logger.info(f"Best method: {best_method}")

    # Evaluate on test set
    logger.info("\n=== EVALUATING BEST MODEL ===")
    evaluator = ModelEvaluator(
        model=best_model, device=best_trainer.device, save_dir="evaluation_results"
    )

    # Set the optimal threshold
    evaluator.threshold = best_trainer.best_threshold

    metrics, predictions, probabilities, targets = evaluator.evaluate(
        test_loader, feature_names=feature_names
    )

    # Save results (convert numpy types to Python types for JSON serialization)
    final_results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "best_method": best_method,
        "model_config": best_model.get_model_info(),
        "training_results": {
            "best_f1": float(best_results["best_f1"]),
            "best_threshold": float(best_results["best_threshold"]),
        },
        "test_metrics": {
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "f1_score": float(metrics["f1"]),
            "roc_auc": float(metrics["roc_auc"]),
            "recall": float(metrics["recall"]),
            "optimal_threshold": float(best_trainer.best_threshold),
        },
        "focal_loss_f1": float(results_focal["best_f1"]),
        "weighted_bce_f1": float(results_weighted["best_f1"]),
    }

    with open("models/improved_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    logger.info("\n🎉 IMPROVED TRAINING COMPLETED!")
    logger.info(f"Best F1 Score: {best_results['best_f1']:.4f}")
    logger.info(f"Optimal Threshold: {best_trainer.best_threshold:.3f}")

    return best_model, best_trainer, final_results


if __name__ == "__main__":
    result = main()
    if result:
        logger.info("\n✨ Pipeline completed successfully!")
    else:
        logger.info("\n❌ Pipeline failed.")

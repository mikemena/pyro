import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.decomposition import PCA
import os


class ModelVisualizer:
    """Handles visualization of model performance metrics and data distributions"""

    def __init__(self, save_dir="plots"):
        """Initialize with directory to save plots"""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_training_history(
        self,
        train_losses,
        val_losses,
        display=True,
        save=True,
        filename="training_history.png",
    ):
        """Plot training and validation loss history"""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title("Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (BCE)")
        plt.legend()
        plt.grid(True)

        if save:
            plt.savefig(os.path.join(self.save_dir, filename))
        if display:
            plt.show()
        plt.close()

    def plot_confusion_matrix(
        self, y_true, y_pred, display=True, save=True, filename="confusion_matrix.png"
    ):
        """Plot confusion matrix as heatmap"""
        cm = confusion_matrix(y_true, (y_pred >= 0.5).astype(int))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        if save:
            plt.savefig(os.path.join(self.save_dir, filename))
        if display:
            plt.show()
        plt.close()

    def plot_data_distribution(
        self,
        data,
        column_names,
        max_cols=3,
        display=True,
        save=True,
        filename="data_distribution.png",
    ):
        """Plot histogram of numerical features to show data distribution with improved readability"""
        df = pd.DataFrame(data, columns=column_names)
        n_features = len(column_names)
        n_rows = (
            n_features + max_cols - 1
        ) // max_cols  # Dynamic rows based on features

        plt.figure(figsize=(15, 5 * n_rows))  # Adjust height based on number of rows
        for idx, column in enumerate(column_names):
            plt.subplot(n_rows, max_cols, idx + 1)
            sns.histplot(df[column], bins=30, kde=True)
            plt.title(f"Distribution of {column}", fontsize=10)
            plt.xlabel(column, rotation=45)
            plt.ylabel("Count", fontsize=8)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

        plt.tight_layout(pad=2.0)  # Increase padding to avoid overlap

        if save:
            plt.savefig(
                os.path.join(self.save_dir, filename), dpi=300, bbox_inches="tight"
            )
        if display:
            plt.show()
        plt.close()

    def plot_prediction_distribution(
        self,
        predictions,
        targets,
        display=True,
        save=True,
        filename="prediction_distribution.png",
    ):
        """Plot distribution of predicted probabilities by class"""
        plt.figure(figsize=(8, 6))
        plt.hist(
            predictions[targets == 0], bins=20, alpha=0.5, label="Class 0", color="blue"
        )
        plt.hist(
            predictions[targets == 1],
            bins=20,
            alpha=0.5,
            label="Class 1",
            color="orange",
        )
        plt.xlabel("Predicted Probabilities")
        plt.ylabel("Count")
        plt.title("Distribution of Predicted Probabilities by Class")
        plt.legend()
        plt.grid(True)

        if save:
            plt.savefig(os.path.join(self.save_dir, filename))
        if display:
            plt.show()
        plt.close()

    def plot_roc_curve(
        self, y_true, y_pred, display=True, save=True, filename="roc_curve.png"
    ):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = np.trapz(tpr, fpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(True)

        if save:
            plt.savefig(os.path.join(self.save_dir, filename))
        if display:
            plt.show()
        plt.close()

    def plot_precision_recall_curve(
        self,
        y_true,
        y_pred,
        display=True,
        save=True,
        filename="precision_recall_curve.png",
    ):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label="Precision-Recall curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(True)

        if save:
            plt.savefig(os.path.join(self.save_dir, filename))
        if display:
            plt.show()
        plt.close()

    def plot_metrics_bar(
        self, metrics, display=True, save=True, filename="metrics_bar.png"
    ):
        """Plot bar chart of various performance metrics"""
        metric_names = ["Accuracy", "F1 Score", "ROC AUC"]
        metric_values = [
            metrics.get("accuracy", 0),
            metrics.get("f1_score", 0),
            metrics.get("roc_auc", 0) if metrics.get("roc_auc") is not None else 0,
        ]

        plt.figure(figsize=(8, 6))
        sns.barplot(x=metric_names, y=metric_values)
        plt.title("Model Performance Metrics")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        for i, v in enumerate(metric_values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")

        if save:
            plt.savefig(os.path.join(self.save_dir, filename))
        if display:
            plt.show()
        plt.close()

    def visualize_original_vs_synthetic_samples():
        original_train = pd.read_excel("debug_splits/raw_train_split.xlsx")
        resampled_train = pd.read_excel(
            "preprocessing_artifacts/loan_train_resampled.xlsx"
        )
        X_original = original_train.drop("personal_loan", axis=1)
        y_original = original_train["personal_loan"]
        X_resampled = resampled_train.drop("personal_loan", axis=1)
        y_resampled = resampled_train["personal_loan"]

        pca = PCA(n_components=2)
        X_original_pca = pca.fit_transform(X_original)
        X_resampled_pca = pca.transform(X_resampled)

        plt.scatter(
            X_original_pca[:, 0],
            X_original_pca[:, 1],
            c=y_original,
            label="Original",
            alpha=0.5,
        )
        plt.scatter(
            X_resampled_pca[:, 0],
            X_resampled_pca[:, 1],
            c=y_resampled,
            label="Resampled",
            alpha=0.2,
        )
        plt.legend()
        plt.title("PCA of Original vs. Resampled Data")
        plt.show()

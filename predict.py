import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessor import DataPreprocessor
from datetime import datetime
from logger import setup_logger

logger = setup_logger(__name__, include_location=True)


class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128], dropout_rate=0.7):
        super(Predictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


def load_dataset(file_path, preprocessor_state_path, is_preprocessed=True):
    preprocessor = DataPreprocessor()
    preprocessor.load_state(preprocessor_state_path)

    df = pd.read_excel(file_path)

    if not is_preprocessed:
        logger.info("Adding temp_index for new dataset...")
        df["temp_index"] = np.arange(len(df))
        logger.info("Preprocessing new dataset...")
        df = preprocessor.process_inference_data(
            df, excel_filename="new_dataset_processed.xlsx"
        )

    # Drop target column if present and exclude temp_index, then convert to tensor
    X = df.drop(
        ["personal_loan", "temp_index"], axis=1, errors="ignore"
    ).values  # Drop 'personal_loan' and 'temp_index' if present
    X_tensor = torch.tensor(X, dtype=torch.float32)

    feature_names = preprocessor.feature_columns
    return X_tensor, feature_names, df  # Return df which includes 'temp_index'


def create_data_loader(X, batch_size=64):
    """Create DataLoaders for predictions"""
    dataset = TensorDataset(X)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def make_predictions(model, data_loader, device):
    """Make predictions with the trained model"""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_x in data_loader:
            batch_x = batch_x[0].to(device)
            logits = model(batch_x)
            probabilities = torch.sigmoid(logits)
            predictions.extend(probabilities.cpu().numpy())
    return np.array(predictions).flatten()


def main():
    # Make paths relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessing_artifacts_dir = os.path.join(script_dir, "preprocessing_artifacts")
    state_file = os.path.join(preprocessing_artifacts_dir, "preprocessor_state.json")
    model_file = os.path.join(script_dir, "training/models/best_final_model.pt")

    # Specify the dataset for predictions
    dataset_file = os.path.join(
        preprocessing_artifacts_dir, "bank_loans_test_processed.xlsx"
    )

    # Check for required files
    for file_path in [state_file, model_file, dataset_file]:
        if not os.path.exists(file_path):
            logger.error(f"{file_path} not found")
            return

    # Determine the original dataset path
    is_preprocessed = True  # Set to True for test dataset, False for new raw dataset
    if is_preprocessed:
        original_dataset_path = os.path.join("debug_splits", "raw_test_split.xlsx")
    else:
        original_dataset_path = dataset_file

    # Load original dataset
    df_original = pd.read_excel(original_dataset_path)

    # For new datasets (not preprocessed), add temp_index since the raw file doesn't have it
    if not is_preprocessed:
        df_original["temp_index"] = np.arange(len(df_original))

    # Load preprocessor state and dataset
    logger.info("LOADING DATASET AND PREPROCESSOR STATE...")
    X, feature_names, input_df = load_dataset(
        dataset_file, state_file, is_preprocessed=is_preprocessed
    )
    data_loader = create_data_loader(X, batch_size=64)

    # Initialize model
    input_dim = X.shape[1]
    model = Predictor(input_dim=input_dim, hidden_dims=[128], dropout_rate=0.7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load trained model weights
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Make predictions
    logger.info("MAKING PREDICTIONS...")
    predictions = make_predictions(model, data_loader, device)
    binary_predictions = (predictions >= 0.5).astype(
        int
    )  # Threshold for binary classification

    # Create predictions DataFrame with temp_index
    pred_df = pd.DataFrame(
        {
            "temp_index": input_df["temp_index"],
            "Probability": predictions,
            "Prediction": binary_predictions,
        }
    )

    # Merge with original data using temp_index and drop temp_index
    output_df = df_original.merge(pred_df, on="temp_index", how="left")
    output_df = output_df.drop("temp_index", axis=1)

    # Optional: Check for row mismatch
    if len(df_original) != len(input_df):
        logger.warning("Row count mismatch between original and processed data.")

    output_file = os.path.join(script_dir, "predictions/predictions.xlsx")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_df.to_excel(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")

    # Save results summary
    results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": dataset_file,
        "model_config": {
            "input_dim": input_dim,
            "hidden_dims": [128],
            "dropout_rate": 0.7,
        },
        "num_predictions": len(predictions),
        "output_file": output_file,
    }
    with open(
        os.path.join(script_dir, "predictions/prediction_results.json"), "w"
    ) as f:
        json.dump(results, f, indent=2)
    logger.info("Prediction results saved.")


if __name__ == "__main__":
    main()
    logger.info("Prediction pipeline completed!")

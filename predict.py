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

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128], dropout_rate=0.7):
        super(Predictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()

def load_dataset(file_path, preprocessor_state_path):
    preprocessor = DataPreprocessor()
    preprocessor.load_state(state_file=preprocessor_state_path)

    df =pd.read_excel(file_path)
    X = df.drop('Status', axis=1).values # Drop 'Status' if present
    X_tensor = torch.tensor(X, dtype=torch.float32)

    feature_names = preprocessor.feature_columns
    return X_tensor, feature_names

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
    preprocessing_artifacts_dir = os.path.join(script_dir, 'preprocessing_artifacts')
    state_file = os.path.join(preprocessing_artifacts_dir, 'preprocessor_state.json')
    model_file = os.path.join(script_dir, 'training/models/best_final_model.pt')

    # Specify the dataset for predictions
    dataset_file = os.path.join(preprocessing_artifacts_dir, 'loan_default_test_processed.xlsx')

    # Check for required files
    for file_path in [state_file, model_file,dataset_file]:
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found")
            return

    # Load preprocessor state and dataset
    print('LOADING DATASET AND PREPROCESSOR STATE...')
    X, feature_names = load_dataset(dataset_file, state_file)
    data_loader = create_data_loader(X, batch_size=64)

    # Initialize model
    input_dim = X.shape[1]
    model = Predictor(input_dim=input_dim, hidden_dims=[128], dropout_rate=0.7)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load trained model weights
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Make predictions
    print('MAKING PREDICTIONS...')
    predictions = make_predictions(model, data_loader, device)
    binary_predictions = (predictions >= 0.5).astype(int) # Threshold for binary classification

    # Save predictions
    output_df = pd.DataFrame({
        'Probability': predictions,
        'Prediction': binary_predictions
    })

    output_file = os.path.join(script_dir, 'predictions/predictions.xlsx')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_df.to_excel(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    # Save results summary
    results = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'dataset': dataset_file,
        'model_config': {
            'input_dim': input_dim,
            'hidden_dims': [128],
            'dropout_rate': 0.7
            },
            'num_predictions': len(predictions),
            'output_file': output_file
        }
    with open(os.path.join(script_dir, 'predictions/prediction_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("Prediction results saved.")

if __name__ == "__main__":
    main()
    print("Prediction pipeline completed!")

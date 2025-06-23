"""
Data preparation script - just handles preprocessing
"""
import torch
from data_pipeline import DataPipeline

def prepare_training_data():
    """Prepare and save training data"""
    print("Starting data preparation...")

    # Initialize pipeline
    pipeline = DataPipeline()

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_training_data(
        'data/student-mat.xlsx',
        target_column='G3',
        test_size=0.2,
        val_size=0.2,
        random_state=42
    )

    # Print shapes
    print(f"\nData preparation complete!")
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # Save processed data (optional - for reuse without reprocessing)
    torch.save({
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }, 'processed_data.pt')

    print("Processed data saved to 'processed_data.pt'")

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    prepare_training_data()

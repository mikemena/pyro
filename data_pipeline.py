"""
High-level data pipeline for training and inference
"""
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from data_analyzer import analyze_dataset
from data_preprocessor import DataPreprocessor

class DataPipeline:
    """Complete data pipeline from raw data to PyTorch tensors"""

    def __init__(self, save_dir='preprocessing_artifacts'):
        self.preprocessor = DataPreprocessor(save_dir)
        self.column_config = None

    def prepare_training_data(self, file_path, target_column=None,
                            test_size=0.2, val_size=0.2, random_state=42):
        """
        Complete pipeline for training data preparation

        Args:
            file_path: Path to training data
            target_column: Name of target column (if supervised)
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed

        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test) or just X splits if unsupervised
        """
        # Analyze data first
        print("Analyzing dataset structure...")
        analysis = analyze_dataset(file_path)

        # Load data
        df = pd.read_excel(file_path)

        # Separate features and target
        if target_column and target_column in df.columns:
            y = df[target_column]
            X = df.drop(target_column, axis=1)
        else:
            y = None
            X = df

        # Auto-detect column types from analysis
        self.column_config = self._extract_column_config(analysis, X.columns)

        # Preprocess
        X_processed = self._preprocess_features(X, fit=True)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_processed.values)
        y_tensor = torch.FloatTensor(y.values) if y is not None else None

        # Split data
        if y is not None:
            splits = self._split_supervised_data(X_tensor, y_tensor, test_size, val_size, random_state)
        else:
            splits = self._split_unsupervised_data(X_tensor, test_size, val_size, random_state)

        # Save preprocessing state
        self.preprocessor.save_state()

        print(f"Training data prepared: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
        return splits

    def prepare_inference_data(self, file_path):
        """
        Prepare new data for inference using saved preprocessing state

        Args:
            file_path: Path to new data

        Returns:
            torch.Tensor: Preprocessed data ready for model
        """
        # Load preprocessing state
        self.preprocessor.load_state()

        # Load and preprocess data
        df = pd.read_excel(file_path)
        X_processed = self._preprocess_features(df, fit=False)

        return torch.FloatTensor(X_processed.values)

    def _extract_column_config(self, analysis, columns):
        """Extract column configuration from analysis results"""
        config = {
            'datetime': [],
            'categorical': [],
            'binary': [],
            'numerical': []
        }

        for col in columns:
            col_type = analysis['columns'][col]['recommended_type']
            if col_type == 'datetime':
                config['datetime'].append(col)
            elif col_type == 'binary':
                config['binary'].append(col)
            elif col_type == 'categorical':
                config['categorical'].append(col)
            else:
                config['numerical'].append(col)

        return config

    def _preprocess_features(self, df, fit=True):
        """Apply all preprocessing steps"""
        # Handle missing values
        df = self.preprocessor.handle_missing_values(
            df,
            self.column_config['numerical'],
            self.column_config['categorical'] + self.column_config['binary']
        )

        # Process datetime columns
        if self.column_config['datetime']:
            df = self.preprocessor.preprocess_datetime(df, self.column_config['datetime'])

        # Process categorical columns
        if self.column_config['categorical'] or self.column_config['binary']:
            df = self.preprocessor.preprocess_categorical(
                df,
                self.column_config['categorical'],
                self.column_config['binary']
            )

        # Align features (important for inference)
        df = self.preprocessor.align_features(df, fit=fit)

        # Scale features
        df = self.preprocessor.scale_features(df, fit=fit)

        return df

    def _split_supervised_data(self, X, y, test_size, val_size, random_state):
        """Split data for supervised learning"""
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _split_unsupervised_data(self, X, test_size, val_size, random_state):
        """Split data for unsupervised learning"""
        # First split: train+val vs test
        X_temp, X_test = train_test_split(
            X, test_size=test_size, random_state=random_state
        )

        # Second split: train vs val
        X_train, X_val = train_test_split(
            X_temp, test_size=val_size/(1-test_size), random_state=random_state
        )

        return X_train, X_val, X_test

# Convenience functions for quick usage
def prepare_training_data(file_path, target_column=None, **kwargs):
    """Quick function to prepare training data"""
    pipeline = DataPipeline()
    return pipeline.prepare_training_data(file_path, target_column, **kwargs)

def prepare_inference_data(file_path):
    """Quick function to prepare inference data"""
    pipeline = DataPipeline()
    return pipeline.prepare_inference_data(file_path)

if __name__ == "__main__":
    # Example usage
    pipeline = DataPipeline()

    # For supervised learning
    splits = pipeline.prepare_training_data('student-mat.xlsx', target_column='G3')
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    # For unsupervised learning
    # splits = pipeline.prepare_training_data('data.xlsx')
    # X_train, X_val, X_test = splits

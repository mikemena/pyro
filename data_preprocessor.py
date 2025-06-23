"""
Core data preprocessing functions
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

class DataPreprocessor:
    """Handles all data preprocessing with state management"""

    def __init__(self, save_dir='preprocessing_artifacts'):
        self.save_dir = save_dir
        self.scaler = None
        self.feature_columns = None
        self.column_mappings = {}

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

    def preprocess_datetime(self, df, date_columns):
        """Extract features from datetime columns"""
        df = df.copy()

        for col in date_columns:
            if col in df.columns:
                # Extract datetime features
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day_of_week'] = df[col].dt.dayofweek
                df[f'{col}_is_weekend'] = (df[col].dt.dayofweek >= 5).astype(int)

                # Cyclical encoding for month
                df[f'{col}_month_sin'] = np.sin(2 * np.pi * df[f'{col}_month'] / 12)
                df[f'{col}_month_cos'] = np.cos(2 * np.pi * df[f'{col}_month'] / 12)

                # Drop original column
                df = df.drop(col, axis=1)

        return df

    def preprocess_categorical(self, df, categorical_columns, binary_columns=None):
        """Handle categorical and binary variables"""
        df = df.copy()

        # Handle binary columns first
        if binary_columns:
            for col in binary_columns:
                if col in df.columns:
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) == 2:
                        # Create mapping
                        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                        df[col] = df[col].map(mapping)
                        self.column_mappings[col] = mapping

        # One-hot encode categorical columns
        categorical_to_encode = [col for col in categorical_columns
                               if col in df.columns and col not in (binary_columns or [])]

        if categorical_to_encode:
            df = pd.get_dummies(df, columns=categorical_to_encode, dtype=int)

        return df

    def handle_missing_values(self, df, numerical_columns, categorical_columns):
        """Fill missing values appropriately"""
        df = df.copy()

        # Numerical columns: fill with median
        for col in numerical_columns:
            if col in df.columns and df[col].isnull().any():
                fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)

        # Categorical columns: fill with mode
        for col in categorical_columns:
            if col in df.columns and df[col].isnull().any():
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(fill_value)

        return df

    def scale_features(self, df, fit=True):
        """Scale features using StandardScaler"""
        if fit:
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(df)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Set fit=True or load saved scaler.")
            scaled_data = self.scaler.transform(df)

        return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

    def align_features(self, df, fit=True):
        """Ensure feature alignment between training and prediction"""
        if fit:
            self.feature_columns = df.columns.tolist()
        else:
            if self.feature_columns is None:
                raise ValueError("Feature columns not set. Set fit=True or load saved state.")

            # Add missing columns
            missing_cols = set(self.feature_columns) - set(df.columns)
            for col in missing_cols:
                df[col] = 0

            # Remove extra columns and reorder
            df = df[self.feature_columns]

        return df

    def save_state(self):
        """Save preprocessing state for later use"""
        state = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'column_mappings': self.column_mappings
        }
        with open(os.path.join(self.save_dir, 'preprocessor_state.pkl'), 'wb') as f:
            pickle.dump(state, f)

    def load_state(self):
        """Load saved preprocessing state"""
        state_file = os.path.join(self.save_dir, 'preprocessor_state.pkl')
        if os.path.exists(state_file):
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
            self.scaler = state['scaler']
            self.feature_columns = state['feature_columns']
            self.column_mappings = state['column_mappings']
        else:
            raise FileNotFoundError(f"No saved state found at {state_file}")

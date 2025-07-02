import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import os

class DataPreprocessor:
    """Handles all data preprocessing with state management and Excel output"""

    def __init__(self, save_dir='preprocessing_artifacts'):
        self.save_dir = save_dir
        self.scaler = None
        self.feature_columns = None
        self.column_mappings = {}
        self.target_encoding_mappings = {}
        self.excel_output_enabled = True
        os.makedirs(save_dir, exist_ok=True)

    def preprocess_datetime(self, df, date_columns):
        """Extract features from datetime columns"""
        df = df.copy()
        for col in date_columns:
            if col in df.columns:
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day_of_week'] = df[col].dt.dayofweek
                df[f'{col}_is_weekend'] = (df[col].dt.dayofweek >= 5).astype(int)
                df[f'{col}_month_sin'] = np.sin(2 * np.pi * df[f'{col}_month'] / 12)
                df[f'{col}_month_cos'] = np.cos(2 * np.pi * df[f'{col}_month'] / 12)
                df = df.drop(col, axis=1)
        return df

    def preprocess_categorical(self, df, categorical_columns, binary_columns=None):
        """Handle categorical and binary variables"""
        df = df.copy()
        if binary_columns:
            for col in binary_columns:
                if col in df.columns:
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) == 2:
                        mapping = {str(unique_vals[0]): 0, str(unique_vals[1]): 1}
                        df[col] = df[col].map(mapping)
                        self.column_mappings[col] = mapping
        categorical_to_encode = [col for col in categorical_columns
                               if col in df.columns and col not in (binary_columns or [])]
        if categorical_to_encode:
            df = pd.get_dummies(df, columns=categorical_to_encode, dtype=int)
        return df

    # Transform a high-cardinality categorical feature into a numerical feature
    def target_encode(self, df, feature_column, target_data=None, alpha=0.1, fit=True):
        print(f"Debugging target_encode for feature: {feature_column} (fit={fit})")
        df = df.copy()

        if fit:
            if target_data is None:
                raise ValueError("Target data must be provided when fit=True")
            if not isinstance(target_data, pd.Series):
                target_data = pd.Series(target_data, index=df.index)
            target_data = pd.to_numeric(target_data, errors='coerce')
            global_mean = target_data.mean(skipna=True)
            temp_df = pd.DataFrame({'feature': df[feature_column].astype(str), 'target': target_data})
            mapping = temp_df.groupby('feature')['target'].apply(
                lambda x: (x.sum(skipna=True) + alpha * global_mean) / (x.count() + alpha)
            ).to_dict()
            # Save the mapping and global mean
            self.target_encoding_mappings[feature_column] = {
                'mapping': mapping,
                'global_mean': float(global_mean)
            }
            df[f"{feature_column}_encoded"] = df[feature_column].astype(str).map(mapping)
            df[f"{feature_column}_encoded"] = df[f"{feature_column}_encoded"].fillna(global_mean)
        else:
            if feature_column not in self.target_encoding_mappings:
                raise ValueError(f"No target encoding mapping found for {feature_column}. Run fit=True first.")
            mapping_info = self.target_encoding_mappings[feature_column]
            mapping = mapping_info['mapping']
            global_mean = mapping_info['global_mean']
            df[f"{feature_column}_encoded"] = df[feature_column].astype(str).map(mapping)
            df[f"{feature_column}_encoded"] = df[f"{feature_column}_encoded"].fillna(global_mean)

        return df.drop(feature_column, axis=1)

    def handle_missing_values(self, df, numerical_columns, categorical_columns):
        """Fill missing values appropriately"""
        df = df.copy()
        for col in numerical_columns:
            if col in df.columns and df[col].isnull().any():
                fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)
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
            missing_cols = set(self.feature_columns) - set(df.columns)
            for col in missing_cols:
                df[col] = 0
            df = df[self.feature_columns]
        return df

    def process_and_save(self, df, target_column=None, excel_filename=None,
                        numerical_columns=None, low_cardinality_categorical_columns=None,
                        high_cardinality_categorical_columns=None,
                        binary_columns=None, datetime_columns=None, fit=True):

        print(f"🔄 Starting preprocessing pipeline (fit={fit})...")

        # Separate target if provided
        if target_column and target_column in df.columns:
            target_data = df[target_column].copy()
            print(f"Target data dtype: {target_data.dtype}")
            features_df = df.drop(target_column, axis=1).copy()
            print(f"   ✓ Separated target column '{target_column}'")
        else:
            target_data = None
            features_df = df.copy()

        # Store original shape for reporting
        original_shape = features_df.shape

        # Step 1: Handle missing values
        if numerical_columns or low_cardinality_categorical_columns:
            print("Handling missing values...")
            all_categorical = (low_cardinality_categorical_columns or []) + (binary_columns or [])
            features_df = self.handle_missing_values(features_df, numerical_columns or [], all_categorical)

        # Step 2: Process datetime columns
        if datetime_columns:
            print(f"Processing {len(datetime_columns)} datetime columns...")
            features_df = self.preprocess_datetime(features_df, datetime_columns)

        # Step 3: Process low-cardinality categorical feature
        if low_cardinality_categorical_columns or binary_columns:
            print(f"Processing low cardinality categorical/binary columns...")
            features_df = self.preprocess_categorical(features_df, low_cardinality_categorical_columns or [], binary_columns)

        # Step 4: Process high-cardinality categorical feature
        if high_cardinality_categorical_columns:
            print(f"Processing high cardinality categorical columns...")
            for col in high_cardinality_categorical_columns:
                if col in features_df.columns:
                        features_df = self.target_encode(
                            features_df,
                            col,
                            target_data=target_data if fit else None,
                            fit=fit
                        )

        # Step 5: Align features (important for inference)
        print("   🔄 Aligning features...")
        features_df = self.align_features(features_df, fit=fit)

        # Step 6: Scale features
        print("   📊 Scaling features...")
        features_df = self.scale_features(features_df, fit=fit)

        print(f"   ✓ Preprocessing complete: {original_shape} → {features_df.shape}")

        # Step 7: Save to Excel if enabled
        if self.excel_output_enabled:
            self._save_to_excel(features_df, target_data, target_column, excel_filename, fit)

        # Step 8: Save preprocessing state if fitting
        if fit:
            print("   💾 Saving preprocessing state...")
            self.save_state()
            print(f"   ✓ State saved to {os.path.join(self.save_dir, 'preprocessor_state.json')}")

        return features_df

    def _save_to_excel(self, features_df, target_data, target_column, excel_filename, fit):
        """Internal method to save processed data to Excel"""
        if excel_filename is None:
            # Auto-generate filename
            suffix = "processed" if fit else "inference_processed"
            excel_filename = f"data_{suffix}.xlsx"

        excel_path = os.path.join(self.save_dir, excel_filename)

        # Combine features and target for Excel output
        if target_data is not None:
            excel_df = features_df.copy()
            excel_df[target_column] = target_data.values
            print(f"   📄 Saving processed data with target to Excel...")
        else:
            excel_df = features_df.copy()
            print(f"   📄 Saving processed features to Excel...")

        # Save to Excel
        excel_df.to_excel(excel_path, index=False)
        print(f"   ✅ Excel saved: {excel_path}")
        print(f"      Shape: {excel_df.shape}")
        print(f"      Columns: {len(excel_df.columns)} ({list(excel_df.columns)[:3]}...)")

        return excel_path

    def enable_excel_output(self, enabled=True):
        """Enable or disable Excel output"""
        self.excel_output_enabled = enabled
        print(f"Excel output {'enabled' if enabled else 'disabled'}")

    def save_state(self):
        """Save preprocessing state as JSON"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Cannot save state.")

        # Extract StandardScaler parameters
        scaler_params = {
            'mean_': self.scaler.mean_.tolist() if self.scaler.mean_ is not None else [],
            'scale_': self.scaler.scale_.tolist() if self.scaler.scale_ is not None else [],
            'var_': self.scaler.var_.tolist() if self.scaler.var_ is not None else [],
            'n_samples_seen_': int(self.scaler.n_samples_seen_) if self.scaler.n_samples_seen_ is not None else 0
        }

        # Prepare state dictionary
        state = {
            'scaler_params': scaler_params,
            'feature_columns': self.feature_columns if self.feature_columns is not None else [],
            'column_mappings': self.column_mappings,
            'target_encoding_mappings': self.target_encoding_mappings
        }

        # Save to JSON
        state_file = os.path.join(self.save_dir, 'preprocessor_state.json')
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        """Load preprocessing state from JSON"""
        state_file = os.path.join(self.save_dir, 'preprocessor_state.json')
        if not os.path.exists(state_file):
            raise FileNotFoundError(f"No saved state found at {state_file}")

        # Load JSON
        with open(state_file, 'r') as f:
            state = json.load(f)

        # Reconstruct StandardScaler
        self.scaler = StandardScaler()
        scaler_params = state['scaler_params']
        if scaler_params['mean_']:
            self.scaler.mean_ = np.array(scaler_params['mean_'])
            self.scaler.scale_ = np.array(scaler_params['scale_'])
            self.scaler.var_ = np.array(scaler_params['var_'])
            self.scaler.n_samples_seen_ = scaler_params['n_samples_seen_']

        # Load other state
        self.feature_columns = state['feature_columns']
        self.column_mappings = state['column_mappings']

        # Load target encoding mappings
        self.target_encoding_mappings = state.get('target_encoding_mappings', {})

        print(f"✅ Loaded preprocessing state from {state_file}")
        print(f"   Features: {len(self.feature_columns)}")
        print(f"   Column mappings: {len(self.column_mappings)}")
        print(f"   Target encoding mappings: {len(self.target_encoding_mappings)}")


    def process_training_data(self, df, target_column, column_config, excel_filename=None):
        """
        Convenience method for processing training data

        Args:
            df: Raw training DataFrame
            target_column: Name of target column
            column_config: Dictionary with column type classifications
            excel_filename: Custom Excel filename

        Returns:
            pandas.DataFrame: Processed features ready for training
        """
        print("🎯 Processing training data...")

        return self.process_and_save(
            df=df,
            target_column=target_column,
            excel_filename=excel_filename or "training_processed.xlsx",
            numerical_columns=column_config.get('numerical', []),
            low_cardinality_categorical_columns=column_config.get('low_cardinality_categorical', []),
            high_cardinality_categorical_columns=column_config.get('high_cardinality_categorical', []),
            binary_columns=column_config.get('binary', []),
            datetime_columns=column_config.get('datetime', []),
            fit=True
        )

    def process_inference_data(self, df, excel_filename=None):
        """
        Convenience method for processing inference data

        Args:
            df: Raw inference DataFrame
            excel_filename: Custom Excel filename

        Returns:
            pandas.DataFrame: Processed features ready for prediction
        """
        print("🔮 Processing inference data...")

        # Load existing state first
        self.load_state()

        return self.process_and_save(
            df=df,
            target_column=None,
            excel_filename=excel_filename or "inference_processed.xlsx",
            fit=False
        )

# # Example usage demonstrating the clean integration
# if __name__ == "__main__":
#     print("🧪 DataPreprocessor Excel Output Demo")
#     print("-" * 50)

#     # Create sample data
#     sample_data = pd.DataFrame({
#         'school': ['GP', 'MS', 'GP'],
#         'sex': ['F', 'M', 'F'],
#         'age': [17, 18, 16],
#         'Mjob': ['teacher', 'health', 'other'],
#         'studytime': [2, 1, 3],
#         'G3': [15, 12, 18]
#     })

#     print("Sample raw data:")
#     print(sample_data)

#     # Initialize preprocessor
#     preprocessor = DataPreprocessor(save_dir='demo_artifacts')

#     # Define column configuration
#     column_config = {
#         'numerical': ['age', 'studytime'],
#         'categorical': ['Mjob'],
#         'binary': ['school', 'sex']
#     }

#     # Process training data (generates Excel + artifacts)
#     print("\n1. Processing training data...")
#     processed_features = preprocessor.process_training_data(
#         df=sample_data,
#         target_column='G3',
#         column_config=column_config,
#         excel_filename='demo_training.xlsx'
#     )

#     print("\nProcessed features:")
#     print(processed_features.head())

#     # Process inference data (uses artifacts, generates Excel)
#     print("\n2. Processing inference data...")
#     new_student = pd.DataFrame({
#         'school': ['GP'],
#         'sex': ['M'],
#         'age': [17],
#         'Mjob': ['services'],
#         'studytime': [2]
#     })

#     processed_inference = preprocessor.process_inference_data(
#         df=new_student,
#         excel_filename='demo_inference.xlsx'
#     )

#     print("\nProcessed inference data:")
#     print(processed_inference)

#     print("\nCheck demo_artifacts/ for generated files.")

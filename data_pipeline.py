import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import os
from data_analyzer import analyze_dataset
from data_preprocessor import DataPreprocessor

class DataPipeline:
    def __init__(self, save_dir='preprocessing_artifacts'):
        self.preprocessor = DataPreprocessor(save_dir)
        self.column_config = None
        self.save_dir = save_dir

    def prepare_training_data_with_splits(self, file_path, target_column,test_size=0.2, val_size=0.2, random_state=42):
        """
        Complete pipeline that generates separate Excel files for train/val/test

        """
        print("=" * 70)
        print("SPLIT EXCEL DATA PIPELINE - MAXIMUM TRANSPARENCY")
        print("=" * 70)
        print(f" target column parameter from prepare_training_data_with_splits, {target_column} ")

        # Step 1: Analyze dataset
        print("\n1. ANALYZING DATASET STRUCTURE...")
        analysis = analyze_dataset(file_path)
        print(f"   ✓ Raw data shape: {analysis['shape']}")

        # Step 2: Load and validate data
        print("\n2. LOADING RAW DATA...")
        df = pd.read_excel(file_path)
        print(f"   ✓ Loaded: {df.shape}")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        # Step 3: Configure preprocessing
        print("\n3. CONFIGURING PREPROCESSING...")
        X = df.drop(target_column, axis=1)

        self.column_config = self._extract_column_config(analysis, X.columns)

        # Step 4: Create raw data splits FIRST (before preprocessing)
        print("\n4. CREATING RAW DATA SPLITS...")
        train_df, val_df, test_df = self._split_raw_dataframe(df, test_size, val_size, random_state)

        # Step 5: Process each split and save to Excel
        print("\n5. PROCESSING AND SAVING SPLITS...")
        base_filename = os.path.splitext(os.path.basename(file_path))[0]

        # Process training data (fit=True)
        print("   🔄 Processing training split...")
        train_excel = f"{base_filename}_train_processed.xlsx"
        X_train_processed = self.preprocessor.process_training_data(
            df=train_df,
            target_column=target_column,
            column_config=self.column_config,
            excel_filename=train_excel
        )
        y_train = train_df[target_column].values

        # Process validation data (fit=False)
        print("   🔄 Processing validation split...")
        val_excel = f"{base_filename}_val_processed.xlsx"
        X_val_processed = self._process_split_with_target(
            val_df, target_column, val_excel, fit=False
        )
        y_val = val_df[target_column].values

        # Process test data (fit=False)
        print("   🔄 Processing test split...")
        test_excel = f"{base_filename}_test_processed.xlsx"
        X_test_processed = self._process_split_with_target(
            test_df, target_column, test_excel, fit=False
        )
        y_test = test_df[target_column].values

        # Step 6: Convert to tensors
        print("\n6. CONVERTING TO TENSORS...")
        X_train = torch.FloatTensor(X_train_processed.values)
        X_val = torch.FloatTensor(X_val_processed.values)
        X_test = torch.FloatTensor(X_test_processed.values)
        y_train = torch.FloatTensor(y_train)
        y_val = torch.FloatTensor(y_val)
        y_test = torch.FloatTensor(y_test)

        print(f"   ✓ Training tensors: {X_train.shape}")
        print(f"   ✓ Validation tensors: {X_val.shape}")
        print(f"   ✓ Test tensors: {X_test.shape}")

        # Step 7: Summary
        print("\n7. PIPELINE SUMMARY")
        print("   📁 Generated Files:")
        print(f"      📄 {train_excel}")
        print(f"      📄 {val_excel}")
        print(f"      📄 {test_excel}")

        return X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df

    def load_split_data_for_training(self, base_filename):
        """
        Load pre-split Excel files for training

        Args:
            base_filename: Base name (e.g., 'student-mat')

        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("=" * 70)
        print("LOADING PRE-SPLIT EXCEL FILES FOR TRAINING")
        print("=" * 70)

        # Define file paths
        train_file = os.path.join(self.save_dir, f"{base_filename}_train_processed.xlsx")
        val_file = os.path.join(self.save_dir, f"{base_filename}_val_processed.xlsx")
        test_file = os.path.join(self.save_dir, f"{base_filename}_test_processed.xlsx")

        # Check all files exist
        files_to_check = [train_file, val_file, test_file]
        missing_files = [f for f in files_to_check if not os.path.exists(f)]

        if missing_files:
            print("❌ ERROR: Missing split files!")
            for file in missing_files:
                print(f"   Missing: {file}")
            print("\nRun the split pipeline first to generate these files.")
            raise FileNotFoundError("Split Excel files not found")

        print("✅ All split files found!")

        # Load each split
        def load_split(file_path, split_name):
            print(f"   📄 Loading {split_name}: {os.path.basename(file_path)}")
            df = pd.read_excel(file_path)
            y = df['G3'].values  # Assuming G3 is target
            X = df.drop('G3', axis=1).values
            print(f"      Shape: {X.shape}")
            return torch.FloatTensor(X), torch.FloatTensor(y)

        X_train, y_train = load_split(train_file, "Training")
        X_val, y_val = load_split(val_file, "Validation")
        X_test, y_test = load_split(test_file, "Test")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _split_raw_dataframe(self, df, test_size, val_size, random_state):
        """Split raw DataFrame into train/val/test before preprocessing"""
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=None
        )

        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size/(1-test_size), random_state=random_state
        )

        return train_df, val_df, test_df

    def _process_split_with_target(self, df, target_column, excel_filename, fit=False):
        """Process a data split and save with target included"""
        # Temporarily disable auto Excel output
        self.preprocessor.enable_excel_output(False)

        # Process features only
        X = df.drop(target_column, axis=1)
        if self.column_config is None:
            raise ValueError("No column_config provided")
        X_processed = self.preprocessor.process_and_save(
            df=X,
            target_column=None,
            excel_filename=None,
            numerical_columns=self.column_config.get('numerical', []),
            low_cardinality_categorical_columns=self.column_config.get('low_cardinality_categorical', []),
            high_cardinality_categorical_columns=self.column_config.get('low_cardinality_categorical_columns',[]),
            binary_columns=self.column_config.get('binary', []),
            datetime_columns=self.column_config.get('datetime', []),
            fit=fit
        )

        # Re-enable Excel output and save manually with target
        self.preprocessor.enable_excel_output(True)

        # Create full DataFrame with target for Excel
        full_df = X_processed.copy()
        full_df[target_column] = df[target_column].values

        # Save to Excel
        excel_path = os.path.join(self.save_dir, excel_filename)
        full_df.to_excel(excel_path, index=False)
        print(f"      ✅ Saved: {excel_filename} ({full_df.shape})")

        return X_processed

    def _extract_column_config(self, analysis, columns):
        """Extract column configuration from analysis results"""
        config = {
            'datetime': [],
            'binary': [],
            'numerical': [],
            'low_cardinality_categorical': [],
            'high_cardinality_categorical': [],
            'text': []
        }

        for col in columns:
            col_type = analysis['columns'][col]['recommended_type']
            if col_type == 'datetime':
                config['datetime'].append(col)
            elif col_type == 'binary':
                config['binary'].append(col)
            elif col_type == 'low_cardinality_categorical':
                config['low_cardinality_categorical'].append(col)
            elif col_type == 'high_cardinality_categorical':
                config['high_cardinality_categorical'].append(col)
            elif col_type == 'numerical':
                config['numerical'].append(col)
            else:
                config['text'].append(col)

        print("Config:", config)
        return config

# Convenience functions
def prepare_split_training_data(file_path, target_column='G3', **kwargs):
    """Prepare training data with separate Excel files for each split"""
    pipeline = DataPipeline()
    return pipeline.prepare_training_data_with_splits(file_path, target_column, **kwargs)

def load_split_training_data(base_filename='student-mat'):
    """Load pre-split Excel files for training"""
    pipeline = DataPipeline()
    return pipeline.load_split_data_for_training(base_filename)

if __name__ == "__main__":

    print("\n" + "="*70)
    print("DEMO: SPLIT EXCEL PIPELINE")
    print("="*70)

    try:
        # Generate split Excel files
        pipeline = DataPipeline()
        splits = pipeline.prepare_training_data_with_splits(
            'data/student-mat.xlsx',
            target_column='G3',
            test_size=0.2,
            val_size=0.2,
            random_state=42
        )

        # Later, load the split files
        print("\n" + "="*70)
        print("DEMO: LOADING SPLIT FILES")
        print("="*70)

        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.load_split_data_for_training('student-mat')

        print("\n🎉 Split Excel pipeline demonstration complete!")

    except FileNotFoundError: print("The pipeline is ready to use with your data!")

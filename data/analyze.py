# import pandas as pd

# def analyze_file(file_path):
#     # Read the Excel file
#     df = pd.read_excel(file_path)

#     # Get dataset overview
#     print("Dataset Overview:")
#     print(f"Total rows: {len(df)}, Total columns: {len(df.columns)}")

#     # Get sheet names (for completeness, though we use first sheet)
#     xl = pd.ExcelFile(file_path)
#     print(f"\nSheet names: {xl.sheet_names}")

#     # Column analysis
#     print("\nColumn Analysis:")
#     for index, column in enumerate(df.columns, 1):
#         # Get column data
#         series = df[column]

#         # Get unique values and sample
#         unique_values = series.unique()
#         sample_values = unique_values[:8]  # Get up to 8 samples

#         # Print column info
#         print(f"\n{index}. {column}:")
#         print(f"   Sample values: {', '.join(map(str, sample_values))}")
#         print(f"   Unique count: {len(unique_values)}")
#         print(f"   Missing values: {series.isna().sum()}")

#         # Determine data type
#         dtype = series.dtype
#         unique_count = len(unique_values)

#         # Check if column contains mixed types
#         is_numeric = pd.api.types.is_numeric_dtype(series)
#         is_string = pd.api.types.is_string_dtype(series)

#         if unique_count <= 10 and not is_numeric:
#             print(f"   → CATEGORICAL ({unique_count} categories)")
#         elif is_numeric and not is_string:
#             print("   → NUMERICAL")
#         else:
#             print("   → MIXED (needs investigation)")

# # Execute analysis
# if __name__ == "__main__":
#     file_path = 'student-mat.xlsx'
#     analyze_file(file_path)


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

def analyze_and_preprocess(file_path, is_training=True, output_dir='preprocessing_artifacts'):
    # Create output directory for saving preprocessing artifacts
    if is_training and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the Excel file
    df = pd.read_excel(file_path)

    # Dataset overview
    print("Dataset Overview:")
    print(f"Total rows: {len(df)}, Total columns: {len(df.columns)}")

    # Get sheet names
    xl = pd.ExcelFile(file_path)
    print(f"\nSheet names: {xl.sheet_names}")

    # Column analysis and preprocessing
    print("\nColumn Analysis and Preprocessing:")
    processed_df = df.copy()
    column_info = {}  # Store column types and metadata

    for index, column in enumerate(df.columns, 1):
        series = df[column]
        unique_values = series.unique()
        sample_values = unique_values[:8]
        unique_count = len(unique_values)
        missing_count = series.isna().sum()

        print(f"\n{index}. {column}:")
        print(f"   Sample values: {', '.join(map(str, sample_values))}")
        print(f"   Unique count: {unique_count}")
        print(f"   Missing values: {missing_count}")

        # Determine column type
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_string = pd.api.types.is_string_dtype(series)
        is_binary = unique_count == 2 and not is_numeric

        # Handle missing values
        if missing_count > 0:
            if is_numeric and not is_binary:
                fill_value = series.median()
                processed_df[column] = series.fillna(fill_value)
                print(f"   Filled {missing_count} missing numerical values with median: {fill_value}")
            else:
                fill_value = series.mode()[0]
                processed_df[column] = series.fillna(fill_value)
                print(f"   Filled {missing_count} missing categorical/binary values with mode: {fill_value}")

        # Store column info
        column_info[column] = {'type': 'numerical' if is_numeric and not is_binary else 'categorical',
                              'is_binary': is_binary}

        # Convert binary Yes/No to 1/0
        if is_binary and is_string:
            unique_vals = series.dropna().unique()
            mapping = {unique_vals[0]: 1, unique_vals[1]: 0}
            processed_df[column] = processed_df[column].map(mapping)
            column_info[column]['binary_mapping'] = mapping
            print(f"   Converted binary {unique_vals} to 1/0")

        # Mark categorical columns for one-hot encoding
        if not is_numeric and not is_binary and unique_count <= 10:
            print(f"   → CATEGORICAL ({unique_count} categories, will be one-hot encoded)")
        elif is_numeric and not is_binary:
            print("   → NUMERICAL")
        else:
            print("   → MIXED (treated as categorical for one-hot encoding)")

    # One-hot encode categorical variables
    categorical_cols = [col for col, info in column_info.items()
                       if info['type'] == 'categorical' and not info.get('is_binary')]
    if categorical_cols:
        processed_df = pd.get_dummies(processed_df, columns=categorical_cols, prefix=categorical_cols)
        print(f"\nOne-hot encoded categorical columns: {categorical_cols}")

    # Feature alignment
    if is_training:
        # Save feature columns
        feature_columns = processed_df.columns.tolist()
        with open(os.path.join(output_dir, 'feature_columns.pkl'), 'wb') as f:
            pickle.dump(feature_columns, f)
    else:
        # Load training feature columns
        with open(os.path.join(output_dir, 'feature_columns.pkl'), 'rb') as f:
            feature_columns = pickle.load(f)

        # Align features
        missing_cols = set(feature_columns) - set(processed_df.columns)
        for col in missing_cols:
            processed_df[col] = 0  # Add missing columns with zeros
            print(f"   Added missing column from training: {col}")

        extra_cols = set(processed_df.columns) - set(feature_columns)
        if extra_cols:
            processed_df = processed_df.drop(columns=extra_cols)
            print(f"   Removed extra columns not in training: {extra_cols}")

        # Ensure same column order
        processed_df = processed_df[feature_columns]

    # Scaling
    scaler = StandardScaler()
    if is_training:
        scaled_data = scaler.fit_transform(processed_df)
        # Save scaler
        with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
    else:
        # Load saved scaler
        with open(os.path.join(output_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        scaled_data = scaler.transform(processed_df)

    processed_df = pd.DataFrame(scaled_data, columns=processed_df.columns)
    print("\nApplied StandardScaler to all features")

    # Save column info for prediction
    if is_training:
        with open(os.path.join(output_dir, 'column_info.pkl'), 'wb') as f:
            pickle.dump(column_info, f)

    # Save processed DataFrame to a new Excel file in the same directory
    input_dir = os.path.dirname(file_path)
    output_file_name = os.path.splitext(os.path.basename(file_path))[0] + '_processed.xlsx'
    output_file_path = os.path.join(input_dir, output_file_name)
    processed_df.to_excel(output_file_path, index=False)
    print(f"\nSaved processed data to: {output_file_path}")

    return processed_df

if __name__ == "__main__":
    file_path = 'student-mat.xlsx'
    processed_data = analyze_and_preprocess(file_path, is_training=True)
    print("\nProcessed Data Preview:")
    print(processed_data.head())

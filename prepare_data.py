"""
Data preparation script - handles preprocessing AND saves raw splits for debugging
"""
import torch
import os
from data_pipeline import DataPipeline

def prepare_training_data():
    """Prepare and save training data with both processed and raw splits"""
    print("Starting data preparation...")

    # Initialize pipeline
    pipeline = DataPipeline()

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df = pipeline.prepare_training_data_with_splits(
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

    # Save processed tensors (for model training)
    torch.save({
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }, 'processed_data.pt')
    print("✅ Processed tensors saved to 'processed_data.pt'")

    # Save raw dataframes for debugging and human inspection
    print("\n🔍 Saving raw dataframes for debugging...")

    # Create debugging directory if it doesn't exist
    debug_dir = 'debug_splits'
    os.makedirs(debug_dir, exist_ok=True)

    # Save raw splits (before any preprocessing)
    train_df.to_excel(os.path.join(debug_dir, 'raw_train_split.xlsx'), index=False)
    val_df.to_excel(os.path.join(debug_dir, 'raw_val_split.xlsx'), index=False)
    test_df.to_excel(os.path.join(debug_dir, 'raw_test_split.xlsx'), index=False)

    print(f"✅ Raw splits saved to '{debug_dir}/' directory:")
    print(f"   📄 raw_train_split.xlsx ({train_df.shape})")
    print(f"   📄 raw_val_split.xlsx ({val_df.shape})")
    print(f"   📄 raw_test_split.xlsx ({test_df.shape})")

    # Create a summary file for quick debugging
    create_split_summary(train_df, val_df, test_df, debug_dir)

    return X_train, X_val, X_test, y_train, y_val, y_test

def create_split_summary(train_df, val_df, test_df, debug_dir):
    """Create a summary Excel file for quick split analysis"""

    # Calculate summary statistics for each split
    summary_data = []

    for split_name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        summary_data.append({
            'Split': split_name,
            'Count': len(df),
            'Percentage': f"{len(df)/len(train_df.index.union(val_df.index).union(test_df.index))*100:.1f}%",
            'G3_Mean': f"{df['G3'].mean():.2f}",
            'G3_Std': f"{df['G3'].std():.2f}",
            'G3_Min': df['G3'].min(),
            'G3_Max': df['G3'].max(),
            'Missing_Values': df.isnull().sum().sum()
        })

    # Convert to DataFrame and save
    import pandas as pd
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(debug_dir, 'split_summary.xlsx')
    summary_df.to_excel(summary_path, index=False)

    print(f"✅ Split summary saved to '{summary_path}'")
    print("\n📊 Quick Summary:")
    for _, row in summary_df.iterrows():
        print(f"   {row['Split']}: {row['Count']} samples ({row['Percentage']}), G3 avg: {row['G3_Mean']}")

def debug_splits():
    """
    Function to help debug splitting issues after training
    Call this if you notice weird model behavior
    """
    print("🔍 DEBUGGING SPLIT INTEGRITY...")

    debug_dir = 'debug_splits'

    try:
        import pandas as pd

        # Load raw splits
        train_df = pd.read_excel(os.path.join(debug_dir, 'raw_train_split.xlsx'))
        val_df = pd.read_excel(os.path.join(debug_dir, 'raw_val_split.xlsx'))
        test_df = pd.read_excel(os.path.join(debug_dir, 'raw_test_split.xlsx'))

        print("✅ Raw split files loaded successfully")

        # Check for data leakage (overlapping rows)
        train_indices = set(train_df.index)
        val_indices = set(val_df.index)
        test_indices = set(test_df.index)

        overlaps = []
        if train_indices & val_indices:
            overlaps.append("Train-Val overlap detected!")
        if train_indices & test_indices:
            overlaps.append("Train-Test overlap detected!")
        if val_indices & test_indices:
            overlaps.append("Val-Test overlap detected!")

        if overlaps:
            print("❌ DATA LEAKAGE DETECTED:")
            for overlap in overlaps:
                print(f"   {overlap}")
        else:
            print("✅ No data leakage detected")

        # Check target distribution
        print("\n📊 Target Distribution Check:")
        print(f"   Train G3 range: {train_df['G3'].min()}-{train_df['G3'].max()}")
        print(f"   Val G3 range: {val_df['G3'].min()}-{val_df['G3'].max()}")
        print(f"   Test G3 range: {test_df['G3'].min()}-{test_df['G3'].max()}")

        # Check for missing critical grades in any split
        critical_grades = [0, 20]  # Assuming 0-20 scale
        for grade in critical_grades:
            train_has = (train_df['G3'] == grade).any()
            val_has = (val_df['G3'] == grade).any()
            test_has = (test_df['G3'] == grade).any()
            print(f"   Grade {grade}: Train={train_has}, Val={val_has}, Test={test_has}")

    except FileNotFoundError:
        print("❌ Debug files not found. Run prepare_training_data() first.")
    except Exception as e:
        print(f"❌ Error during debugging: {e}")

if __name__ == "__main__":
    prepare_training_data()

    # Optional: Run debugging check
    print("\n" + "="*50)
    debug_splits()

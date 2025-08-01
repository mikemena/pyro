import torch
import os
from data_pipeline_v2 import DataPipeline
import numpy as np
import pandas as pd
from hashlib import sha1
from logger import setup_logger

logger = setup_logger(__name__,include_location=True)


def prepare_training_data():
    """Prepare and save training data with both processed and raw splits"""
    logger.info("Starting data preparation...")

    pipeline = DataPipeline()
    X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df = (
        pipeline.prepare_training_data_with_splits(
            "data/bank_loans.xlsx",
            target_column="personal_loan",
            test_size=0.2,
            val_size=0.2,
            random_state=42,
            imbalance_threshold=0.3,
        )
    )
    logger.info(f"With SMOTE - Training class distribution: {pd.Series(y_train.numpy()).value_counts(normalize=True)}")
    logger.info(f"\nData preparation complete!")
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    logger.info(f"Test set: {X_test.shape}")

    torch.save(
        {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        },
        "processed_data.pt",
    )
    logger.info("✅ Processed tensors saved to 'processed_data.pt'")
    logger.info("\n🔍 Saving raw dataframes for debugging...")
    debug_dir = "debug_splits"
    os.makedirs(debug_dir, exist_ok=True)

    # Add temp_index to validation and test raw splits for consistent merging in predictions
    val_df["temp_index"] = np.arange(len(val_df))
    test_df["temp_index"] = np.arange(len(test_df))

    train_df.to_excel(os.path.join(debug_dir, "raw_train_split.xlsx"), index=False)
    val_df.to_excel(os.path.join(debug_dir, "raw_val_split.xlsx"), index=False)
    test_df.to_excel(os.path.join(debug_dir, "raw_test_split.xlsx"), index=False)

    logger.info(f"✅ Raw splits saved to '{debug_dir}/' directory:")
    logger.info(f"📄 raw_train_split.xlsx ({train_df.shape})")
    logger.info(f"📄 raw_val_split.xlsx ({val_df.shape})")
    logger.info(f"📄 raw_test_split.xlsx ({test_df.shape})")

    create_split_summary(
        train_df, val_df, test_df, debug_dir, target_column="personal_loan"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_split_summary(train_df, val_df, test_df, debug_dir, target_column):
    """Create a summary Excel file for quick split analysis"""
    # Calculate total rows for percentage calculation
    total_rows = len(train_df) + len(val_df) + len(test_df)

    summary_data = []
    for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        summary = {
            "Split": split_name,
            "Count": len(df),
            "Percentage": f"{len(df)/total_rows*100:.1f}%",
            "Missing_Values": df.isnull().sum().sum(),
        }
        if pd.api.types.is_numeric_dtype(df[target_column]):
            summary.update(
                {
                    "Target_Mean": f"{df[target_column].mean():.2f}",
                    "Target_Std": f"{df[target_column].std():.2f}",
                    "Target_Min": df[target_column].min(),
                    "Target_Max": df[target_column].max(),
                }
            )
        else:
            value_counts = df[target_column].value_counts(normalize=True)
            summary.update(
                {
                    "Target_Distribution": ", ".join(
                        [f"{k}: {v:.2%}" for k, v in value_counts.items()]
                    )
                }
            )
        summary_data.append(summary)

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(debug_dir, "split_summary.xlsx")
    summary_df.to_excel(summary_path, index=False)

    logger.info(f"✅ Split summary saved to '{summary_path}'")
    logger.info("\n📊 Quick Summary:")
    for _, row in summary_df.iterrows():
        logger.info(f"{row['Split']}: {row['Count']} samples ({row['Percentage']})")
        if "Target_Mean" in row:
            logger.info(f"Target avg: {row['Target_Mean']}, std: {row['Target_Std']}")
        else:
            logger.info(f"Target distribution: {row['Target_Distribution']}")


def debug_splits(target_column="personal_loan"):
    """Function to help debug splitting issues after training"""
    logger.info("🔍 DEBUGGING SPLIT INTEGRITY...")
    debug_dir = "debug_splits"

    try:
        train_df = pd.read_excel(os.path.join(debug_dir, "raw_train_split.xlsx"))
        val_df = pd.read_excel(os.path.join(debug_dir, "raw_val_split.xlsx"))
        test_df = pd.read_excel(os.path.join(debug_dir, "raw_test_split.xlsx"))

        logger.info("✅ Raw split files loaded successfully")

        # Use hashes as unique identifiers to prevent data leakage detection
        def hash_rows(df):
            return set(
                df.astype(str).apply(
                    lambda row: sha1("||".join(row).encode()).hexdigest(), axis=1
                )
            )

        train_hashes = hash_rows(train_df)
        val_hashes = hash_rows(val_df)
        test_hashes = hash_rows(test_df)

        overlaps = []
        if train_hashes & val_hashes:
            overlaps.append(
                f"Train-Val overlap detected! ({len(train_hashes & val_hashes)} rows)"
            )
        if train_hashes & test_hashes:
            overlaps.append(
                f"Train-Test overlap detected! ({len(train_hashes & test_hashes)} rows)"
            )
        if val_hashes & test_hashes:
            overlaps.append(
                f"Val-Test overlap detected! ({len(val_hashes & test_hashes)} rows)"
            )

        if overlaps:
            logger.info("❌ DATA LEAKAGE DETECTED:")
            for msg in overlaps:
                logger.info(f"{msg}")
        else:
            logger.info("✅ No data leakage detected")

        logger.info("\n📊 Target Distribution Check:")
        if pd.api.types.is_numeric_dtype(train_df[target_column]):
            logger.info(f"Train {target_column} range: {train_df[target_column].min()}-{train_df[target_column].max()}")
            logger.info(f"Val {target_column} range: {val_df[target_column].min()}-{val_df[target_column].max()}")
            logger.info(f"Test {target_column} range: {test_df[target_column].min()}-{test_df[target_column].max()}")

        else:
            for split_name, df in [
                ("Train", train_df),
                ("Val", val_df),
                ("Test", test_df),
            ]:
                value_counts = df[target_column].value_counts(normalize=True)
                logger.info(f"{split_name} {target_column} distribution: {', '.join([f'{k}: {v:.2%}' for k, v in value_counts.items()])}")

    except FileNotFoundError:
        logger.error("❌ Debug files not found. Run prepare_training_data() first.")
    except Exception as e:
        logger.error(f"❌ Error during debugging: {e}")

if __name__ == "__main__":
    prepare_training_data()
    logger.info("\n" + "=" * 50)
    debug_splits()

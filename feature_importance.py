import pandas as pd
import os

try:
    import xgboost as xgb

    USE_XGB = True
except ImportError:
    print("XGBoost not available, using LightGBM instead")
    import lightgbm as lgb

    USE_XGB = False

import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings("ignore")


class FeatureImportanceAnalyzer:
    """
    A class for analyzing feature importance in imbalanced datasets using SHAP values
    and XGBoost. Helps identify noisy or irrelevant features.
    """

    def __init__(self, threshold: float = 0.01, random_state: int = 42):
        """
        Initialize the analyzer.

        Args:
            threshold: SHAP value threshold below which features are considered low importance
            random_state: Random state for reproducibility
        """
        self.threshold = threshold
        self.random_state = random_state
        self.model = None
        self.explainer = None
        self.feature_importance = None
        self.low_importance_features = None

    def prepare_data(self, df: pd.DataFrame, features: List[str], target: str) -> Tuple:
        """
        Prepare data for analysis including encoding and splitting.

        Args:
            df: Input dataframe
            features: List of feature column names
            target: Target column name

        Returns:
            Tuple of (X, y)
        """
        # Check if target exists
        if target not in df.columns:
            raise ValueError(
                f"Target column '{target}' not found in dataset. Available columns: {list(df.columns)}"
            )

        # Check if features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            features = [f for f in features if f in df.columns]
            print(f"Using {len(features)} available features")

        # Create a copy to avoid modifying original data
        df_copy = df.copy()

        # Encode target if it's categorical (yes/no, strings, etc.)
        if (
            df_copy[target].dtype == "object"
            or df_copy[target].dtype.name == "category"
        ):
            le = LabelEncoder()
            df_copy[target] = le.fit_transform(df_copy[target])
            print(f"Target variable encoded. Original classes: {le.classes_}")
            print(f"Encoded as: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        # Separate features and target
        X = df_copy[features].copy()
        y = df_copy[target].copy()

        # Check for missing values in features
        missing_in_features = X.isnull().sum()
        if missing_in_features.any():
            print("Warning: Missing values found in features:")
            for feature, count in missing_in_features[missing_in_features > 0].items():
                print(f"  {feature}: {count} missing values")
            # Simple imputation - you might want more sophisticated methods
            X = X.fillna(X.median(numeric_only=True))
            print("Missing values filled with median for numeric columns")

        # Check class distribution
        class_dist = y.value_counts().sort_index()
        print(f"\nClass distribution:\n{class_dist}")
        if len(class_dist) == 2:
            print(f"Imbalance ratio: {class_dist.min() / class_dist.max():.3f}")

        print(f"\nFinal dataset shape: {X.shape}")
        print(f"Features: {len(features)} columns")
        print(f"Target: '{target}' with {len(y.unique())} unique values")

        return X, y

    def train_model(
        self, X: pd.DataFrame, y: pd.Series, use_existing_split: bool = False
    ) -> Dict:
        """
        Train XGBoost model with proper handling of class imbalance.

        Args:
            X: Feature matrix
            y: Target vector
            use_existing_split: If True, assumes data is already split for training

        Returns:
            Dictionary with training results and data splits
        """
        if not use_existing_split:
            # Split into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
        else:
            # Use all data for training (assuming it's already the training set)
            X_train, X_test, y_train, y_test = X, X, y, y
            print("Using provided data as training set (no train-test split)")

        # Calculate class weights for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"Scale pos weight for imbalance: {scale_pos_weight:.3f}")

        # Train model with enhanced parameters for feature selection
        if USE_XGB:
            self.model = xgb.XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="auc",
            )
        else:
            # Use LightGBM as alternative
            pos_weight = scale_pos_weight
            self.model = lgb.LGBMClassifier(
                class_weight={0: 1, 1: pos_weight},
                random_state=self.random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary",
                metric="auc",
                verbosity=-1,
            )

        self.model.fit(X_train, y_train)

        # Model performance evaluation
        if not use_existing_split:
            train_score = roc_auc_score(
                y_train, self.model.predict_proba(X_train)[:, 1]
            )
            test_score = roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])
            print(f"\nModel Performance:")
            print(f"Training AUC: {train_score:.4f}")
            print(f"Testing AUC: {test_score:.4f}")

            # Cross-validation for more robust estimate
            cv_scores = cross_val_score(
                self.model, X_train, y_train, cv=5, scoring="roc_auc"
            )
            print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "model": self.model,
        }

    def analyze_feature_importance(
        self, X_test: pd.DataFrame, feature_names: List[str] = None
    ) -> pd.DataFrame:
        """
        Analyze feature importance using SHAP values.

        Args:
            X_test: Test set for SHAP analysis
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance rankings
        """
        if self.model is None:
            raise ValueError("Model must be trained first. Call train_model().")

        if feature_names is None:
            feature_names = X_test.columns.tolist()

        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

        # Compute SHAP values
        print("Computing SHAP values...")
        shap_values = self.explainer.shap_values(X_test)

        # Calculate mean absolute SHAP values for each feature
        shap_values_abs_mean = np.abs(shap_values).mean(axis=0)

        # Create feature importance dataframe
        self.feature_importance = pd.DataFrame(
            {
                "Feature": feature_names,
                "Mean_Abs_SHAP": shap_values_abs_mean,
                "XGB_Importance": self.model.feature_importances_,
            }
        ).sort_values(by="Mean_Abs_SHAP", ascending=False)

        # Add ranking
        self.feature_importance["SHAP_Rank"] = range(
            1, len(self.feature_importance) + 1
        )

        return self.feature_importance, shap_values

    def identify_low_importance_features(self) -> List[str]:
        """
        Identify features with low importance based on threshold.

        Returns:
            List of feature names with low importance
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance must be analyzed first.")

        self.low_importance_features = self.feature_importance[
            self.feature_importance["Mean_Abs_SHAP"] < self.threshold
        ]["Feature"].tolist()

        return self.low_importance_features

    def create_visualizations(
        self, shap_values, X_test: pd.DataFrame, save_plots: bool = False
    ):
        """
        Create SHAP visualizations for feature importance.

        Args:
            shap_values: SHAP values from analysis
            X_test: Test data
            save_plots: Whether to save plots to files
        """
        # Summary plot (bar)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        if save_plots:
            plt.savefig("shap_importance_bar.png", bbox_inches="tight", dpi=300)
        plt.show()

        # Summary plot (beeswarm)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        if save_plots:
            plt.savefig("shap_summary_beeswarm.png", bbox_inches="tight", dpi=300)
        plt.show()

        # Feature importance comparison
        plt.figure(figsize=(12, 8))
        comparison_df = self.feature_importance.head(20)  # Top 20 features

        x = np.arange(len(comparison_df))
        width = 0.35

        plt.subplot(1, 1, 1)
        plt.barh(
            x, comparison_df["Mean_Abs_SHAP"], width, label="SHAP Importance", alpha=0.8
        )
        plt.barh(
            x + width,
            comparison_df["XGB_Importance"],
            width,
            label="XGB Importance",
            alpha=0.8,
        )

        plt.ylabel("Features")
        plt.title("Feature Importance: SHAP vs XGBoost")
        plt.yticks(x + width / 2, comparison_df["Feature"])
        plt.legend()
        plt.tight_layout()

        if save_plots:
            plt.savefig("importance_comparison.png", bbox_inches="tight", dpi=300)
        plt.show()

    def generate_report(self) -> str:
        """
        Generate a comprehensive report of the analysis.

        Returns:
            String containing the analysis report
        """
        if self.feature_importance is None:
            raise ValueError("Analysis must be completed first.")

        report = []
        report.append("=" * 60)
        report.append("FEATURE IMPORTANCE ANALYSIS REPORT")
        report.append("=" * 60)

        report.append(f"\nTotal features analyzed: {len(self.feature_importance)}")
        report.append(f"Threshold for low importance: {self.threshold}")
        report.append(f"Features below threshold: {len(self.low_importance_features)}")

        report.append("\nTOP 10 MOST IMPORTANT FEATURES:")
        report.append("-" * 40)
        top_features = self.feature_importance.head(10)
        for idx, row in top_features.iterrows():
            report.append(
                f"{row['SHAP_Rank']:2d}. {row['Feature']:30s} "
                f"SHAP: {row['Mean_Abs_SHAP']:.4f}"
            )

        if self.low_importance_features:
            report.append(f"\nLOW IMPORTANCE FEATURES (candidates for removal):")
            report.append("-" * 40)
            for feature in self.low_importance_features:
                shap_val = self.feature_importance[
                    self.feature_importance["Feature"] == feature
                ]["Mean_Abs_SHAP"].iloc[0]
                report.append(f"  - {feature:30s} SHAP: {shap_val:.6f}")

        report.append(f"\nRECOMMENDATIONS:")
        report.append("-" * 40)
        if len(self.low_importance_features) > 0:
            report.append(
                f"• Consider removing {len(self.low_importance_features)} low-importance features"
            )
            report.append("• Test model performance with and without these features")
            report.append("• Consider domain expertise when making final decisions")
        else:
            report.append(
                "• All features appear to have meaningful importance above threshold"
            )
            report.append(
                "• Consider lowering threshold or examining bottom-ranked features"
            )

        return "\n".join(report)


def main():
    """
    Main function demonstrating usage of the FeatureImportanceAnalyzer.
    Modify this section based on your specific needs.
    """
    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer(threshold=0.01, random_state=42)

    # Load your data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(
        script_dir, "preprocessing_artifacts", "specs_test_processed.xlsx"
    )
    df = pd.read_excel(f"{data}")

    # Define features and target
    features = [
        "cred_type",
        "group_sb_count",
        "status_AP",
        "status_DR",
        "status_PD",
        "status_TM",
        "send_for_credentialing_in_0",
        "send_for_credentialing_in_1",
        "credentialing_override_in_0",
        "credentialing_required_cd_EVERY_3_YRS",
        "credentialing_required_cd_EVERY_3_YRS_CONT_ENTITY",
        "credentialing_required_cd_NPPPCP_OR_EVERY_3_YRS_CE",
        "credentialing_required_cd_No",
        "credentialed_cd_No",
        "credentialed_cd_Not Required",
        "credentialed_cd_Yes",
        "credentialing_delegated_in_0",
        "credentialing_delegated_in_1",
        "gender_F",
        "gender_M",
        "prac_category_PRO",
        "prac_category_SI",
        "medicaid_info_onfile_N",
        "medicaid_info_onfile_Y",
        "degree_encoded",
        "specialty_encoded",
        "county_encoded",
        "temp_index",
    ]
    target = "non_responder_in"

    # Prepare data
    X, y = analyzer.prepare_data(df, features, target)

    # Train model (set use_existing_split=True if your data is already split)
    train_results = analyzer.train_model(X, y, use_existing_split=False)

    # Analyze feature importance
    feature_importance, shap_values = analyzer.analyze_feature_importance(
        train_results["X_test"], features
    )

    # Identify low importance features
    low_importance = analyzer.identify_low_importance_features()

    # Create visualizations
    analyzer.create_visualizations(
        shap_values, train_results["X_test"], save_plots=True
    )

    # Generate and print report
    report = analyzer.generate_report()
    print(report)

    # Save results
    feature_importance.to_csv("feature_importance_analysis.csv", index=False)

    return analyzer, feature_importance, low_importance


if __name__ == "__main__":
    analyzer, importance_df, low_imp_features = main()

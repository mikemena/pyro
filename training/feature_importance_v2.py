import pandas as pd
import os
import sys
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import setup_logger

logger = setup_logger(__name__, include_location=True)


class FeatureImportanceAnalyzer:
    def __init__(
        self,
        threshold: float = 0.01,
        random_state: int = 42,
        save_dir="evaluation_results",
    ):
        self.threshold = threshold
        self.random_state = random_state
        self.model = None
        self.explainer = None
        self.feature_importance = None
        self.low_importance_features = None
        self.label_encoder = None
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def prepare_data(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, features, target
    ):

        # Check if target column is in both DataFrames
        for df, name in [(train_df, "train"), (test_df, "test")]:
            if target not in df.columns:
                logger.error(
                    f"Target column '{target}' not found in {name} dataset. Available columns: {list(df.columns)}"
                )
                raise ValueError(
                    f"Target column '{target}' not found in {name} dataset."
                )

        # Keep features present in train and test; logs warnings for missing ones.
        filtered_features = [
            f for f in features if f in train_df.columns and f in test_df.columns
        ]
        if len(filtered_features) < len(features):
            missing_features = [f for f in features if f not in filtered_features]
            logger.warning(
                f"Missing features in train or test dataset: {missing_features}"
            )
            logger.info(f"Using {len(filtered_features)} available features")

        # Create copies
        train_df_copy = train_df.copy()
        test_df_copy = test_df.copy()

        # Encode target if categorical
        if (
            train_df_copy[target].dtype == "object"
            or train_df_copy[target].dtype.name == "category"
        ):
            self.label_encoder = LabelEncoder()
            train_df_copy[target] = self.label_encoder.fit_transform(
                train_df_copy[target]
            )
            logger.info(
                f"Target variable encoded. Original classes: {self.label_encoder.classes_}"
            )
            logger.info(
                f"Encoded as: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}"
            )
            try:
                test_df_copy[target] = self.label_encoder.transform(
                    test_df_copy[target]
                )
            except ValueError as e:
                logger.error(f"Test set contains unseen target labels: {e}")
                raise ValueError(f"Test set contains unseen target labels: {e}")
        else:
            logger.info("Target variable is numeric, no encoding needed.")

        # Separate features and target
        X_train = train_df_copy[filtered_features].copy()
        y_train = train_df_copy[target].copy()
        X_test = test_df_copy[filtered_features].copy()
        y_test = test_df_copy[target].copy()

        # Check for missing values
        for X, name in [(X_train, "train"), (X_test, "test")]:
            missing_in_features = X.isnull().sum()
            if missing_in_features.any():
                logger.warning(f"Missing values found in {name} features:")
                for feature, count in missing_in_features[
                    missing_in_features > 0
                ].items():
                    logger.info(f"{feature}: {count} missing values")
                X.fillna(X_train.median(numeric_only=True), inplace=True)
                logger.info(
                    f"Missing values in {name} filled with median from training data"
                )

        # Log class distribution
        class_dist = y_train.value_counts().sort_index()
        logger.info(f"\nTraining class distribution:\n{class_dist}")
        if len(class_dist) == 2:
            logger.info(f"Imbalance ratio: {class_dist.min() / class_dist.max():.3f}")

        logger.info(f"\nTraining dataset shape: {X_train.shape}")
        logger.info(f"Test dataset shape: {X_test.shape}")
        logger.info(f"Features: {len(filtered_features)} columns")
        logger.info(
            f"Target: '{target}' with {len(y_train.unique())} unique values in train"
        )

        return X_train, y_train, X_test, y_test, filtered_features

    def train_model(self, X: pd.DataFrame, y: pd.Series, use_existing_split=True):
        if not use_existing_split:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
            logger.info("Using provided data as training set (no train-test split)")

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        logger.info(f"Scale pos weight for imbalance: {scale_pos_weight:.3f}")

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
        self.model.fit(X_train, y_train)

        train_score = roc_auc_score(y_train, self.model.predict_proba(X_train)[:, 1])
        test_score = roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])
        logger.info("\nModel Performance:")
        logger.info(f"Training AUC: {train_score:.4f}")
        logger.info(f"Testing AUC: {test_score:.4f}")

        cv_scores = cross_val_score(
            self.model, X_train, y_train, cv=5, scoring="roc_auc"
        )
        logger.info(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "model": self.model,
        }

    def analyze_feature_importance(self, X_test: pd.DataFrame, feature_names=None):
        if self.model is None:
            logger.error("Model must be trained first. Call train_model().")
            raise ValueError("Model must be trained first. Call train_model().")
        if feature_names is None:
            feature_names = X_test.columns.tolist()
        self.explainer = shap.TreeExplainer(self.model)
        logger.info("Computing SHAP values...")
        shap_values = self.explainer.shap_values(X_test)
        shap_values_abs_mean = np.abs(shap_values).mean(axis=0)
        self.feature_importance = pd.DataFrame(
            {
                "Feature": feature_names,
                "Mean_Abs_SHAP": shap_values_abs_mean,
                "XGB_Importance": self.model.feature_importances_,
            }
        ).sort_values(by="Mean_Abs_SHAP", ascending=False)
        self.feature_importance["SHAP_Rank"] = range(
            1, len(self.feature_importance) + 1
        )
        return self.feature_importance, shap_values

    def identify_low_importance_features(self):
        if self.feature_importance is None:
            logger.error("Feature importance must be analyzed first.")
            raise ValueError("Feature importance must be analyzed first.")
        self.low_importance_features = self.feature_importance[
            self.feature_importance["Mean_Abs_SHAP"] < self.threshold
        ]["Feature"].tolist()
        return self.low_importance_features

    def create_visualizations(
        self, shap_values, X_test: pd.DataFrame, save_plots=False
    ):
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        if save_plots:
            plt.savefig(
                os.path.join(self.save_dir, "shap_importance_bar.png"),
                bbox_inches="tight",
                dpi=300,
            )
        plt.show()

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        if save_plots:
            plt.savefig(
                os.path.join(self.save_dir, "shap_summary_beeswarm.png"),
                bbox_inches="tight",
                dpi=300,
            )
        plt.show()

        plt.figure(figsize=(12, 8))
        comparison_df = self.feature_importance.head(20)
        x = np.arange(len(comparison_df))
        width = 0.35
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
            plt.savefig(
                os.path.join(self.save_dir, "importance_comparison.png"),
                bbox_inches="tight",
                dpi=300,
            )
        plt.show()

    def generate_report(self):
        if self.feature_importance is None:
            logger.error("Analysis must be completed first.")
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
                f"{row['SHAP_Rank']:2d}. {row['Feature']:30s} SHAP: {row['Mean_Abs_SHAP']:.4f}"
            )
        if self.low_importance_features:
            report.append("\nLOW IMPORTANCE FEATURES (candidates for removal):")
            report.append("-" * 40)
            for feature in self.low_importance_features:
                shap_val = self.feature_importance[
                    self.feature_importance["Feature"] == feature
                ]["Mean_Abs_SHAP"].iloc[0]
                report.append(f" - {feature:30s} SHAP: {shap_val:.6f}")
        report.append("\nRECOMMENDATIONS:")
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
    analyzer = FeatureImportanceAnalyzer(threshold=0.01, random_state=42)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_data = os.path.join(
        script_dir, "preprocessing_artifacts", "specs_train_processed.xlsx"
    )
    test_data = os.path.join(
        script_dir, "preprocessing_artifacts", "specs_test_processed.xlsx"
    )
    train_df = pd.read_excel(train_data)
    test_df = pd.read_excel(test_data)
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
    # Prepare data, get filtered features
    X_train, y_train, X_test, y_test, filtered_features = analyzer.prepare_data(
        train_df, test_df, features, target
    )
    # Train model
    train_results = analyzer.train_model(X_train, y_train, use_existing_split=True)
    # Analyze feature importance
    feature_importance, shap_values = analyzer.analyze_feature_importance(
        X_test, filtered_features
    )
    # Identify low importance features
    low_importance = analyzer.identify_low_importance_features()
    # Create visualizations
    analyzer.create_visualizations(shap_values, X_test, save_plots=True)
    # Generate and print report
    report = analyzer.generate_report()
    logger.info(f"{report}")
    # Save results
    feature_importance.to_csv("feature_importance_analysis.csv", index=False)
    return analyzer, feature_importance, low_importance


if __name__ == "__main__":
    analyzer, importance_df, low_imp_features = main()

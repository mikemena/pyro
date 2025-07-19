
-- apply SMOTE to the training data --

[] add imblearn to requirements.txt
[] data_pipeline.py -> do the following changes to class DataPipeline:
   add def _detect_class_imbalance,
   add def _apply_smote
[] visualize.py -> add def visualize_original_vs_synthetic_samples
    - is it correct to compare raw train split with train resampled?

---Learn this stuff---
[] How can i engineer new features to improve performance?


-- Ways to validate if SMOTE is working

[] Visualize Synthetic Samples
   Why: Visualizing the data can reveal whether synthetic samples are reasonable (e.g., they lie in plausible regions of the feature space).
   How: Use dimensionality reduction (e.g., PCA or t-SNE) to visualize original vs. synthetic samples in 2D.

[] Check for Overfitting
    Why: SMOTE can sometimes generate synthetic samples that are too similar, leading to overfitting on the training set.
    How:
    Compare training vs. validation performance. If the model performs significantly better on the training set than the validation set after SMOTE, it may indicate overfitting.

    Use cross-validation to assess generalization:

    `from sklearn.model_selection import cross_val_score
    clf = RandomForestClassifier(random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1_macro")
    print(f"Cross-validation F1 scores: {scores.mean():.2f} ± {scores.std():.2f}")`

    If cross-validation scores are low or highly variable, consider tuning SMOTE parameters (e.g., reduce k_neighbors to generate more diverse samples).

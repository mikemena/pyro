# Accuracy

- Measures the proportion of correct predictions (TP + TN) / (TP + TN + FP + FN)

- accuracy can be misleading in imbalanced datasets because it’s dominated by the majority class

# Precision

- Precision = TP / (TP + FP). It measures how many predicted positives are actually positive.

# Recall

- Recall = TP / (TP + FN). It measures how many actual positives are correctly identified

# F1-Score

- F1 = 2 * (Precision * Recall) / (Precision + Recall). It balances precision and recall

# F2-Score

- F2-score weights recall higher than precision, useful when false negatives are more costly
- A high F2-score reflects the strong recall, suggesting the model is effective at minimizing missed positives.

# Matthews Correlation Coefficient (MCC)

- MCC ranges from -1 to 1 and accounts for all four confusion matrix quadrants (TP, TN, FP, FN), making it robust for imbalanced datasets

# ROC AUC

- Measures the model’s ability to distinguish between classes across all thresholds
- This is one of the best indicators of performance in imbalanced datasets.

# Average Precision

- Summarizes the precision-recall curve, focusing on the positive class
- A high value confirms strong performance on the minority class

# Log Loss

- Measures the uncertainty of predictions. Lower is better

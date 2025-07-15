---Changing loss function from regression to categorical---

[X] In train model -> evaluate function modify the with statement:
    logits = self.model(batch_x)
    probabilities = torch.sigmoid(logits)  # Convert logits to probabilities
    predictions.extend(probabilities.cpu().numpy())

[X] In train model -> evaluate function: import accuracy_score, f1_score,       roc_auc_score, confusion_matrix from sklearn
   and update the evaluate function to include those.

[X] In train model -> train function: switch to nn.BCEWithLogitsLoss() instead
   of nn.MSELoss() for a Yes/No target column.

[] In the main() -> update results object test_metrics section

[X] In train model -> Pull out the graph/chart [matplotlib.pyplot] and create a seperate py file that i can import and use. include:

- seaborn.heatmap
- Numerical data histogram to show the distribution  balance of data and ranges
- training history
-predictions

---Learn this stuff---
[] How can i engineer new features to improve performance?

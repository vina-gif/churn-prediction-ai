import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

# 1. Load the test data (never seen by the models during training)
X_test = pd.read_csv("Data Folder/X_test.csv")
y_test = pd.read_csv("Data Folder/y_test.csv").squeeze()

# 2. Load the trained models saved in Step 3
model_files = {
    "Logistic Regression": "Data Folder/logistic_regression.pkl",
    "Random Forest": "Data Folder/random_forest.pkl",
    "Gradient Boosting": "Data Folder/gradient_boosting.pkl",
}

results = []
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 3. Evaluate each model on the test data
for ax, (name, path) in zip(axes, model_files.items()):
    model = joblib.load(path)

    y_pred = model.predict(X_test)                # predicted class (0 or 1)
    y_prob = model.predict_proba(X_test)[:, 1]    # predicted probability of churn

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1 Score": round(f1_score(y_test, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
    })

    # Draw the confusion matrix for this model
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(ax=ax, colorbar=False)
    ax.set_title(name)

# 4. Build the comparison table and sort by ROC-AUC (best model first)
results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
print("\nModel Comparison:\n")
print(results_df.to_string(index=False))

# 5. Save the table and the confusion matrix chart (for report and GitHub README)
results_df.to_csv("Data Folder/model_comparison.csv", index=False)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
print("\nSaved: model_comparison.csv and confusion_matrices.png")

# 6. Print the best model
best = results_df.iloc[0]["Model"]
print(f"Best model (by ROC-AUC): {best}")
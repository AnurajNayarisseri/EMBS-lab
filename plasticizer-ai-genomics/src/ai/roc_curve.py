"""
roc_curve.py
Generate ROC curve and AUC plot for plasticizer hydrolase prediction model.

Input:
    roc_data.csv with columns:
        true  -> true class labels (0/1)
        prob  -> predicted probabilities

Output:
    results/figures/roc_curve.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def main():

    input_file = "roc_data.csv"
    output_dir = "results/figures"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading ROC data...")
    df = pd.read_csv(input_file)

    y_true = df["true"].values
    y_scores = df["prob"].values

    print("Computing ROC curve...")
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    print(f"ROC AUC: {roc_auc:.4f}")

    # Plot
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Plasticizer Hydrolase Prediction")
    plt.legend(loc="lower right")

    output_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"ROC curve saved to: {output_path}")


if __name__ == "__main__":
    main()

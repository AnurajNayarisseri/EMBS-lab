"""
feature_importance.py
Extract and plot feature importance from the trained Random Forest model.

Inputs:
  - rf_plasticizer_model.joblib (saved by train_model.py)

Outputs:
  - results/tables/feature_importance_all.csv
  - results/tables/feature_importance_top30.csv
  - results/figures/feature_importance_top15.png

This script assumes the feature order used in train_model.py:
  1) length (1)
  2) amino acid composition (20) in AA order: ACDEFGHIKLMNPQRSTVWY
  3) motif features (1): gxsxg
  4) domain binary vector (len(domain_vocab))
"""

import os
import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt


AA = "ACDEFGHIKLMNPQRSTVWY"


def build_feature_names(domain_vocab):
    names = []
    # 1) Length
    names.append("length")

    # 2) Amino acid composition
    for a in AA:
        names.append(f"aac_{a}")

    # 3) Motif features (as in train_model.py)
    names.append("motif_gxsxg")

    # 4) Pfam domain binary features
    for d in domain_vocab:
        names.append(f"pfam_{d}")

    return names


def main():
    parser = argparse.ArgumentParser(description="Extract and plot Random Forest feature importances.")
    parser.add_argument("--model", default="rf_plasticizer_model.joblib", help="Path to saved RF model bundle.")
    parser.add_argument("--outdir_tables", default="results/tables", help="Output folder for CSV tables.")
    parser.add_argument("--outdir_figures", default="results/figures", help="Output folder for figures.")
    parser.add_argument("--top_csv", type=int, default=30, help="Top N features to save in top CSV.")
    parser.add_argument("--top_plot", type=int, default=15, help="Top N features to plot.")
    args = parser.parse_args()

    os.makedirs(args.outdir_tables, exist_ok=True)
    os.makedirs(args.outdir_figures, exist_ok=True)

    # Load model bundle
    bundle = joblib.load(args.model)
    model = bundle["model"]
    domain_vocab = bundle["domain_vocab"]

    if not hasattr(model, "feature_importances_"):
        raise ValueError("Loaded model does not have feature_importances_. Expected RandomForestClassifier.")

    importances = model.feature_importances_
    feature_names = build_feature_names(domain_vocab)

    if len(importances) != len(feature_names):
        raise ValueError(
            f"Feature count mismatch: model has {len(importances)} importances, "
            f"but constructed {len(feature_names)} feature names. "
            "Check that your feature engineering order matches train_model.py."
        )

    # Save full table
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    out_all = os.path.join(args.outdir_tables, "feature_importance_all.csv")
    df.to_csv(out_all, index=False)
    print(f"Saved: {out_all}")

    # Save top N table
    df_top = df.head(args.top_csv).copy()
    out_top = os.path.join(args.outdir_tables, f"feature_importance_top{args.top_csv}.csv")
    df_top.to_csv(out_top, index=False)
    print(f"Saved: {out_top}")

    # Plot top N (horizontal bar plot)
    df_plot = df.head(args.top_plot).copy()
    df_plot = df_plot.iloc[::-1]  # reverse for nicer barh order

    plt.figure(figsize=(10, 7))
    plt.barh(df_plot["feature"], df_plot["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {args.top_plot} Predictive Features (Random Forest)")
    plt.tight_layout()

    out_fig = os.path.join(args.outdir_figures, f"feature_importance_top{args.top_plot}.png")
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_fig}")


if __name__ == "__main__":
    main()

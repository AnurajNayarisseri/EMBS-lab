"""
score_distribution.py
Plot prediction score distributions for plasticizer hydrolase classification.

This script supports TWO common inputs:
1) Candidate ranking output from score_candidates.py
   - CSV with columns: protein_id, plasticizer_score
   - Default file: KSSKSLAB04_plasticizer_ranked.csv

2) ROC evaluation data (optional)
   - CSV with columns: true, prob
   - Default file: roc_data.csv

Outputs:
- results/figures/score_distribution_candidates.png   (from candidate scores)
- results/figures/score_distribution_by_class.png     (from ROC data, if provided)
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_candidate_scores(df: pd.DataFrame, outpath: str, bins: int = 40) -> None:
    if "plasticizer_score" not in df.columns:
        raise ValueError("Candidate CSV must contain a 'plasticizer_score' column.")

    scores = df["plasticizer_score"].astype(float).values

    plt.figure(figsize=(7, 5))
    plt.hist(scores, bins=bins)
    plt.xlabel("Predicted plasticizer hydrolase score")
    plt.ylabel("Count")
    plt.title("Score Distribution (Candidate Proteins)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_scores_by_class(df: pd.DataFrame, outpath: str, bins: int = 40) -> None:
    required = {"true", "prob"}
    if not required.issubset(df.columns):
        raise ValueError("ROC CSV must contain columns: 'true' and 'prob'.")

    df = df.copy()
    df["true"] = df["true"].astype(int)
    df["prob"] = df["prob"].astype(float)

    pos = df.loc[df["true"] == 1, "prob"].values
    neg = df.loc[df["true"] == 0, "prob"].values

    plt.figure(figsize=(7, 5))
    plt.hist(neg, bins=bins, alpha=0.6, label="Negative (non-plasticizer)")
    plt.hist(pos, bins=bins, alpha=0.6, label="Positive (plasticizer)")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Score Distribution by Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot score distribution figures for candidate predictions and/or ROC evaluation."
    )
    parser.add_argument(
        "--candidate_csv",
        default="KSSKSLAB04_plasticizer_ranked.csv",
        help="CSV from score_candidates.py with 'plasticizer_score' column.",
    )
    parser.add_argument(
        "--roc_csv",
        default=None,
        help="Optional ROC CSV with columns 'true' and 'prob' (e.g., roc_data.csv).",
    )
    parser.add_argument(
        "--outdir",
        default="results/figures",
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of histogram bins.",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Candidate score distribution
    if os.path.exists(args.candidate_csv):
        cand_df = pd.read_csv(args.candidate_csv)
        out1 = os.path.join(args.outdir, "score_distribution_candidates.png")
        plot_candidate_scores(cand_df, out1, bins=args.bins)
        print(f"Saved candidate score distribution: {out1}")
    else:
        print(f"[WARN] Candidate CSV not found: {args.candidate_csv} (skipping)")

    # 2) Score distribution by class (if roc csv provided)
    if args.roc_csv:
        if os.path.exists(args.roc_csv):
            roc_df = pd.read_csv(args.roc_csv)
            out2 = os.path.join(args.outdir, "score_distribution_by_class.png")
            plot_scores_by_class(roc_df, out2, bins=args.bins)
            print(f"Saved class-wise score distribution: {out2}")
        else:
            print(f"[WARN] ROC CSV not found: {args.roc_csv} (skipping)")


if __name__ == "__main__":
    main()

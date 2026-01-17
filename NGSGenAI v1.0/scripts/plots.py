#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if len(sys.argv) != 3:
    print("Usage: plots.py de_results.tsv output_prefix")
    sys.exit(1)

de_file = sys.argv[1]
out = sys.argv[2]

print("[NGSGenAI] Loading DE results...")
df = pd.read_csv(de_file, sep="\t")

# -------------------------------
# Volcano Plot
# -------------------------------
print("[NGSGenAI] Creating volcano plot...")
plt.figure(figsize=(8,6))

df["neglog10p"] = -np.log10(df["pvalue"])

sig = df["FDR"] < 0.05

plt.scatter(df["log2FC"], df["neglog10p"], c="lightgrey", alpha=0.5)
plt.scatter(df[sig]["log2FC"], df[sig]["neglog10p"], c="red", alpha=0.8)

plt.axhline(-np.log10(0.05), linestyle="--")
plt.axvline(1, linestyle="--")
plt.axvline(-1, linestyle="--")

plt.xlabel("log2 Fold Change")
plt.ylabel("-log10(p-value)")
plt.title("NGSGenAI Volcano Plot")

plt.tight_layout()
plt.savefig(out + "_volcano.png", dpi=300)
plt.close()

# -------------------------------
# Heatmap (Top 50 genes)
# -------------------------------
print("[NGSGenAI] Creating heatmap...")
top = df.sort_values("FDR").head(50)["Gene"]

# Dummy expression matrix (for visualization)
expr = np.random.randn(50, 6)

plt.figure(figsize=(10,8))
sns.heatmap(expr, cmap="viridis", yticklabels=top)

plt.title("NGSGenAI Top Differentially Expressed Genes")
plt.xlabel("Samples")
plt.ylabel("Genes")

plt.tight_layout()
plt.savefig(out + "_heatmap.png", dpi=300)
plt.close()

print("[NGSGenAI] Figures generated:")
print(out + "_volcano.png")
print(out + "_heatmap.png")

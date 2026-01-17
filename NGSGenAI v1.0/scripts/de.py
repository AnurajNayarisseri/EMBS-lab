#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

if len(sys.argv) != 3:
    print("Usage: de.py counts.tsv de_results.tsv")
    sys.exit(1)

counts_file = sys.argv[1]
out_file = sys.argv[2]

# -----------------------------
# Load featureCounts table
# -----------------------------
df = pd.read_csv(counts_file, sep="\t", comment="#")

# Remove annotation columns
counts = df.iloc[:, 6:]

genes = df.iloc[:, 0]

# Split samples into two groups
# (first half = control, second half = treatment)
n = counts.shape[1]
half = n // 2

group1 = counts.iloc[:, :half]
group2 = counts.iloc[:, half:]

# -----------------------------
# Normalize (CPM)
# -----------------------------
cpm = counts.div(counts.sum(axis=0), axis=1) * 1e6

g1 = cpm.iloc[:, :half]
g2 = cpm.iloc[:, half:]

# -----------------------------
# Compute log2 fold change
# -----------------------------
log2fc = np.log2((g2.mean(axis=1) + 1) / (g1.mean(axis=1) + 1))

# -----------------------------
# Compute p-values
# -----------------------------
pvals = []
for i in range(len(g1)):
    try:
        p = ttest_ind(g1.iloc[i], g2.iloc[i], equal_var=False).pvalue
    except:
        p = 1.0
    pvals.append(p)

pvals = np.array(pvals)

# -----------------------------
# Multiple testing correction
# -----------------------------
fdr = multipletests(pvals, method="fdr_bh")[1]

# -----------------------------
# Output
# -----------------------------
out = pd.DataFrame({
    "Gene": genes,
    "log2FC": log2fc,
    "pvalue": pvals,
    "FDR": fdr
})

out = out.sort_values("pvalue")

out.to_csv(out_file, sep="\t", index=False)

print(f"[OK] Differential expression written to {out_file}")

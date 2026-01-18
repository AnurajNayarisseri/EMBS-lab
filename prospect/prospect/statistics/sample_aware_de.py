"""
sample_aware_de.py
------------------

Sample-aware differential expression for PROSPECT.

This module performs differential expression analysis using
biological samples (not cells) as the unit of replication,
thereby avoiding pseudoreplication and inflated significance.

Key principles:
- Samples are independent units
- Cells are nested within samples
- Conservative statistical testing
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helper: aggregate expression per sample
# ------------------------------------------------------------------
def _aggregate_by_sample(
    adata: AnnData,
    sample_key: str,
    layer: Optional[str] = None,
) -> pd.DataFrame:
    """
    Aggregate gene expression per biological sample.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    sample_key : str
        Column in adata.obs identifying biological samples.
    layer : str, optional
        Layer to use for expression values.

    Returns
    -------
    pd.DataFrame
        Samples x genes aggregated expression matrix.
    """

    logger.info("Aggregating expression by biological sample")

    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    if not isinstance(X, np.ndarray):
        X = X.toarray()

    df = pd.DataFrame(
        X,
        index=adata.obs[sample_key].values,
        columns=adata.var_names,
    )

    # Mean expression per sample
    sample_expr = df.groupby(df.index).mean()

    logger.info(
        f"Aggregated expression for {sample_expr.shape[0]} samples"
    )

    return sample_expr


# ------------------------------------------------------------------
# Main DE function
# ------------------------------------------------------------------
def sample_aware_de(
    adata: AnnData,
    sample_key: str,
    group_key: str,
    group1: str,
    group2: str,
    layer: Optional[str] = None,
    min_samples: int = 2,
) -> pd.DataFrame:
    """
    Perform sample-aware differential expression.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data.
    sample_key : str
        Column in adata.obs specifying biological samples.
    group_key : str
        Column in adata.obs specifying experimental condition.
    group1 : str
        First condition label.
    group2 : str
        Second condition label.
    layer : str, optional
        Layer to use for expression values.
    min_samples : int
        Minimum number of samples per group.

    Returns
    -------
    pd.DataFrame
        Differential expression results.
    """

    logger.info("Running sample-aware differential expression")

    # --------------------------------------------------------------
    # Aggregate expression per sample
    # --------------------------------------------------------------
    sample_expr = _aggregate_by_sample(
        adata,
        sample_key=sample_key,
        layer=layer,
    )

    # --------------------------------------------------------------
    # Map samples to groups
    # --------------------------------------------------------------
    sample_groups = (
        adata.obs[[sample_key, group_key]]
        .drop_duplicates()
        .set_index(sample_key)[group_key]
    )

    group1_samples = sample_groups[sample_groups == group1].index
    group2_samples = sample_groups[sample_groups == group2].index

    if len(group1_samples) < min_samples or len(group2_samples) < min_samples:
        raise ValueError(
            "Insufficient biological samples for DE analysis"
        )

    expr1 = sample_expr.loc[group1_samples]
    expr2 = sample_expr.loc[group2_samples]

    # --------------------------------------------------------------
    # Differential expression testing
    # --------------------------------------------------------------
    results = []

    for gene in sample_expr.columns:
        x1 = expr1[gene].values
        x2 = expr2[gene].values

        # Two-sample t-test (Welch)
        stat, pval = ttest_ind(
            x1,
            x2,
            equal_var=False,
            nan_policy="omit",
        )

        # Effect size (log2 fold change)
        mean1 = np.mean(x1) + 1e-8
        mean2 = np.mean(x2) + 1e-8
        log2fc = np.log2(mean2 / mean1)

        results.append(
            {
                "gene": gene,
                "log2FC": log2fc,
                "p_value": pval,
                "mean_group1": mean1,
                "mean_group2": mean2,
                "n_samples_group1": len(x1),
                "n_samples_group2": len(x2),
            }
        )

    de_df = pd.DataFrame(results)

    # --------------------------------------------------------------
    # Multiple testing correction
    # --------------------------------------------------------------
    _, p_adj, _, _ = multipletests(
        de_df["p_value"].values,
        method="fdr_bh",
    )

    de_df["p_adj"] = p_adj

    logger.info("Sample-aware DE completed successfully")

    return de_df

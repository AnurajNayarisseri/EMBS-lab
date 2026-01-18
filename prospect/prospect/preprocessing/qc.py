"""
qc.py
-----

Quality control module for PROSPECT.

This module performs conservative quality control on single-cell
and spatial transcriptomics data without imposing deterministic
assumptions about expression magnitude.

Design principles:
- Remove clearly low-quality cells and genes
- Avoid normalization, log-transformation, or HVG selection
- Preserve uncertainty for downstream probabilistic modeling
"""

import logging
from typing import Sequence

import numpy as np
import scanpy as sc
from anndata import AnnData


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Main QC function
# ------------------------------------------------------------------
def run_qc(
    adata: AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    max_mito_pct: float = 20.0,
    mito_prefixes: Sequence[str] = ("MT-", "mt-"),
    copy: bool = True,
) -> AnnData:
    """
    Perform conservative quality control.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object containing raw counts.
    min_genes : int, default=200
        Minimum number of detected genes per cell.
    min_cells : int, default=3
        Minimum number of cells a gene must be detected in.
    max_mito_pct : float, default=20.0
        Maximum allowed percentage of mitochondrial counts per cell.
    mito_prefixes : sequence of str
        Prefixes used to identify mitochondrial genes.
    copy : bool, default=True
        Whether to operate on a copy of AnnData.

    Returns
    -------
    AnnData
        QC-filtered AnnData object.
    """

    if copy:
        adata = adata.copy()

    logger.info("Starting PROSPECT quality control")
    logger.info(f"Initial cells: {adata.n_obs}")
    logger.info(f"Initial genes: {adata.n_vars}")

    # --------------------------------------------------------------
    # Identify mitochondrial genes
    # --------------------------------------------------------------
    mito_mask = np.zeros(adata.n_vars, dtype=bool)
    for prefix in mito_prefixes:
        mito_mask |= adata.var_names.str.startswith(prefix)

    adata.var["mt"] = mito_mask

    # --------------------------------------------------------------
    # Compute QC metrics
    # --------------------------------------------------------------
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    # --------------------------------------------------------------
    # Cell-level filtering
    # --------------------------------------------------------------
    before_cells = adata.n_obs

    sc.pp.filter_cells(adata, min_genes=min_genes)
    adata = adata[adata.obs.pct_counts_mt < max_mito_pct]

    after_cells = adata.n_obs

    logger.info(
        f"Filtered cells: {before_cells - after_cells} "
        f"(remaining: {after_cells})"
    )

    # --------------------------------------------------------------
    # Gene-level filtering
    # --------------------------------------------------------------
    before_genes = adata.n_vars

    sc.pp.filter_genes(adata, min_cells=min_cells)

    after_genes = adata.n_vars

    logger.info(
        f"Filtered genes: {before_genes - after_genes} "
        f"(remaining: {after_genes})"
    )

    # --------------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------------
    if adata.n_obs == 0:
        raise ValueError(
            "All cells removed during QC. "
            "Please check QC thresholds."
        )

    if adata.n_vars == 0:
        raise ValueError(
            "All genes removed during QC. "
            "Please check QC thresholds."
        )

    logger.info("Quality control completed successfully")

    return adata

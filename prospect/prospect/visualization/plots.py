"""
plots.py
--------

Visualization utilities for PROSPECT.

Design principles:
- Visualize uncertainty explicitly
- Avoid over-interpretation via hard clustering
- Nature Methods–style minimalism
- Publication-ready figures
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from anndata import AnnData


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


# ------------------------------------------------------------------
# Latent space visualization
# ------------------------------------------------------------------
def plot_latent_space(
    adata: AnnData,
    basis: str = "latent",
    color: Optional[str] = None,
    uncertainty: Optional[str] = None,
    title: Optional[str] = None,
    point_size: int = 8,
):
    """
    Plot 2D latent space representation.

    Parameters
    ----------
    adata : AnnData
        AnnData object with adata.obsm[basis].
    basis : str
        Key in adata.obsm containing latent coordinates.
    color : str, optional
        adata.obs key for coloring points.
    uncertainty : str, optional
        adata.obs key for uncertainty overlay.
    title : str, optional
        Plot title.
    point_size : int
        Size of scatter points.
    """

    if basis not in adata.obsm:
        raise ValueError(f"{basis} not found in adata.obsm")

    X = adata.obsm[basis][:, :2]

    plt.figure(figsize=(5, 5))

    if color is not None:
        c = adata.obs[color]
        sc = plt.scatter(
            X[:, 0],
            X[:, 1],
            c=c,
            s=point_size,
            cmap="viridis",
        )
        plt.colorbar(sc, label=color)

    elif uncertainty is not None:
        u = adata.obs[uncertainty]
        sc = plt.scatter(
            X[:, 0],
            X[:, 1],
            c=u,
            s=point_size,
            cmap="inferno",
        )
        plt.colorbar(sc, label="Uncertainty")

    else:
        plt.scatter(
            X[:, 0],
            X[:, 1],
            s=point_size,
            color="gray",
            alpha=0.7,
        )

    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# Soft identity heatmap
# ------------------------------------------------------------------
def plot_soft_identity_heatmap(
    identity_probs: np.ndarray,
    max_cells: int = 500,
):
    """
    Plot heatmap of soft identity probabilities.

    Parameters
    ----------
    identity_probs : np.ndarray
        Cells x programs probability matrix.
    max_cells : int
        Maximum number of cells to display.
    """

    if identity_probs.shape[0] > max_cells:
        idx = np.random.choice(
            identity_probs.shape[0],
            max_cells,
            replace=False,
        )
        identity_probs = identity_probs[idx]

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        identity_probs,
        cmap="viridis",
        cbar_kws={"label": "Identity probability"},
    )

    plt.xlabel("Latent programs")
    plt.ylabel("Cells")
    plt.title("Soft cell identity distributions")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# Detection probability plot
# ------------------------------------------------------------------
def plot_detection_probabilities(
    gene_probs: np.ndarray,
    n_top: int = 50,
):
    """
    Plot ranked detection probabilities for genes.

    Parameters
    ----------
    gene_probs : np.ndarray
        Detection probabilities per gene.
    n_top : int
        Number of top genes to display.
    """

    sorted_probs = np.sort(gene_probs)[::-1][:n_top]

    plt.figure(figsize=(6, 3))

    plt.bar(
        range(len(sorted_probs)),
        sorted_probs,
        color="steelblue",
    )

    plt.ylabel("Detection probability")
    plt.xlabel("Genes (ranked)")
    plt.title("Top gene detection probabilities")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# Sample-aware DE volcano plot
# ------------------------------------------------------------------
def plot_sample_aware_volcano(
    de_df: pd.DataFrame,
    p_col: str = "p_adj",
    fc_col: str = "log2FC",
    alpha: float = 0.05,
):
    """
    Volcano plot for sample-aware differential expression.

    Parameters
    ----------
    de_df : pd.DataFrame
        Differential expression results.
    p_col : str
        Adjusted p-value column.
    fc_col : str
        Log2 fold-change column.
    alpha : float
        Significance threshold.
    """

    if p_col not in de_df.columns or fc_col not in de_df.columns:
        raise ValueError("Required columns not found in DE DataFrame")

    plt.figure(figsize=(5, 4))

    sig = de_df[p_col] < alpha

    plt.scatter(
        de_df.loc[~sig, fc_col],
        -np.log10(de_df.loc[~sig, p_col]),
        s=10,
        color="gray",
        label="Not significant",
    )

    plt.scatter(
        de_df.loc[sig, fc_col],
        -np.log10(de_df.loc[sig, p_col]),
        s=10,
        color="firebrick",
        label="Significant",
    )

    plt.axhline(
        -np.log10(alpha),
        color="black",
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("log2 fold change")
    plt.ylabel("-log10 adjusted p-value")
    plt.legend(frameon=False)
    plt.title("Sample-aware differential expression")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# Spatial uncertainty plot
# ------------------------------------------------------------------
def plot_spatial_uncertainty(
    coords: np.ndarray,
    uncertainty: np.ndarray,
    title: str = "Spatial uncertainty",
    point_size: int = 20,
):
    """
    Plot spatial uncertainty over tissue coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates (spots x 2).
    uncertainty : np.ndarray
        Uncertainty score per spot.
    title : str
        Plot title.
    point_size : int
        Size of spatial points.
    """

    if coords.shape[0] != uncertainty.shape[0]:
        raise ValueError("coords and uncertainty must have same length")

    plt.figure(figsize=(5, 5))

    sc = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=uncertainty,
        cmap="inferno",
        s=point_size,
    )

    plt.colorbar(sc, label="Uncertainty")
    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.title(title)

    plt.tight_layout()
    plt.show()

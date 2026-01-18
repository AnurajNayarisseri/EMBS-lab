"""
run_prospect_example.py
----------------------

Minimal example demonstrating how to run the PROSPECT pipeline
on a single-cell or spatial transcriptomics dataset.

This script is intended as:
- A usage example for new users
- A sanity check for installation
- Supplementary material for the PROSPECT manuscript

Expected input:
- AnnData (.h5ad) file with raw counts in adata.X
- adata.obs must contain:
    - sample_id : biological sample identifier
    - condition : experimental condition
    - (optional) batch : batch identifier
- adata.obsm may contain:
    - spatial : spatial coordinates (for spatial transcriptomics)
"""

import logging
from pathlib import Path

import scanpy as sc

from prospect.preprocessing import run_qc
from prospect.models.detection import run_detection
from prospect.models.probabilistic_vae import ProbabilisticVAE, train_vae
from prospect.models.soft_identity import SoftIdentity
from prospect.statistics import sample_aware_de, score_de_results
from prospect.visualization import (
    plot_latent_space,
    plot_soft_identity_heatmap,
    plot_sample_aware_volcano,
)

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("PROSPECT-EXAMPLE")


# ------------------------------------------------------------------
# Main example workflow
# ------------------------------------------------------------------
def main():

    # --------------------------------------------------------------
    # Load example dataset
    # --------------------------------------------------------------
    input_path = Path("data/raw/example.h5ad")

    if not input_path.exists():
        raise FileNotFoundError(
            "Example dataset not found at data/raw/example.h5ad"
        )

    logger.info("Loading example dataset")
    adata = sc.read_h5ad(input_path)

    # --------------------------------------------------------------
    # Quality control
    # --------------------------------------------------------------
    logger.info("Running quality control")
    adata = run_qc(adata)

    # --------------------------------------------------------------
    # Detection probability modeling
    # --------------------------------------------------------------
    logger.info("Running detection modeling")
    adata = run_detection(adata)

    # --------------------------------------------------------------
    # Probabilistic VAE
    # --------------------------------------------------------------
    logger.info("Training probabilistic VAE")
    X = adata.X.toarray() if not isinstance(adata.X, (list, tuple)) else adata.X

    model = ProbabilisticVAE(
        n_genes=adata.n_vars,
        latent_dim=20,
    )

    model = train_vae(model, data=torch.tensor(X, dtype=torch.float32))

    with torch.no_grad():
        _, mu, _ = model(torch.tensor(X, dtype=torch.float32))

    adata.obsm["latent"] = mu.numpy()

    # --------------------------------------------------------------
    # Soft identity inference
    # --------------------------------------------------------------
    logger.info("Inferring soft identities")
    soft_id = SoftIdentity(
        latent_dim=20,
        n_programs=8,
    )

    identity_probs = soft_id(
        torch.tensor(adata.obsm["latent"], dtype=torch.float32)
    ).detach().numpy()

    adata.obsm["soft_identity"] = identity_probs

    # --------------------------------------------------------------
    # Sample-aware differential expression
    # --------------------------------------------------------------
    logger.info("Running sample-aware DE")
    de_df = sample_aware_de(
        adata,
        sample_key="sample_id",
        group_key="condition",
        group1="control",
        group2="treated",
    )

    de_df = score_de_results(de_df)

    # --------------------------------------------------------------
    # Visualization
    # --------------------------------------------------------------
    logger.info("Generating example plots")

    plot_latent_space(
        adata,
        basis="latent",
        title="PROSPECT latent space",
    )

    plot_soft_identity_heatmap(identity_probs)

    plot_sample_aware_volcano(de_df)

    logger.info("Example PROSPECT run completed successfully")


# ------------------------------------------------------------------
if __name__ == "__main__":
    import torch

    main()

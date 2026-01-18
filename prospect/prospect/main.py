"""
main.py
-------

End-to-end runner for PROSPECT:
PRObabilistic Single-cell and Spatial Epistemic Context Toolkit

This script orchestrates:
1. Quality control
2. Detection uncertainty modeling
3. Probabilistic latent representation
4. Batch latent modeling
5. Soft cell identity inference
6. Optional spatial constraints
7. Sample-aware differential expression
8. Confidence scoring
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

from prospect.preprocessing.qc import run_qc
from prospect.models.detection_model import estimate_detection_model
from prospect.models.probabilistic_vae import ProbabilisticVAE, train_vae
from prospect.models.batch_latent import BatchLatent, encode_batches
from prospect.models.soft_identity import SoftIdentity
from prospect.spatial.spatial_constraints import SpatialConstraint
from prospect.statistics.sample_aware_de import sample_aware_de
from prospect.statistics.confidence_scoring import score_de_results


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("PROSPECT")


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the PROSPECT probabilistic single-cell pipeline"
    )

    parser.add_argument("--input", required=True, help="Input .h5ad file")
    parser.add_argument("--output", default="prospect_output", help="Output directory")

    parser.add_argument("--sample_key", required=True, help="Sample ID column in adata.obs")
    parser.add_argument("--group_key", required=True, help="Condition column in adata.obs")
    parser.add_argument("--group1", required=True, help="Group 1 label")
    parser.add_argument("--group2", required=True, help="Group 2 label")

    parser.add_argument("--batch_key", default=None, help="Batch column in adata.obs")
    parser.add_argument("--latent_dim", type=int, default=20)
    parser.add_argument("--n_programs", type=int, default=10)

    parser.add_argument("--use_spatial", action="store_true", help="Enable spatial constraints")
    parser.add_argument("--spatial_radius", type=float, default=50.0)

    return parser.parse_args()


# ------------------------------------------------------------------
# Main workflow
# ------------------------------------------------------------------
def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading input data")
    adata = sc.read_h5ad(args.input)

    # --------------------------------------------------------------
    # 1. Quality control
    # --------------------------------------------------------------
    logger.info("Running quality control")
    adata = run_qc(adata)

    # --------------------------------------------------------------
    # 2. Detection uncertainty modeling
    # --------------------------------------------------------------
    logger.info("Estimating detection probabilities")
    counts = torch.tensor(adata.X.toarray(), dtype=torch.float32)
    detection_model = estimate_detection_model(counts)

    gene_detection_probs = torch.sigmoid(
        detection_model.gene_logits
    ).detach().cpu().numpy()

    adata.var["detection_probability"] = gene_detection_probs

    # --------------------------------------------------------------
    # 3. Probabilistic VAE
    # --------------------------------------------------------------
    logger.info("Training probabilistic VAE")
    vae = ProbabilisticVAE(
        n_genes=adata.n_vars,
        latent_dim=args.latent_dim,
    )

    vae = train_vae(vae, counts)

    with torch.no_grad():
        _, mu, logvar = vae(counts)

    latent = mu.detach().cpu().numpy()
    adata.obsm["latent"] = latent
    adata.obsm["latent_uncertainty"] = np.exp(logvar.detach().cpu().numpy())

    # --------------------------------------------------------------
    # 4. Batch latent modeling (optional)
    # --------------------------------------------------------------
    if args.batch_key is not None:
        logger.info("Modeling batch effects as latent variables")
        batch_ids, batch_map = encode_batches(adata.obs[args.batch_key].values)

        batch_model = BatchLatent(
            n_batches=len(batch_map),
            latent_dim=args.latent_dim,
        )

        z_batch = batch_model(
            torch.tensor(latent, dtype=torch.float32),
            batch_ids,
        ).detach().cpu().numpy()

        adata.obsm["latent_batch"] = z_batch

    # --------------------------------------------------------------
    # 5. Soft cell identity
    # --------------------------------------------------------------
    logger.info("Inferring soft cell identities")
    soft_identity = SoftIdentity(
        latent_dim=args.latent_dim,
        n_programs=args.n_programs,
    )

    identity_probs = soft_identity(
        torch.tensor(latent, dtype=torch.float32)
    ).detach().cpu().numpy()

    adata.obsm["soft_identity"] = identity_probs

    # --------------------------------------------------------------
    # 6. Spatial constraints (optional)
    # --------------------------------------------------------------
    if args.use_spatial:
        logger.info("Applying spatial constraints")
        if "spatial" not in adata.obsm:
            raise ValueError("Spatial coordinates not found in adata.obsm['spatial']")

        spatial = SpatialConstraint(
            coords=adata.obsm["spatial"],
            radius=args.spatial_radius,
        )

        spatial_uncertainty = spatial.uncertainty(
            torch.tensor(identity_probs, dtype=torch.float32)
        ).detach().cpu().numpy()

        adata.obs["spatial_uncertainty"] = spatial_uncertainty

    # --------------------------------------------------------------
    # 7. Sample-aware differential expression
    # --------------------------------------------------------------
    logger.info("Running sample-aware differential expression")
    de_df = sample_aware_de(
        adata,
        sample_key=args.sample_key,
        group_key=args.group_key,
        group1=args.group1,
        group2=args.group2,
    )

    de_path = output_dir / "sample_aware_de.csv"
    de_df.to_csv(de_path, index=False)

    # --------------------------------------------------------------
    # 8. Confidence scoring
    # --------------------------------------------------------------
    logger.info("Scoring confidence of DE results")
    de_df_scored = score_de_results(de_df)

    scored_path = output_dir / "sample_aware_de_confidence.csv"
    de_df_scored.to_csv(scored_path, index=False)

    # --------------------------------------------------------------
    # Save AnnData
    # --------------------------------------------------------------
    adata_out = output_dir / "prospect_results.h5ad"
    adata.write(adata_out)

    logger.info("PROSPECT analysis completed successfully")
    logger.info(f"Results saved to: {output_dir.resolve()}")


# ------------------------------------------------------------------
if __name__ == "__main__":
    main()

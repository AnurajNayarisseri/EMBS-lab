# PROSPECT Workflow Overview

This document provides a step-by-step overview of the **PROSPECT**
(PRObabilistic Single-cell and Spatial Epistemic Context Toolkit)
analysis workflow.

PROSPECT is designed as a **modular, uncertainty-aware pipeline**
for single-cell RNA sequencing and spatial transcriptomics data,
explicitly avoiding deterministic assumptions and overconfident
biological inference.

---

## Design Philosophy

PROSPECT is built around the following core principles:

- Uncertainty is intrinsic and should be modeled explicitly
- Cell identity is continuous, not discrete
- Biological samples are the unit of replication
- Batch effects are sources of variation, not artifacts to erase
- Confidence must be reported separately from statistical significance

Each step in the workflow reflects these principles.

---

## High-Level Workflow




---

## 1. Input Data

PROSPECT expects input data in **AnnData (`.h5ad`) format**, with:

- Raw counts stored in `adata.X`
- Cell metadata in `adata.obs`
- Gene metadata in `adata.var`
- (Optional) spatial coordinates in `adata.obsm["spatial"]`

Required `adata.obs` fields:
- `sample_id` (biological replicate)
- `condition` (experimental group)

Optional fields:
- `batch` (technical batch)

---

## 2. Quality Control (`preprocessing/qc.py`)

Purpose:
- Remove clearly low-quality cells and genes
- Preserve raw counts for probabilistic modeling

Key operations:
- Filter cells with very low gene counts
- Filter genes expressed in very few cells
- Remove cells with high mitochondrial content

What PROSPECT does **not** do at this stage:
- No normalization
- No log-transformation
- No HVG selection

---

## 3. Probabilistic Detection Modeling (`models/detection_model.py`)

Purpose:
- Model transcript detection as a probabilistic process
- Distinguish technical non-detection from biological absence

Approach:
- Bernoulli detection model
- Gene-specific detection probabilities
- Optional cell-level bias

Outputs:
- Gene-level detection probabilities stored in `adata.var`

---

## 4. Probabilistic Latent Representation (`models/probabilistic_vae.py`)

Purpose:
- Represent cellular transcriptional states as probability distributions

Approach:
- Variational Autoencoder (VAE)
- Latent variables with mean and variance per cell
- Explicit uncertainty retained in latent space

Outputs:
- Latent means: `adata.obsm["latent"]`
- Latent uncertainty: `adata.obsm["latent_uncertainty"]`

---

## 5. Latent Batch Modeling (`models/batch_latent.py`, optional)

Purpose:
- Model batch effects without removing biological signal

Approach:
- Learnable batch embeddings
- Additive or concatenated interaction with biological latents

Key principle:
- Batch effects are modeled, not corrected away

Outputs:
- Batch-adjusted latent representations (optional)

---

## 6. Soft Cell Identity Inference (`models/soft_identity.py`)

Purpose:
- Represent cell identity as a probability distribution

Approach:
- Latent biological programs represented by centroids
- Soft assignment via distance-based probabilities
- Entropy used as a measure of identity uncertainty

Outputs:
- Identity probabilities: `adata.obsm["soft_identity"]`

---

## 7. Spatial Constraints (`spatial/spatial_constraints.py`, optional)

Purpose:
- Incorporate spatial information conservatively

Approach:
- Define spatial neighborhoods
- Penalize strong identity disagreement among neighbors
- Increase uncertainty where spatial signals conflict

Key principle:
- Spatial information constrains plausibility, not certainty

Outputs:
- Spatial uncertainty scores: `adata.obs["spatial_uncertainty"]`

---

## 8. Sample-Aware Differential Expression (`statistics/sample_aware_de.py`)

Purpose:
- Perform statistically valid differential expression analysis

Approach:
- Aggregate expression at the biological sample level
- Use samples (not cells) as replicates
- Welch’s t-test with multiple testing correction

Avoids:
- Pseudoreplication
- Inflated significance from large cell counts

Outputs:
- Differential expression table with effect sizes and p-values

---

## 9. Confidence Scoring (`statistics/confidence_scoring.py`)

Purpose:
- Quantify confidence in inferred biological results

Evidence sources:
- Replication across samples
- Effect size magnitude
- Multimodal consistency (optional)
- Uncertainty estimates

Outputs:
- Confidence score (0–1)
- Confidence category:
  - High confidence
  - Moderate confidence
  - Hypothesis-generating

---

## 10. Visualization (`visualization/plots.py`)

Purpose:
- Enable transparent and interpretable exploration of results

Key plots:
- Latent space with uncertainty overlays
- Soft identity heatmaps
- Sample-aware volcano plots
- Spatial uncertainty maps

Design choices:
- No hard clustering by default
- Minimal, publication-ready aesthetics

---

## Summary

PROSPECT provides an end-to-end framework for analyzing single-cell
and spatial transcriptomics data with explicit modeling of uncertainty,
biological replication, and epistemic limits.

The workflow is modular, conservative, and designed to support
reproducible and defensible biological interpretation.

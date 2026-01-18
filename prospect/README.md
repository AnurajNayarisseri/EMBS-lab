# PROSPECT

## PRObabilistic Single-cell and Spatial Epistemic Context Toolkit

**PROSPECT** is an uncertainty-aware, probabilistic AI framework for the analysis of  
**single-cell RNA sequencing (scRNA-seq)** and **spatial transcriptomics** data.

The framework is designed to address fundamental limitations of existing pipelines by
explicitly modeling **measurement uncertainty**, **continuous cell identity**, and
**sample-level biological replication**, rather than relying on deterministic assumptions.

PROSPECT was developed at **Eminent Biosciences (EMBS)**.

---

## Rationale

Single-cell and spatial transcriptomics analyses often rely on assumptions that are
frequently violated in practice, including:

- Zero counts interpreted as true absence of expression  
- Discrete clustering of inherently continuous cell states  
- Aggressive batch correction that removes biological signal  
- Cell-level differential expression that ignores sample structure  

These practices can lead to **overconfident and irreproducible conclusions**.

PROSPECT reframes analysis as a **probabilistic inference problem**, explicitly
propagating uncertainty across all stages of the pipeline.

---

## Core Principles

PROSPECT is built on the following principles:

- **Uncertainty is intrinsic**, not noise to be eliminated  
- **Cell identity is continuous**, not categorical  
- **Batch effects are latent variables**, not artifacts to erase  
- **Biological samples are the unit of replication**  
- **Confidence must be reported explicitly**

---

## Key Features

- **Probabilistic transcript detection modeling**  
  Separates technical non-detection from biological absence without imputation.

- **Continuous latent representation of cell states**  
  Cells are represented as probability distributions in latent space.

- **Latent batch-effect modeling**  
  Batch effects are jointly modeled with biological variation.

- **Constraint-based spatial integration**  
  Spatial information constrains plausible cellular configurations without
  deterministic deconvolution.

- **Sample-aware differential expression**  
  Differential expression is performed at the biological sample level,
  avoiding pseudoreplication.

- **Explicit confidence scoring**  
  Results are categorized as high-confidence, moderate-confidence, or
  hypothesis-generating.

---

## Scope of Application

PROSPECT is suitable for:

- scRNA-seq datasets (10x Genomics, Smart-seq, etc.)
- Spatial transcriptomics platforms (e.g., Visium, Slide-seq)
- Multimodal single-cell studies
- Methodological benchmarking
- Reproducibility-focused biological analysis

---

## Repository Structure


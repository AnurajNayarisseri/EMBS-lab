# Methods

This document describes the computational methods, software, and workflows used for hybrid genome analysis and AI-guided prediction of candidate plasticizer-degrading enzymes in *Rhizobium pusense* strain KSSKSLAB04. The repository contains scripts and notebooks to reproduce the key analyses and figures.

---

## 1. Project Overview

The analysis consists of two major components:

1) **Genome analysis and visualization**
- Hybrid genome assembly integration and replicon-level visualization
- Functional annotation and detection of biodegradation-associated gene clusters
- Comparative genomics (FastANI, pangenome outputs, etc., as applicable)

2) **AI-guided enzyme prediction**
- Feature engineering from protein sequences and Pfam domain annotations
- Model training using Random Forest and a neural network (Keras)
- Model evaluation and interpretability (ROC curve, score distribution, feature importance)
- Scoring and ranking candidate enzymes from the KSSKSLAB04 proteome

---

## 2. Software and Environment

### 2.1 Computational Environment
All code in this repository was executed in a reproducible software environment defined by:

- `environment.yml` (recommended for full reproducibility)
- `requirements.txt` (Python-only installation)

### 2.2 Major Software Packages (Python)
Key Python libraries include:
- NumPy, Pandas (data processing)
- scikit-learn (Random Forest training and evaluation)
- TensorFlow/Keras (neural network training)
- Matplotlib/Seaborn (visualization)
- Joblib (model persistence)
- Jupyter (notebook workflows)

### 2.3 NGS/Bioinformatics Tools
Bioinformatics tools referenced in the analysis environment may include:
- Read QC/processing: fastp, porechop, filtlong, seqkit
- Assembly: canu, flye, unicycler, spades, shasta
- Alignment/polishing: minimap2, bwa, samtools, racon, pilon
- QC: quast, busco, checkm2
- Annotation: prokka/bakta, eggnog-mapper
- Comparative genomics: fastani, panaroo, mafft, iqtree
- Domain/function: hmmer (Pfam hmmscan), diamond

> Note: Exact tool usage depends on the dataset and analysis stage; large data outputs are deposited in Figshare (see Section 7).

---

## 3. Input Data

### 3.1 Example Inputs (GitHub)
Small example files for demonstration and testing are provided under:
- `data/example_inputs/`

These are reduced datasets intended for pipeline testing only.

### 3.2 Full Project Data (Figshare)
All full datasets and analysis outputs (including AI training data and genomics results) are deposited in Figshare and can be referenced in the manuscript:

- **Figshare dataset (DOI/link):** [INSERT FIGSHARE DOI/LINK]

The Figshare deposit includes:
- Positive and negative AI training datasets
- Esterase candidate data and scoring outputs
- Oxygenase outputs
- Mobile element results
- Variant calling outputs
- EggNOG results
- FastANI results
- Pangenome outputs
- Additional processed/intermediate results

---

## 4. AI Pipeline: Feature Engineering

### 4.1 Input Files
The AI pipeline uses:
- Protein FASTA files (`.faa` / `.fasta`)
- Pfam domain annotations in HMMER `domtblout` format (`*.pfam.domtblout`)

### 4.2 Feature Extraction
Features are constructed per protein sequence in the following order:

1. **Protein length** (1 feature)
2. **Amino acid composition (AAC)** for 20 amino acids in fixed order:
   `ACDEFGHIKLMNPQRSTVWY` (20 features)
3. **Motif feature** indicating a GXSXG-like esterase motif (1 feature)
4. **Pfam domain presence/absence** based on a learned domain vocabulary (binary vector)

A consistent feature order is required across training, evaluation, and candidate scoring.

Implementation:
- `src/utils/feature_builder.py`

---

## 5. AI Pipeline: Model Training

### 5.1 Random Forest Classifier
Training script:
- `src/ai/train_model.py`

Procedure:
1. Build Pfam domain vocabulary from positive and negative domtblout files
2. Extract features for positive and negative sets
3. Construct training matrix `X` and labels `y`
4. Perform an 80/20 stratified train/test split
5. Train Random Forest classifier with class balancing
6. Report ROC-AUC, PR-AUC, Accuracy, F1-score, and classification report
7. Perform 5-fold cross-validation (ROC-AUC)
8. Save trained model bundle:
   - `rf_plasticizer_model.joblib` containing:
     - `model`
     - `domain_vocab`

### 5.2 Keras Neural Network Model
Training script:
- `src/ai/train_keras_model.py`

Procedure:
1. Use the same feature engineering procedure and domain vocabulary
2. Scale features using `StandardScaler`
3. Perform an 80/20 stratified train/test split
4. Train a feed-forward neural network with dropout regularization
5. Early stopping is used to reduce overfitting
6. Report ROC-AUC and PR-AUC on the held-out test set

---

## 6. Evaluation, Interpretation, and Visualization

### 6.1 ROC Curve
Script:
- `src/ai/roc_curve.py`

Input:
- `roc_data.csv` (or an example version provided in `data/example_inputs/`)

Output:
- `results/figures/roc_curve.png`

### 6.2 Score Distribution
Script:
- `src/ai/score_distribution.py`

Inputs:
- Candidate scoring table (e.g., `KSSKSLAB04_plasticizer_ranked.csv`)
- Optional ROC evaluation data (e.g., `roc_data.csv`)

Outputs:
- `results/figures/score_distribution_candidates.png`
- `results/figures/score_distribution_by_class.png` (if class labels provided)

### 6.3 Feature Importance (Random Forest)
Script:
- `src/ai/feature_importance.py`

Inputs:
- `rf_plasticizer_model.joblib`

Outputs:
- `results/tables/feature_importance_all.csv`
- `results/tables/feature_importance_top30.csv`
- `results/figures/feature_importance_top15.png`

Notebook (optional visualization workflow):
- `src/ai/feature_importance_graph.ipynb`

---

## 7. Candidate Enzyme Scoring

Script:
- `src/ai/score_candidates.py`

Purpose:
Scores candidate proteins (e.g., esterases) from the KSSKSLAB04 proteome and outputs a ranked list of predicted plasticizer-degrading candidates.

Inputs:
- Candidate FASTA (e.g., `KSSKSLAB04_esterases.faa`)
- Candidate Pfam domtblout (e.g., `KSSK_esterase.pfam.domtblout`)
- Trained model bundle: `rf_plasticizer_model.joblib`

Output:
- `KSSKSLAB04_plasticizer_ranked.csv` (ranked by predicted probability)

---

## 8. Genome Visualization (Circular Map and Donut Plot)

Notebooks:
- `src/genomics/Genome circular map.ipynb`
- `src/genomics/DONUT GRAPH 3.ipynb`

These notebooks generate genome visualization graphics, including:
- Replicon-level circular genome map tracks
- Genome donut plot summaries
- Highlighting of labelled plasticizer-degrading gene clusters and related features

Outputs should be saved into:
- `results/figures/`

---

## 9. Reproducing the Analysis

### 9.1 Install Environment
Using conda:
```bash
conda env create -f environment.yml
conda activate plasticizer-ai

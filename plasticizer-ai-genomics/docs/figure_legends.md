# Figure Legends and Reproducibility Map

This document provides **publication-ready figure legends** and a **reproducibility map** linking each figure to the script/notebook and the expected inputs/outputs in this repository.

> **Important:** Full datasets and large outputs are available via Figshare: **[INSERT FIGSHARE DOI/LINK]**  
> This GitHub repository provides code and small example inputs for demonstration.

---

## Figure 1. Genome circular map of *Rhizobium pusense* KSSKSLAB04

### Legend (Manuscript-ready)
**Figure 1. Circular genome map of *Rhizobium pusense* strain KSSKSLAB04 highlighting plasticizer-degrading gene clusters and genomic features.**  
The circular genome map represents a chromosomal replicon of strain KSSKSLAB04 and displays the distribution of coding sequences (CDS), rRNA and tRNA genes, and annotated genome features. Labelled loci correspond to plasticizer-degrading gene clusters identified in the assembled genome (e.g., *pht, tph, benzoate, box, xyl, phn, lig, cat, oph,* and *pad*), supporting the metabolic potential for plasticizer and aromatic xenobiotic biodegradation. Additional tracks highlight mobile genetic elements, CRISPR–Cas loci, and prophage-associated regions. GC content and GC skew profiles indicate overall genomic stability with localized compositional variation suggestive of adaptive regions.

### Reproducibility
- **Notebook:** `src/genomics/Genome circular map.ipynb`  
- **Inputs:** genome assembly/replicon FASTA, annotation files (e.g., GFF/GenBank), optional feature tracks (mobile elements, prophages, CRISPR)  
- **Outputs:** `results/figures/genome_circular_map.png` (recommended filename)

---

## Figure 2. Genome donut plot summary of KSSKSLAB04

### Legend (Manuscript-ready)
**Figure 2. Genome donut plot summarizing key genomic features of *Rhizobium pusense* KSSKSLAB04.**  
The donut plot provides a visual summary of genome characteristics and functional categories, including annotation-derived feature counts and/or functional distributions relevant to biodegradation-associated pathways. The visualization highlights genome-scale functional composition and supports interpretation of metabolic versatility.

### Reproducibility
- **Notebook:** `src/genomics/DONUT GRAPH 3.ipynb`  
- **Inputs:** genome feature tables/annotation summaries (as used in notebook)  
- **Outputs:** `results/figures/genome_donut_plot.png` (recommended filename)

---

## Figure 3. ROC curve for AI-based plasticizer hydrolase prediction

### Legend (Manuscript-ready)
**Figure 3. Receiver operating characteristic (ROC) curve of the machine-learning classifier for plasticizer hydrolase prediction.**  
Model performance is summarized using an ROC curve computed from predicted probabilities on a held-out test set. The area under the curve (AUC) indicates classifier discrimination between positive (plasticizer-associated enzymes) and negative classes.

### Reproducibility
- **Script:** `src/ai/roc_curve.py`  
- **Inputs:** `roc_data.csv` with columns: `true` (0/1 labels) and `prob` (predicted probability)  
- **Outputs:** `results/figures/roc_curve.png`

> If ROC data are produced during training, ensure `train_model.py` writes `roc_data.csv` (or use the `roc_data_example.csv` for testing).

---

## Figure 4. Score distribution of predicted plasticizer hydrolase probabilities

### Legend (Manuscript-ready)
**Figure 4. Prediction score distributions for plasticizer hydrolase classification and candidate screening.**  
Histogram plots show the distribution of predicted probabilities across candidate proteins and (optionally) across positive vs negative classes. The score separation supports model discrimination and provides a basis for selecting high-confidence candidate enzymes for downstream analysis.

### Reproducibility
- **Script:** `src/ai/score_distribution.py`  
- **Inputs (candidate mode):** `KSSKSLAB04_plasticizer_ranked.csv` (output of `score_candidates.py`)  
- **Optional inputs (class mode):** `roc_data.csv` with `true` and `prob`  
- **Outputs:**  
  - `results/figures/score_distribution_candidates.png`  
  - `results/figures/score_distribution_by_class.png` (if ROC data provided)

---

## Figure 5. Feature importance for Random Forest model interpretation

### Legend (Manuscript-ready)
**Figure 5. Feature importance analysis of the Random Forest model used for plasticizer hydrolase prediction.**  
Feature importance values derived from the trained Random Forest classifier highlight the most informative sequence-derived and domain-based predictors contributing to classification. Top-ranked features provide interpretability regarding biochemical and functional determinants associated with plasticizer-degrading enzyme candidates.

### Reproducibility
- **Script:** `src/ai/feature_importance.py`  
- **Inputs:** `rf_plasticizer_model.joblib`  
- **Outputs:**  
  - `results/tables/feature_importance_all.csv`  
  - `results/tables/feature_importance_top30.csv`  
  - `results/figures/feature_importance_top15.png`

### Optional notebook visualization
- **Notebook:** `src/ai/feature_importance_graph.ipynb`  
- **Input:** `results/tables/feature_importance_top30.csv`  
- **Output:** `results/figures/feature_importance_top15.png` (or notebook-defined name)

---

## Figure 6. Ranked candidate plasticizer hydrolases from KSSKSLAB04 proteome

### Legend (Manuscript-ready)
**Figure 6 / Table S1. Ranked candidate plasticizer-degrading enzymes predicted from the KSSKSLAB04 proteome.**  
Candidate proteins (e.g., esterases) are scored using the trained machine-learning classifier, and predicted probabilities are used to rank candidates. The ranked list supports prioritization of high-confidence enzymes for downstream functional validation and pathway interpretation.

### Reproducibility
- **Script:** `src/ai/score_candidates.py`  
- **Inputs:**  
  - Candidate protein FASTA (e.g., `KSSKSLAB04_esterases.faa`)  
  - Candidate Pfam domtblout file (e.g., `KSSK_esterase.pfam.domtblout`)  
  - Trained model bundle `rf_plasticizer_model.joblib`  
- **Output:** `KSSKSLAB04_plasticizer_ranked.csv`  
- **Recommended save location:** `results/tables/`

---

## Supplementary Figures / Tables (Suggested)

### Figure S1. Neural network performance (Keras model)
- **Script:** `src/ai/train_keras_model.py`  
- **Outputs:** console performance metrics (ROC-AUC, PR-AUC)  
- **Recommended addition:** save probabilities and/or training curves for plotting.

### Table S2. Model performance summary
- **Script:** `src/ai/train_model.py`  
- **Outputs:** ROC-AUC, PR-AUC, Accuracy, F1-score, classification report, cross-validation ROC-AUC mean ± SD  
- **Recommended addition:** export metrics to `results/tables/model_metrics.csv`.

---

## Notes for Authors

1. Replace **[INSERT FIGSHARE DOI/LINK]** with the final DOI.  
2. Standardize output filenames in scripts/notebooks to ensure consistent reproduction.  
3. Store all final manuscript figures in `results/figures/` and tables in `results/tables/`.  
4. For journals requiring strict reproducibility, consider adding a `run_all.sh` or `Makefile` that regenerates all figures in one command.

---

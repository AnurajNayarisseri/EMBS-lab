# AI-Guided Plasticizer Hydrolase Prediction and Hybrid Genome Analysis of *Rhizobium pusense* KSSKSLAB04

## Overview

This repository contains Python scripts, notebooks, and analysis pipelines used in the study:

**“Hybrid genome sequencing and AI-guided enzyme prediction uncover plasticizer degradation pathways in *Rhizobium pusense* KSSKSLAB04.”**

The project integrates hybrid genome sequencing, bioinformatics analysis, and machine-learning approaches to identify candidate plasticizer-degrading enzymes and characterize genomic features associated with biodegradation potential.

---

## Repository Contents

### 1. Artificial Intelligence / Machine Learning Pipeline

Located in:

```
src/ai/
```

Includes:

* **train_model.py** – Random Forest classifier training using sequence-derived biochemical and domain features
* **train_keras_model.py** – Deep neural network training using TensorFlow/Keras
* **score_candidates.py** – Prediction and ranking of candidate plasticizer hydrolases
* **roc_curve.py** – ROC curve generation and performance evaluation
* **score_distribution.py** – Score distribution visualization
* **feature_importance.py** – Feature importance extraction
* **feature_importance_graph.ipynb** – Notebook for interpretability visualization

These scripts support:

* Feature extraction from protein sequences
* Machine-learning model training
* Performance evaluation
* Prediction of candidate biodegradation enzymes.

---

### 2. Genome Visualization

Located in:

```
src/genomics/
```

Includes:

* Circular genome map generation
* Genome donut visualization
* Annotation highlighting plasticizer degradation gene clusters.

---

### 3. Data Organization

```
data/
```

Contains:

* Example input files (small datasets only)
* Processed data required for reproducibility

**Note:** Large sequencing datasets are not included; accession links should be provided separately.

---

### 4. Results

```
results/
```

Stores:

* Figures generated from AI models
* Genome visualization outputs
* Tables of predicted candidate enzymes.

---

## Installation

### Option 1 — Conda Environment (Recommended)

```bash
conda env create -f environment.yml
conda activate plasticizer-ai
```

### Option 2 — pip Installation

```bash
pip install -r requirements.txt
```

---

## Dependencies

Typical required packages include:

* Python ≥ 3.8
* NumPy
* Pandas
* scikit-learn
* TensorFlow/Keras
* Matplotlib / Seaborn
* Biopython
* Joblib
* Jupyter Notebook

Exact versions are specified in the environment file.

---

## Running the Analyses

### Train Random Forest Model

```bash
python src/ai/train_model.py
```

### Train Neural Network Model

```bash
python src/ai/train_keras_model.py
```

### Score Candidate Plasticizer Hydrolases

```bash
python src/ai/score_candidates.py
```

### Generate Genome Visualization

```bash
jupyter notebook src/genomics/
```

### Generate Model Evaluation Plots

```bash
python src/ai/roc_curve.py
python src/ai/score_distribution.py
```

---

## Scientific Workflow Summary

### Feature Engineering

* Amino acid composition
* Sequence motif detection
* Protein length features
* Pfam domain presence/absence.

### Machine Learning Models

* Random Forest classifier
* Deep neural network model

### Evaluation Metrics

* ROC-AUC
* Precision-Recall AUC
* Cross-validation accuracy
* Feature importance analysis.

---

## Data Availability

If publishing:

* Genome assembly accession number
* Raw sequencing reads (SRA/ENA)
* Protein datasets used for training.

Links should be added here.

---

## Reproducibility Notes

To reproduce results:

1. Install dependencies
2. Provide appropriate input FASTA and domain files
3. Run training scripts
4. Generate prediction outputs and figures.

---

## Citation

If you use this code, please cite:

**Sharma K., Nayarisseri A., Singh S.K.**
Hybrid genome sequencing and AI-guided enzyme prediction uncover plasticizer degradation pathways in *Rhizobium pusense*.

(Add DOI after publication.)

---

## License

This repository is released under the MIT License (see LICENSE file).

---

## Contact

Dr. Anuraj Nayarisseri
Principal Scientist, Eminent Biosciences
Email: [anuraj@eminentbio.com](mailto:anuraj@eminentbio.com)

---

## Acknowledgements

This work integrates computational genomics, environmental microbiology, and artificial intelligence approaches for sustainable biodegradation research.

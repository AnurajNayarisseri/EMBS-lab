# 🧬 AI-Driven Discovery of Novel EGFR Inhibitors for Pancreatic Cancer

### 🚀 Generative AI | Molecular Simulation | Drug Discovery Pipeline

---

## 🏢 About the Lab

This project was developed at **Eminent Biosciences**, under the leadership of **Principal Scientist Dr. Anuraj Nayarisseri**.
The lab focuses on **AI-driven drug discovery, bioinformatics, and precision medicine research**.

---

## 📖 Project Overview

This repository implements a **generative AI-based drug discovery pipeline** for identifying novel inhibitors targeting the **Epidermal Growth Factor Receptor (EGFR)** in pancreatic cancer.

The workflow integrates:

* 🧠 Generative AI (VAE, GAN, Reinforcement Learning)
* 🔬 Molecular docking and binding analysis
* 📊 ADMET and drug-likeness evaluation
* ⚙️ Computational prioritization of lead molecules

👉 The pipeline is inspired by our research work on the discovery of a novel EGFR inhibitor (**EGFPAN_EMBS**).

---

## 🧠 Methodology

### 🔹 AI Framework

* Variational Autoencoder (VAE) → latent chemical space learning
* Generative Adversarial Network (GAN) → molecular diversity
* Reinforcement Learning (RL) → multi-objective optimization

### 🔹 Computational Pipeline

```text
SMILES → SELFIES → VAE → GAN → RL → Filtering → Docking → Lead Selection
```

### 🔹 Evaluation Metrics

* Binding affinity (Docking)
* Drug-likeness (Lipinski rules, QED)
* ADMET prediction
* Molecular stability (future MD integration)

---

## 📁 Repository Structure

```text
egfr-ai-drug-discovery/
│
├── data/                  # Input SMILES dataset
├── src/
│   ├── preprocessing.py   # SMILES/SELFIES processing
│   ├── vae.py             # Variational Autoencoder
│   ├── gan.py             # Generative Adversarial Network
│   ├── rl.py              # Reinforcement Learning
│   ├── reward.py          # Multi-objective scoring
│   ├── docking.py         # Gnina docking integration
│   └── utils.py           # Utility functions
│
├── notebooks/
│   └── exploration.ipynb  # Data exploration & visualization
│
├── config.yaml
├── requirements.txt
└── main.py
```

---

## ⚙️ Installation

### 🔹 Step 1: Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/egfr-ai-drug-discovery.git
cd egfr-ai-drug-discovery
```

### 🔹 Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### 🔹 Step 3 (Optional – Docking setup)

Install:

* RDKit
* OpenBabel
* Gnina

---

## 🚀 Usage

Run the full pipeline:

```bash
python main.py
```

Run notebook:

```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## 🧪 Key Features

✔ End-to-end AI drug discovery pipeline
✔ SELFIES-based robust molecular encoding
✔ Multi-objective optimization (QED + properties)
✔ Docking-ready workflow (Gnina integration)
✔ Modular and extensible architecture

---

## 📊 Results (Highlights)

* Generated **large chemical space (>1M compounds in concept)**
* Identified **high-affinity EGFR inhibitor candidates**
* Demonstrated improved **binding stability and selectivity**
* Achieved **nanomolar-level inhibition (experimental reference)**

---

## ⚠️ Limitations

* Docking integration requires external setup
* MD simulation (GROMACS) not included yet
* Reward function is simplified (can be extended to ADMET models)

---

## 🔬 Future Work

* Graph Neural Networks (ConfGAN)
* Molecular dynamics (MD simulation pipeline)
* MM/GBSA free energy calculations
* Clinical-scale ADMET modeling
* Automated AI + docking feedback loop

---

## 🤝 Contributors

* **Dr. Anuraj Nayarisseri** – Principal Scientist
* Eminent Biosciences Research Team

---

## 📜 License

This project is intended for **academic and research purposes only**.

---

## 📬 Contact

📧 Email: [anuraj@eminentbio.com](mailto:anuraj@eminentbio.com)
🏢 Organization: Eminent Biosciences

---

## ⭐ Acknowledgement

We acknowledge the contributions of the research team at **Eminent Biosciences** for advancing AI-driven drug discovery.

---

## 🚀 Citation

If you use this work, please cite:

> Generative AI–Driven Design and Experimental Validation of EGFPAN_EMBS, a Novel EGFR Inhibitor in Pancreatic Cancer

---

### 💡 Final Note

This repository demonstrates how **AI can accelerate next-generation anticancer drug discovery** by integrating computational intelligence with molecular science.

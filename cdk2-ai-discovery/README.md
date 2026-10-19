# CDK2 Generative-AI Discovery — research code

**Lead discussed in the manuscript:** CDKCRC-EMBS (PubChem CID 9829487)  
**Organization:** Eminent Biosciences (EMBS)

This repository organizes the supplied research code for the manuscript *Generative AI-Driven Discovery and Experimental Validation of a Novel CDK2 Inhibitor for the Treatment of Colorectal Cancer*. It includes separate implementations for fingerprint VAE/GAN models, an autoregressive SMILES GRU policy optimized with REINFORCE, a ConfGAN-style conformational-learning model, molecule evaluation, docking wrappers, kinase-selectivity scripts, and MD analysis utilities.

> **Reproducibility and scientific-integrity note.** This is a cleaned and extended research-code package based on the supplied ZIP and manuscript/supplementary descriptions. It is not a claim that every reported model weight, training dataset, or numerical result has been recovered. The supplied ZIP did not contain the original ConfGAN arrays/weights, a trained Dockproxy model, or all docking/MD inputs. Some original scripts used proxy scoring or incompatible representations; those are documented below. Do not describe a fingerprint-similarity fallback as docking, a heuristic as a trained neural model, or generated conformers as validated structures. Run the pipeline and independently verify results before reporting them.

## AI components and representations

1. **Fingerprint VAE** — 2,048-bit Morgan fingerprints (radius 2); continuous latent vector; reconstruction + KL loss. Fingerprints are descriptors, not molecular graphs or 3D coordinates.
2. **Fingerprint GAN** — generator/discriminator over Morgan fingerprints. Generated bit vectors are not automatically valid molecules; reconstruction from fingerprints is ambiguous and must be separately defined/validated.
3. **SMILES GRU + REINFORCE** — separate autoregressive policy over tokenized SMILES. Reward combines QED, a clearly labelled synthetic-accessibility heuristic, optional supplied Dockproxy values, and a discriminator-derived reward when configured. If Dockproxy is absent, similarity fallback is explicitly marked as a proxy, not docking or binding affinity.
4. **ConfGAN-style conformational model** — MM-GNN-inspired motif/atom feature encoder plus generator/discriminator for distance matrices and energy. The architecture is included; exact reproduction requires the original conformer arrays, atom ordering, conditioning features, and weights. The helper dataset builder uses RDKit/MMFF conformers as a development baseline, not DFT data.
5. **Downstream analysis** — RDKit validity/novelty, GNINA wrapper, MolDock result parser, selectivity workflow, and GROMACS/MDTraj utilities. External programs and correctly prepared structures are required.

The manuscript describes the VAE–GAN fingerprint components, the separate GRU/REINFORCE SMILES policy, and ConfGAN conformational learning as distinct components. This repository does not assume they were one jointly trained model.

## Repository map

```text
cdk2-ai-discovery/
├── configs/                 # example JSON configs
├── data/                    # supplied small reference CSVs; not the claimed 85k dataset
├── vae_gan_rl/              # VAE, fingerprint GAN, GRU policy, REINFORCE, training scripts
├── confgan/                 # conformational model, dataset builder, training/generation
├── docking/                 # GNINA wrapper and MolDock parser
├── selectivity/             # kinase panel docking helpers and plotting
├── md_simulation/            # GROMACS setup/run and MDTraj analysis
├── tests/                   # lightweight unit tests
├── results/                 # supplied example outputs (not automatically reproduced)
├── evaluate_generated_corrected_v5.py
├── validate_cdk2_generalizability.py
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.10 or 3.11 in a fresh environment. PyTorch installation varies by CPU/CUDA; install the matching build from the official PyTorch instructions first, then:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

RDKit installation through conda-forge is often more reliable on Windows:

```bash
conda create -n cdk2-ai python=3.11 -c conda-forge rdkit pandas numpy scikit-learn matplotlib seaborn
conda activate cdk2-ai
# Install a matching PyTorch build separately.
```

## Quick checks

```bash
python -m compileall vae_gan_rl confgan docking selectivity
python -m unittest discover -s tests -v
```

## Run the models

### 1. Fingerprint VAE/GAN training

From the repository root:

```bash
python -m vae_gan_rl.train_fingerprint --data data/cdk2_inhibitors.csv --smiles-column SMILES --outdir results/models --epochs 100
```

The 18-row CDK2 inhibitor CSV is only a smoke-test/reference set. It is far too small to substantiate robust generative-model claims or reproduce training on approximately 85,000 kinase-focused compounds. Supply the original curated dataset to reproduce that experiment.

### 2. GRU policy and REINFORCE

```bash
python -m vae_gan_rl.train_rl --data data/training_smiles.csv --smiles-column SMILES --outdir results/rl --steps 1000
```

This runs a separate SMILES policy. The default reward uses QED, a simple SA heuristic, and a similarity proxy unless a real externally calculated score column is supplied. Similarity is not docking. Use a validated docking/scoring service or the GNINA wrapper for docking-based evaluation.

### 3. ConfGAN

Build a **development** conformer dataset from the available SMILES:

```bash
python -m confgan.prepare_dataset --input data/cdk2_inhibitors.csv --smiles-column SMILES --output data/confgan_mmff_development.npz --max-atoms 32 --conformers-per-molecule 5
```

Then train:

```bash
python -m confgan.train --data data/confgan_mmff_development.npz --outdir results/confgan --epochs 100
```

This helper dataset uses RDKit ETKDG/MMFF, not DFT. It is a development/test path and must not be represented as the manuscript's approximately 42,000 conformers or DFT-optimized dataset. For faithful reproduction, provide the original `confgan_training.npz` (or reconstruct it from the original conformer files and documented feature pipeline) and original model weights.

### 4. Evaluate generated SMILES

```bash
python evaluate_generated_corrected_v5.py --generated results/generated_smiles.csv --train data/training_smiles.csv --out results/novelty.csv
```

Input generated CSV must contain `smiles` or `canonical_smiles`; training CSV must contain `smiles`. Novelty is relative only to the supplied training CSV.

### 5. Docking

Install GNINA and Open Babel separately and ensure `gnina` and `obabel` are on PATH. Prepare a correctly protonated receptor and a ligand-bound reference ligand for the binding-site box. Review `docking/gnina_screen.py` and its CLI before a large run. Do not use `--autobox_ligand` with the entire receptor; it expects a ligand structure. The wrapper requires an explicit reference ligand file for the autobox.

Molegro Virtual Docker is a separate commercial/external program; the parser does not run MVD itself. Only parse its output after checking column labels and units.

### 6. Molecular dynamics

The MD scripts require a prepared complex, force-field parameters, GROMACS and an independently reviewed topology. They do not establish binding affinity. Check all simulation settings and files before execution.

## Data and checkpoints

- CSVs in `data/` are the files supplied in the original ZIP; they are not the manuscript's approximately 85,000-molecule dataset.
- No original trained checkpoints were supplied in the ZIP.
- Do not commit private or licensed datasets, unpublished patient data, API keys, large intermediate files, or checkpoints unless you have the right to distribute them.

## What is and is not reproducible from this package

| Component | Status |
|---|---|
| VAE/GAN architecture | Code included; original checkpoints not supplied |
| SMILES GRU + REINFORCE | Runnable implementation included; reward scores need validation and original settings/data for exact reproduction |
| ConfGAN architecture | Code included; original conformer dataset/weights missing; MMFF builder is a development substitute only |
| Docking | Wrapper included; GNINA/Open Babel and valid receptor/ligand inputs required |
| Selectivity | Scripts/data subset supplied; several receptor structures and ligand file referenced by scripts may be absent |
| MD | Utility scripts supplied; original complex/topology/trajectories not all present |
| Manuscript numerical results | Not automatically reproduced by installing this repository |

## Citation

See `CITATION.cff`. Cite the manuscript and the relevant upstream software (RDKit, PyTorch, GNINA, GROMACS, MDTraj, and any licensed docking package) as appropriate.

## One-command AI workflow

After installing dependencies and confirming your CSV has the configured SMILES column:

```bash
python run_ai_workflow.py --data data/training_smiles.csv --smiles-column SMILES --outdir results/workflow --fingerprint-epochs 100 --rl-steps 500
```

To additionally run the **development-only** ConfGAN path using RDKit ETKDG/MMFF conformers:

```bash
python run_ai_workflow.py --data data/training_smiles.csv --smiles-column SMILES --outdir results/workflow --fingerprint-epochs 100 --rl-steps 500 --confgan
```

The workflow trains the fingerprint VAE/GAN and a separate SMILES-GRU policy, samples SMILES, and optionally trains ConfGAN on a generated development set. It does not imply that a fingerprint vector can be uniquely converted back into a molecule, and it does not reproduce the manuscript's original large training set or DFT conformers unless those original inputs are supplied. GNINA docking and GROMACS MD remain separate steps requiring valid structural inputs and external software.

<<<<<<< HEAD
# CDK2 AI Drug Discovery  
**Eminent Biosciences (EMBS)**

This repository contains the complete computational and analytical pipeline supporting the manuscript:

**Nayarisseri A. et al.**  
*Conformationally informed generative artificial intelligence enables discovery of a CDK2-targeting small molecule for colorectal cancer*  
(Nature Communications, under review)

The repository enables full reproducibility of the artificial intelligence–driven identification, docking, molecular dynamics validation, and prioritization of the lead compound **CDKCRC-EMBS (PubChem CID: 9829487)**.

---

## Scientific Rationale

Cyclin-dependent kinase 2 (CDK2) regulates the G1/S cell-cycle transition and is frequently hyperactivated in colorectal cancer. However, high structural similarity among CDK family members has limited the development of CDK2-targeting inhibitors.  

This work introduces a **conformation-aware generative artificial intelligence framework** that learns molecular energy landscapes and dynamic binding behavior rather than relying on static docking. The pipeline integrates deep generative models, reinforcement learning, deep-learning docking, and molecular dynamics simulations to identify CDK2-targeting small molecules with favorable stability and biological activity.

---

## Pipeline Overview

1. **Target structure**  
   Human CDK2 crystal structure (PDB ID: 1AQ1).

2. **Training chemistry**  
   Known CDK2 inhibitors and kinase-active compounds curated from PubChem and literature.

3. **Conformer learning**  
   ConfGAN trained on DFT-optimized distance–energy profiles using motif-based molecular graph neural networks.

4. **Molecular generation**  
   A VAE–GAN–reinforcement learning framework generates and optimizes drug-like molecules toward CDK2-binding chemical space.

5. **Deep-learning docking**  
   GNINA CNN-based docking combined with Molegro Virtual Docker (MolDock) scoring.

6. **Molecular dynamics**  
   100-ns GROMACS simulations (OPLS-AA, SPC/E, PME, Nosé–Hoover, MTK) to evaluate stability.

7. **Lead selection**  
   CDKCRC-EMBS identified as the most stable and highest-affinity CDK2-targeting compound.

---

## Repository Structure

=======
CDK2 AI Discovery

This project focuses on AI-driven discovery for CDK2.
>>>>>>> ebc203433922815d2fcf279f5100aecaf3b6eca4

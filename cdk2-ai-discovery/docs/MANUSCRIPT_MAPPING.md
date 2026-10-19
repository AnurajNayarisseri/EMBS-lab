# Manuscript-to-code mapping and recovery notes

Based on the supplied revised manuscript and Supplementary Information.

| Manuscript method/component | Repository location | Reproduction requirement / limitation |
|---|---|---|
| Morgan fingerprints (radius 2, 2,048 bits) and VAE | `vae_gan_rl/vae.py`, `vae_gan_rl/train_fingerprint.py` | The supplied ZIP contains a small CSV; manuscript describes a much larger kinase-focused dataset. Original weights were not supplied. |
| GAN on fingerprints | `vae_gan_rl/gan.py`, `vae_gan_rl/train_fingerprint.py` | Fingerprint vectors are not directly valid molecules; molecular reconstruction must be specified and validated separately. |
| Separate GRU SMILES policy and REINFORCE | `vae_gan_rl/smiles_gru.py`, `vae_gan_rl/train_rl.py`, `vae_gan_rl/rl.py` | Reimplemented as a standalone policy because the supplied RL file did not contain a GRU sequence policy. The simple SA term is explicitly a heuristic. No trained Dockproxy was supplied. |
| ConfGAN conformational learning | `confgan/model.py`, `confgan/train.py`, `confgan/prepare_dataset.py` | Original `confgan_training.npz`, motif semantics, DFT conformer inputs and checkpoint were not in the ZIP. The dataset builder makes an ETKDG/MMFF development dataset only, not the manuscript's reported DFT data. |
| Molecular validity/novelty | `evaluate_generated_corrected_v5.py` | Novelty is relative only to the CSV passed to the tool; it is not proof of global novelty or biological activity. |
| GNINA docking | `docking/gnina_screen.py` | Requires GNINA, Open Babel, prepared receptor PDBQT and a ligand-bound reference ligand to define the binding box. CNN score/affinity are predictions, not experimental binding values. |
| MolDock result parsing | `docking/mvd_parser.py` | Requires actual MVD output and verified column/units. This script does not run the proprietary program. |
| Kinase selectivity | `selectivity/` | The supplied receptor panel is incomplete relative to the list in the old script; provide all receptor structures and ligand inputs before claiming a complete panel. |
| MD | `md_simulation/` | Requires original complex, topology, force-field parameters and trajectories. Script presence alone does not reproduce the reported 3 x 200 ns simulations. |

## Important original ZIP issues addressed

- Resolved a Git merge conflict in the README.
- Replaced the original fingerprint-GAN training loop's use of unrelated reference SMILES as if they were rewards with a distinct GRU/REINFORCE policy implementation.
- Corrected the GNINA wrapper to require a reference ligand for the docking box rather than passing the entire receptor as `--autobox_ligand`.
- Added smoke tests and runnable CLI entry points.

These changes improve usability but should be described as code cleanup/extensions, not as recovery of missing historical code or proof that the manuscript's exact results have been reproduced.

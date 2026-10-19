"""Run the reproducible training/evaluation stages available in this repository.

This orchestrator does not run external docking or MD automatically because those
steps require user-supplied receptor/ligand structures, force fields, and software.
"""
import argparse, subprocess, sys
from pathlib import Path

def run(cmd):
    print('\n+',' '.join(map(str,cmd)),flush=True)
    subprocess.run(list(map(str,cmd)),check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/training_smiles.csv')
    p.add_argument('--smiles-column',default='SMILES')
    p.add_argument('--outdir',default='results/workflow')
    p.add_argument('--fingerprint-epochs',type=int,default=100)
    p.add_argument('--rl-steps',type=int,default=500)
    p.add_argument('--confgan',action='store_true',help='Build a development ETKDG/MMFF dataset and train ConfGAN; not the original DFT dataset.')
    a=p.parse_args(); root=Path.cwd(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    run([sys.executable,'-m','vae_gan_rl.train_fingerprint','--data',a.data,'--smiles-column',a.smiles_column,'--outdir',out/'fingerprint_models','--epochs',a.fingerprint_epochs])
    run([sys.executable,'-m','vae_gan_rl.train_rl','--data',a.data,'--smiles-column',a.smiles_column,'--outdir',out/'rl','--steps',a.rl_steps])
    run([sys.executable,'-m','vae_gan_rl.generate_smiles','--checkpoint',out/'rl'/'gru_policy.pt','--data',a.data,'--smiles-column',a.smiles_column,'--out',out/'generated_smiles.csv','--n',1000])
    if a.confgan:
        dataset=out/'confgan_mmff_development.npz'
        run([sys.executable,'-m','confgan.prepare_dataset','--input',a.data,'--smiles-column',a.smiles_column,'--output',dataset])
        run([sys.executable,'-m','confgan.train','--data',dataset,'--outdir',out/'confgan'])
    print('\nWorkflow completed. Inspect metrics and generated molecules; do not interpret heuristic rewards as docking affinity.')
if __name__=='__main__': main()

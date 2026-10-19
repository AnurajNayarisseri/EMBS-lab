"""Train separate Morgan-fingerprint VAE and GAN baselines.

Generated fingerprints are descriptor vectors, not valid molecular structures.
This script does not claim that a fingerprint can be uniquely decoded to SMILES.
"""
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from .vae import MolecularVAE, vae_loss
from .gan import MolGenerator, MolDiscriminator


def fingerprint(smiles, n_bits=2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    rows=[]
    for s in smiles:
        mol=Chem.MolFromSmiles(str(s))
        if mol is None: continue
        rows.append(np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.float32))
    if not rows: raise ValueError('No valid SMILES in input file.')
    return np.stack(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', default='data/cdk2_inhibitors.csv')
    ap.add_argument('--smiles-column', default='SMILES')
    ap.add_argument('--outdir', default='results/models')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--latent-dim', type=int, default=128)
    ap.add_argument('--fp-bits', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out=Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(args.data)
    if args.smiles_column not in df: raise ValueError(f'Missing column {args.smiles_column!r}')
    X=fingerprint(df[args.smiles_column].dropna(), args.fp_bits)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader=DataLoader(TensorDataset(torch.tensor(X)), batch_size=args.batch_size, shuffle=True)
    vae=MolecularVAE(args.fp_bits,args.latent_dim).to(device)
    G=MolGenerator(args.latent_dim,args.fp_bits).to(device)
    D=MolDiscriminator(args.fp_bits).to(device)
    optv=torch.optim.Adam(vae.parameters(),lr=args.lr)
    optg=torch.optim.Adam(G.parameters(),lr=args.lr)
    optd=torch.optim.Adam(D.parameters(),lr=args.lr)
    bce=nn.BCELoss(); history=[]
    for epoch in range(1,args.epochs+1):
        sums={'vae':0.,'d':0.,'g':0.}; n=0
        for (x,) in loader:
            x=x.to(device); b=len(x); n+=b
            # VAE
            recon,mu,logvar=vae(x)
            lv=vae_loss(x,recon,mu,logvar)/b
            optv.zero_grad(); lv.backward(); optv.step()
            # GAN discriminator
            z=torch.randn(b,args.latent_dim,device=device); fake=G(z)
            pr=D(x); pf=D(fake.detach())
            ld=bce(pr,torch.ones_like(pr))+bce(pf,torch.zeros_like(pf))
            optd.zero_grad(); ld.backward(); optd.step()
            # GAN generator: adversarial objective only; RL is separate in train_rl.py
            z=torch.randn(b,args.latent_dim,device=device); fake=G(z); pf=D(fake)
            lg=bce(pf,torch.ones_like(pf))
            optg.zero_grad(); lg.backward(); optg.step()
            sums['vae']+=float(lv.detach())*b; sums['d']+=float(ld.detach())*b; sums['g']+=float(lg.detach())*b
        row={'epoch':epoch,**{k:v/max(n,1) for k,v in sums.items()}}
        history.append(row)
        if epoch==1 or epoch%10==0 or epoch==args.epochs:
            print(json.dumps(row))
            torch.save(vae.state_dict(),out/'vae.pt'); torch.save(G.state_dict(),out/'gan_generator.pt'); torch.save(D.state_dict(),out/'gan_discriminator.pt')
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    meta={'dataset':str(args.data),'valid_molecules':len(X),'fp_bits':args.fp_bits,'radius':2,'latent_dim':args.latent_dim,'epochs':args.epochs,'device':str(device),'rdkit_version':rdBase.rdkitVersion,'warning':'Generated fingerprint vectors are not decoded to valid molecules by this script.'}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__': main()

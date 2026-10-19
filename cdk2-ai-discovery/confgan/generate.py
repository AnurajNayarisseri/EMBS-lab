"""Generate distance matrices/coordinates from a trained ConfGAN checkpoint.
Coordinates are geometric proposals; atom mapping, stereochemistry, valence,
and force-field relaxation must be validated before scientific interpretation.
"""
import argparse
from pathlib import Path
import numpy as np, torch
from rdkit import Chem
from rdkit.Chem import AllChem
from .model import ConfGAN

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True,help='NPZ containing motifs and optionally smiles');p.add_argument('--out',default='results/confgan_generated.sdf');p.add_argument('--max-atoms',type=int,default=32);a=p.parse_args()
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ck=torch.load(a.checkpoint,map_location=device,weights_only=False); model=ConfGAN(n_atoms=a.max_atoms).to(device); model.load_state_dict(ck['model'] if isinstance(ck,dict) and 'model' in ck else ck);model.eval()
 data=np.load(a.data,allow_pickle=True); motifs=torch.tensor(data['motifs'],dtype=torch.float32,device=device)
 with torch.no_grad(): cond=model.mmg(motifs); z=torch.randn(len(motifs),128,device=device); distances,energies=model.G(z,cond)
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out.with_suffix('.npz'),distances=distances.cpu().numpy(),energies=energies.cpu().numpy())
 # Export SDF only where parent SMILES are available and atom counts exactly match.
 if 'smiles' in data.files:
  writer=Chem.SDWriter(str(out)); written=0
  for i,smi in enumerate(data['smiles']):
   mol=Chem.MolFromSmiles(str(smi))
   if mol is None: continue
   mol=Chem.AddHs(mol)
   if mol.GetNumAtoms()!=a.max_atoms: continue
   D=distances[i].detach().cpu(); n=D.shape[0]; J=torch.eye(n)-torch.ones(n,n)/n; B=-.5*J@(D**2)@J; vals,vecs=torch.linalg.eigh(B); xyz=vecs[:,-3:]*torch.sqrt(torch.clamp(vals[-3:],min=1e-6)); conf=Chem.Conformer(mol.GetNumAtoms())
   for j,xyzj in enumerate(xyz.numpy()): conf.SetAtomPosition(j,xyzj.tolist())
   mol.RemoveAllConformers();mol.AddConformer(conf);mol.SetProp('PredictedEnergy',str(float(energies[i].cpu())));mol.SetProp('NOTE','Generated coordinates require geometry and chemistry validation');writer.write(mol);written+=1
  writer.close();print(f'Wrote {written} SDF records (only exact atom-count matches).')
 else: print('No SMILES array found; wrote distance matrices/energies only.')
 print(f'Wrote raw model outputs to {out.with_suffix(".npz")}')
if __name__=='__main__':main()

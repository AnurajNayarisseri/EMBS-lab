"""Create an RDKit ETKDG/MMFF *development* dataset for ConfGAN smoke tests.
This does not reproduce the manuscript's DFT-optimized conformer dataset.
"""
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem,rdMolDescriptors

def atom_features(atom,dim=32):
    x=np.zeros(dim,dtype=np.float32)
    vals=[atom.GetAtomicNum(),atom.GetDegree(),atom.GetFormalCharge(),int(atom.GetIsAromatic()),int(atom.IsInRing()),atom.GetTotalNumHs(),atom.GetMass()/100.0]
    for i,v in enumerate(vals): x[i]=float(v)
    x[7+min(atom.GetAtomicNum(),dim-8)]=1.0
    return x

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--smiles-column',default='SMILES');p.add_argument('--output',required=True);p.add_argument('--max-atoms',type=int,default=32);p.add_argument('--conformers-per-molecule',type=int,default=5);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    df=pd.read_csv(a.input); motifs=[]; distances=[]; energies=[]; smiles_used=[]
    for smi in df[a.smiles_column].dropna().astype(str):
        mol=Chem.MolFromSmiles(smi)
        if mol is None: continue
        mol=Chem.AddHs(mol)
        if mol.GetNumAtoms()>a.max_atoms: continue
        ids=AllChem.EmbedMultipleConfs(mol,numConfs=a.conformers_per_molecule,params=AllChem.ETKDGv3())
        if not ids: continue
        props=AllChem.MMFFGetMoleculeProperties(mol)
        if props is None: continue
        feat=np.zeros((a.max_atoms,32),dtype=np.float32)
        for i,atom in enumerate(mol.GetAtoms()): feat[i]=atom_features(atom)
        for cid in ids:
            ff=AllChem.MMFFGetMoleculeForceField(mol,props,confId=cid)
            if ff is None: continue
            ff.Minimize(maxIts=300); e=float(ff.CalcEnergy()); conf=mol.GetConformer(cid); xyz=np.asarray(conf.GetPositions(),dtype=np.float32)
            n=len(xyz); d=np.zeros((a.max_atoms,a.max_atoms),dtype=np.float32); delta=xyz[:,None,:]-xyz[None,:,:]; d[:n,:n]=np.sqrt((delta**2).sum(-1))
            motifs.append(feat); distances.append(d); energies.append(e); smiles_used.append(smi)
    if not distances: raise RuntimeError('No conformers generated; check input SMILES and max-atoms.')
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,motifs=np.stack(motifs),distances=np.stack(distances),energies=np.asarray(energies,dtype=np.float32),smiles=np.asarray(smiles_used),source='RDKit ETKDG/MMFF development dataset; not DFT')
    print(f'Wrote {len(distances)} conformers to {a.output}. This is not the original DFT dataset.')
if __name__=='__main__':main()

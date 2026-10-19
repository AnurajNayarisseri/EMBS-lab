"""Reward utilities for the SMILES policy.

The synthetic-accessibility term here is a transparent heuristic, NOT the
published Ertl SA score. The similarity fallback is NOT docking or affinity.
"""
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, rdMolDescriptors, rdFingerprintGenerator

FPGEN=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)

def sa_heuristic(mol):
    heavy=mol.GetNumHeavyAtoms(); rings=rdMolDescriptors.CalcNumRings(mol); rot=rdMolDescriptors.CalcNumRotatableBonds(mol)
    return 0.1*heavy + 0.5*rings + 0.3*rot

def similarity_proxy(mol, reference_smiles):
    """Max Tanimoto to references. This is a chemical-similarity proxy only."""
    fp=FPGEN.GetFingerprint(mol); sims=[]
    for s in reference_smiles:
        rm=Chem.MolFromSmiles(str(s))
        if rm is not None: sims.append(DataStructs.TanimotoSimilarity(fp,FPGEN.GetFingerprint(rm)))
    return max(sims,default=0.0)

def score_smiles(smiles, reference_smiles=(), dockproxy_score=None):
    mol=Chem.MolFromSmiles(str(smiles))
    if mol is None: return {'reward':-1.0,'valid':False,'qed':0.0,'sa_heuristic':None,'dockproxy':None,'similarity_proxy':None}
    qed=float(QED.qed(mol)); sa=float(sa_heuristic(mol)); sa_norm=float(np.exp(-0.1*sa))
    sim=similarity_proxy(mol,reference_smiles) if dockproxy_score is None else None
    dock=float(dockproxy_score) if dockproxy_score is not None else float(sim or 0.0)
    reward=0.4*qed+0.3*sa_norm+0.3*dock
    return {'reward':float(reward),'valid':True,'qed':qed,'sa_heuristic':sa,'dockproxy':dockproxy_score,'similarity_proxy':sim}

def reward(smiles, reference_smiles=(), dockproxy_score=None):
    return score_smiles(smiles,reference_smiles,dockproxy_score)['reward']

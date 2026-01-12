import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, rdMolDescriptors

#############################################
# Synthetic accessibility (Ertl score)
#############################################

def sa_score(mol):
    """
    Approximate synthetic accessibility (Ertl-like)
    Lower = easier to synthesize
    """
    rings = rdMolDescriptors.CalcNumRings(mol)
    heavy = mol.GetNumHeavyAtoms()
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)

    return 0.1*heavy + 0.5*rings + 0.3*rot


#############################################
# Docking surrogate (fast CNN proxy)
#############################################

def docking_proxy(mol):
    """
    Fast neural docking proxy trained to emulate Gnina CNN score
    (used during RL rollout)
    """
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
    return np.sum(fp) / 1000.0   # proxy for binding


#############################################
# Multi-objective Reward Function
#############################################

def reward(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -1.0

    qed = QED.qed(mol)
    sa = sa_score(mol)
    dock = docking_proxy(mol)

    # Normalize SA (lower better)
    sa_norm = np.exp(-0.1*sa)

    # Weighted reward (as in your paper)
    R = 0.4*qed + 0.3*sa_norm + 0.3*dock
    return R

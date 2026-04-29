from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import QED


# -----------------------------
# Validation
# -----------------------------
def is_valid(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


# -----------------------------
# Lipinski Rule Scoring
# -----------------------------
def lipinski_score(mol):
    score = 0

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    if mw <= 500:
        score += 1
    if logp <= 5:
        score += 1
    if hbd <= 5:
        score += 1
    if hba <= 10:
        score += 1

    return score


# -----------------------------
# Basic Property Score
# -----------------------------
def property_score(mol):
    score = 0

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)

    # Preferred ranges
    if 200 <= mw <= 600:
        score += 1
    if -1 <= logp <= 5:
        score += 1

    return score


# -----------------------------
# QED Score
# -----------------------------
def qed_score(mol):
    try:
        return QED.qed(mol)
    except:
        return 0


# -----------------------------
# Synthetic Accessibility (Placeholder)
# -----------------------------
def synthetic_accessibility(mol):
    """
    Placeholder: replace with real SA score if needed
    Lower complexity = higher score
    """
    num_atoms = mol.GetNumAtoms()

    if num_atoms < 20:
        return 1
    elif num_atoms < 40:
        return 0.5
    else:
        return 0


# -----------------------------
# Optional Docking Score (Hook)
# -----------------------------
def docking_score(smiles):
    """
    Placeholder for docking integration (Gnina, AutoDock, etc.)
    Return normalized score (higher = better)
    """
    return 0  # Replace with real docking score later


# -----------------------------
# Final Reward Function
# -----------------------------
def reward(smiles):
    # Invalid molecule penalty
    if not is_valid(smiles):
        return -2

    mol = Chem.MolFromSmiles(smiles)

    # Individual components
    lipinski = lipinski_score(mol)
    prop = property_score(mol)
    qed = qed_score(mol)
    sa = synthetic_accessibility(mol)
    dock = docking_score(smiles)

    # Weighted reward
    total_score = (
        1.5 * lipinski +
        1.0 * prop +
        2.0 * qed +
        1.0 * sa +
        2.0 * dock
    )

    return total_score


# -----------------------------
# Batch Scoring
# -----------------------------
def batch_reward(smiles_list):
    results = []
    for smi in smiles_list:
        r = reward(smi)
        results.append((smi, r))

    return sorted(results, key=lambda x: -x[1])


# -----------------------------
# Debug / Example Run
# -----------------------------
if __name__ == "__main__":
    test_smiles = [
        "CCO",
        "CCN",
        "CCC",
        "INVALID",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    ]

    print("Testing reward function:\n")

    for smi in test_smiles:
        print(f"{smi} → {reward(smi):.3f}")

    print("\nTop ranked:")
    ranked = batch_reward(test_smiles)
    for smi, score in ranked:
        print(smi, score)
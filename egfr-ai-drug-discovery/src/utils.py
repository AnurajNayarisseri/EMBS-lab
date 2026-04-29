import os
import random
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Device handling
# -----------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# SMILES Validation
# -----------------------------
def validate(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def clean_smiles(smiles_list):
    return [s for s in smiles_list if validate(s)]


# -----------------------------
# Canonical SMILES
# -----------------------------
def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol)
    except:
        return None


# -----------------------------
# Remove duplicates
# -----------------------------
def remove_duplicates(smiles_list):
    return list(set(smiles_list))


# -----------------------------
# Basic Molecular Properties
# -----------------------------
def compute_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol)
    }


# -----------------------------
# Batch Property Calculation
# -----------------------------
def batch_properties(smiles_list):
    results = []
    for smi in smiles_list:
        props = compute_properties(smi)
        if props:
            results.append((smi, props))
    return results


# -----------------------------
# Logging
# -----------------------------
def log(message):
    print(f"[INFO] {message}")


def warn(message):
    print(f"[WARNING] {message}")


def error(message):
    print(f"[ERROR] {message}")


# -----------------------------
# Save / Load utilities
# -----------------------------
def save_list(file_path, data_list):
    with open(file_path, "w") as f:
        for item in data_list:
            f.write(str(item) + "\n")


def load_list(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        return [line.strip() for line in f]


# -----------------------------
# Top-K Selection
# -----------------------------
def top_k(results, k=10):
    """
    results: list of (item, score)
    """
    return sorted(results, key=lambda x: -x[1])[:k]


# -----------------------------
# Normalize Scores
# -----------------------------
def normalize_scores(scores):
    scores = np.array(scores)
    if len(scores) == 0:
        return scores

    min_val = scores.min()
    max_val = scores.max()

    if max_val == min_val:
        return scores

    return (scores - min_val) / (max_val - min_val)


# -----------------------------
# Progress Display
# -----------------------------
def progress(current, total):
    percent = (current / total) * 100
    print(f"\rProgress: {current}/{total} ({percent:.2f}%)", end="")


# -----------------------------
# Debug / Example Run
# -----------------------------
if __name__ == "__main__":
    set_seed(42)

    sample_smiles = ["CCO", "CCN", "INVALID", "CCC"]

    print("Original:", sample_smiles)

    cleaned = clean_smiles(sample_smiles)
    print("Cleaned:", cleaned)

    print("\nProperties:")
    for smi in cleaned:
        print(smi, compute_properties(smi))

    print("\nTop-K example:")
    results = [("A", 1.2), ("B", 3.4), ("C", 2.1)]
    print(top_k(results, k=2))
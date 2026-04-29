import os
import yaml
import random
import pandas as pd

from src.preprocessing import smiles_to_selfies, selfies_to_smiles
from src.utils import validate
from src.rl import optimize
from src.reward import reward

# Optional imports (safe fallback if not installed)
try:
    from src.docking import run_gnina
    DOCKING_AVAILABLE = True
except:
    DOCKING_AVAILABLE = False


# ---------------------------
# Load config
# ---------------------------
def load_config(path="config.yaml"):
    if not os.path.exists(path):
        print("Config file not found. Using defaults.")
        return {
            "latent_dim": 64,
            "hidden_dim": 256,
            "batch_size": 32,
            "epochs": 10,
            "max_length": 100
        }
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------
# Load dataset
# ---------------------------
def load_data(path="data/egfr_smiles.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    
    df = pd.read_csv(path)
    smiles = df["smiles"].dropna().tolist()
    
    print(f"Loaded {len(smiles)} molecules")
    return smiles


# ---------------------------
# Simple augmentation (mock generation step)
# ---------------------------
def generate_variants(smiles_list, n=50):
    """Simple mutation-based generator (placeholder for VAE/GAN output)"""
    new_smiles = []
    
    for smi in smiles_list:
        for _ in range(n):
            # simple mutation: shuffle string (toy example)
            mutated = ''.join(random.sample(smi, len(smi)))
            new_smiles.append(mutated)
    
    return new_smiles


# ---------------------------
# Validation step
# ---------------------------
def filter_valid(smiles_list):
    valid = []
    for smi in smiles_list:
        if validate(smi):
            valid.append(smi)
    print(f"Valid molecules: {len(valid)}/{len(smiles_list)}")
    return valid


# ---------------------------
# Scoring step
# ---------------------------
def score_molecules(smiles_list):
    scored = []
    for smi in smiles_list:
        r = reward(smi)
        scored.append((smi, r))
    return sorted(scored, key=lambda x: -x[1])


# ---------------------------
# Docking step
# ---------------------------
def docking_pipeline(top_molecules, limit=5):
    if not DOCKING_AVAILABLE:
        print("Docking module not available. Skipping...")
        return
    
    print("\nRunning docking on top candidates...\n")
    
    for smi, score in top_molecules[:limit]:
        print(f"Docking: {smi} (score={score})")
        try:
            run_gnina(smi)
        except Exception as e:
            print(f"Docking failed for {smi}: {e}")


# ---------------------------
# MAIN PIPELINE
# ---------------------------
def main():
    print("\n🚀 Starting EGFR AI Drug Discovery Pipeline\n")
    
    # Load config
    config = load_config()
    print("Config:", config)
    
    # Load data
    smiles = load_data()
    
    # Convert to SELFIES (robust encoding)
    selfies = smiles_to_selfies(smiles)
    print("\nSample SELFIES:")
    print(selfies[:5])
    
    # Back to SMILES (sanity check)
    recon_smiles = selfies_to_smiles(selfies)
    
    # Generate new molecules (placeholder for AI models)
    print("\nGenerating new molecules...")
    generated = generate_variants(recon_smiles, n=5)
    
    # Combine original + generated
    all_molecules = list(set(smiles + generated))
    print(f"Total molecules after generation: {len(all_molecules)}")
    
    # Filter valid molecules
    valid_molecules = filter_valid(all_molecules)
    
    # Score molecules
    print("\nScoring molecules...")
    scored = score_molecules(valid_molecules)
    
    print("\nTop 10 molecules:")
    for smi, score in scored[:10]:
        print(f"{smi}  → Score: {score}")
    
    # RL optimization (selection-based)
    print("\nRunning RL optimization...")
    optimized = optimize(valid_molecules)
    
    print("\nTop optimized molecules:")
    for smi, score in optimized[:10]:
        print(f"{smi}  → Score: {score}")
    
    # Docking step
    docking_pipeline(optimized)
    
    print("\n✅ Pipeline completed successfully!\n")


# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    main()
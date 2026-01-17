import os
import subprocess
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

##############################################
# CONFIG (matches your manuscript)
##############################################

PDB_ID = "1AQ1"
RECEPTOR = "cdk2_clean.pdb"
GNINA = "gnina"   # must be in PATH
EXHAUSTIVENESS = "16"
NUM_MODES = "10"

##############################################
# Prepare CDK2 receptor
##############################################

def prepare_receptor():
    if not os.path.exists(RECEPTOR):
        print("Preparing CDK2 receptor from PDB 1AQ1")
        os.system(f"wget https://files.rcsb.org/download/{PDB_ID}.pdb -O {PDB_ID}.pdb")
        os.system(f"grep -v HETATM {PDB_ID}.pdb > {RECEPTOR}")

##############################################
# Convert SMILES to 3D PDBQT
##############################################

def smiles_to_pdbqt(smiles, outname):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    Chem.MolToMolFile(mol, f"{outname}.mol")

    os.system(f"obabel {outname}.mol -O {outname}.pdbqt")

##############################################
# Run GNINA Docking
##############################################

def dock_ligand(ligand_name):
    cmd = [
        GNINA,
        "-r", RECEPTOR,
        "-l", f"{ligand_name}.pdbqt",
        "--autobox_ligand", f"{PDB_ID}.pdb",
        "--num_modes", NUM_MODES,
        "--exhaustiveness", EXHAUSTIVENESS,
        "--score_only"
    ]

    result = subprocess.check_output(cmd).decode()

    # Extract CNN affinity
    for line in result.split("\n"):
        if "CNNscore" in line:
            return float(line.split()[-1])

    return None

##############################################
# Screen AI-generated molecules
##############################################

def screen_library(smiles_csv):
    prepare_receptor()

    df = pd.read_csv(smiles_csv)   # columns: Name, SMILES
    results = []

    for i, row in df.iterrows():
        name = row["Name"]
        smi = row["SMILES"]

        print(f"Docking {name}")
        smiles_to_pdbqt(smi, name)

        score = dock_ligand(name)

        results.append([name, smi, score])

    out = pd.DataFrame(results, columns=["Name", "SMILES", "CNN_Affinity"])
    out = out.sort_values("CNN_Affinity", ascending=False)
    out.to_csv("../results/gnina_ranked_hits.csv", index=False)

    print("Top 10 hits:")
    print(out.head(10))


##############################################
# Main
##############################################

if __name__ == "__main__":
    screen_library("../results/ai_generated_hits.csv")

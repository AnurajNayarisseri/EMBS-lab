import os
import subprocess
import pandas as pd

############################################
# Kinase structures (KLIFS aligned)
############################################

KINASES = {
    "CDK2": "1AQ1.pdb",
    "CDK1": "6GU2.pdb",
    "CDK4": "2W96.pdb",
    "CDK7": "1UA2.pdb",
    "CDK9": "4BCF.pdb"
}

LIGAND = "../docking/cdkcrc_embs.pdbqt"

############################################
# Clean receptors
############################################

def clean_pdb(pdb):
    out = pdb.replace(".pdb", "_clean.pdb")
    os.system(f"grep -v HETATM {pdb} > {out}")
    return out

############################################
# GNINA docking
############################################

def dock(receptor):
    cmd = [
        "gnina",
        "-r", receptor,
        "-l", LIGAND,
        "--autobox_ligand", receptor,
        "--score_only"
    ]
    result = subprocess.check_output(cmd).decode()
    for line in result.split("\n"):
        if "CNNscore" in line:
            return float(line.split()[-1])
    return None

############################################
# Run KinomeScan-like profiling
############################################

results = []

for kinase, pdb in KINASES.items():
    print("Docking to", kinase)
    clean = clean_pdb(pdb)
    score = dock(clean)
    results.append([kinase, score])

df = pd.DataFrame(results, columns=["Kinase","CNN_Affinity"])
df["Relative_to_CDK2"] = df["CNN_Affinity"] / df[df.Kinase=="CDK2"]["CNN_Affinity"].values[0]

df.to_csv("selectivity_scores.csv", index=False)
print(df)

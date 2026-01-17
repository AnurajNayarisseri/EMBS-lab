import os
import subprocess
import pandas as pd

KINASES = {
    "CDK2": "1AQ1.pdb",
    "CDK1": "6GU2.pdb",
    "CDK4": "2W96.pdb",
    "CDK7": "1UA2.pdb",
    "CDK9": "4BCF.pdb"
}

LIGAND = "../docking/cdkcrc_embs.pdbqt"
GNINA = "gnina"

def clean(pdb):
    clean_name = pdb.replace(".pdb","_clean.pdb")
    os.system(f"grep -v HETATM {pdb} > {clean_name}")
    return clean_name

def dock(receptor):
    cmd = [
        GNINA,
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

results = []

for kinase, pdb in KINASES.items():
    print("Docking", kinase)
    rec = clean(pdb)
    score = dock(rec)
    results.append([kinase, score])

df = pd.DataFrame(results, columns=["Kinase","CNN_Affinity"])
cdk2_score = df[df.Kinase=="CDK2"]["CNN_Affinity"].values[0]
df["Relative_to_CDK2"] = df["CNN_Affinity"] / cdk2_score

df.to_csv("selectivity_scores.csv", index=False)
print(df)

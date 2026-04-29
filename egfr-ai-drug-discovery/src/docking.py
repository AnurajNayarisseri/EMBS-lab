import os
import subprocess
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem


# -----------------------------
# Convert SMILES → 3D structure
# -----------------------------
def smiles_to_3d(smiles, out_sdf):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)

        Chem.MolToMolFile(mol, out_sdf)
        return True
    except:
        return False


# -----------------------------
# Convert SDF → PDBQT (OpenBabel)
# -----------------------------
def sdf_to_pdbqt(sdf_file, pdbqt_file):
    cmd = f"obabel {sdf_file} -O {pdbqt_file} --gen3d"
    result = os.system(cmd)
    return result == 0


# -----------------------------
# Run Gnina docking
# -----------------------------
def run_gnina(receptor_pdbqt, ligand_pdbqt, out_file):
    cmd = [
        "gnina",
        "-r", receptor_pdbqt,
        "-l", ligand_pdbqt,
        "--autobox_ligand", ligand_pdbqt,
        "-o", out_file
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        print("Gnina failed:", e)
        return None


# -----------------------------
# Extract docking score
# -----------------------------
def parse_gnina_output(output_text):
    """
    Extract binding affinity from Gnina output
    """
    if output_text is None:
        return None

    for line in output_text.split("\n"):
        if "Affinity:" in line:
            try:
                return float(line.split()[1])
            except:
                continue
    return None


# -----------------------------
# Full Docking Pipeline
# -----------------------------
def dock_smiles(smiles, receptor_pdbqt):
    with tempfile.TemporaryDirectory() as tmpdir:
        sdf_file = os.path.join(tmpdir, "ligand.sdf")
        pdbqt_file = os.path.join(tmpdir, "ligand.pdbqt")
        out_file = os.path.join(tmpdir, "out.sdf")

        # Step 1: SMILES → 3D
        success = smiles_to_3d(smiles, sdf_file)
        if not success:
            return None

        # Step 2: SDF → PDBQT
        success = sdf_to_pdbqt(sdf_file, pdbqt_file)
        if not success:
            return None

        # Step 3: Run docking
        output = run_gnina(receptor_pdbqt, pdbqt_file, out_file)

        # Step 4: Parse score
        score = parse_gnina_output(output)

        return score


# -----------------------------
# Batch Docking
# -----------------------------
def batch_docking(smiles_list, receptor_pdbqt):
    results = []

    for smi in smiles_list:
        print(f"Docking: {smi}")
        score = dock_smiles(smi, receptor_pdbqt)

        if score is not None:
            results.append((smi, score))

    # Sort by best affinity (more negative = better)
    results = sorted(results, key=lambda x: x[1])

    return results


# -----------------------------
# Normalize docking score
# -----------------------------
def normalize_score(score):
    """
    Convert docking score to reward-friendly value
    (lower energy → higher reward)
    """
    if score is None:
        return 0

    return -score / 10  # simple normalization


# -----------------------------
# Debug Run
# -----------------------------
if __name__ == "__main__":
    test_smiles = [
        "CCO",
        "CCN",
        "CCC"
    ]

    receptor = "egfr.pdbqt"  # YOU MUST PROVIDE THIS

    print("\nRunning docking...\n")

    results = batch_docking(test_smiles, receptor)

    print("\nResults:")
    for smi, score in results:
        print(f"{smi} → {score}")
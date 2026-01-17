import pandas as pd
import re

############################################################
# Molegro Virtual Docker (MVD) Result Parser
############################################################
# Extracts:
#  - MolDock score
#  - ReRank score
#  - H-bond score
#  - Ligand name
############################################################

def parse_mvd(filename):
    """
    Parses Molegro Virtual Docker text export (.txt)
    """
    results = []

    with open(filename, "r") as f:
        block = {}
        for line in f:
            line = line.strip()

            if line.startswith("Ligand"):
                if block:
                    results.append(block)
                    block = {}
                block["Ligand"] = line.split(":")[1].strip()

            if "MolDock Score" in line:
                block["MolDock"] = float(line.split(":")[1])

            if "ReRank Score" in line:
                block["ReRank"] = float(line.split(":")[1])

            if "H-Bond" in line:
                block["Hbond"] = float(line.split(":")[1])

        if block:
            results.append(block)

    return pd.DataFrame(results)


############################################################
# Combine MVD + GNINA scores
############################################################

def merge_scores(mvd_file, gnina_file, output="final_ranked_hits.csv"):

    mvd = parse_mvd(mvd_file)
    gnina = pd.read_csv(gnina_file)   # Name, SMILES, CNN_Affinity

    df = pd.merge(gnina, mvd, left_on="Name", right_on="Ligand")

    # Multi-tier scoring
    df["FinalScore"] = (
        -df["CNN_Affinity"] * 0.5 +
        -df["ReRank"] * 0.3 +
        -df["MolDock"] * 0.2
    )

    df = df.sort_values("FinalScore", ascending=False)
    df.to_csv(output, index=False)

    print("Top ranked compounds:")
    print(df.head(10))


############################################################
# Main
############################################################

if __name__ == "__main__":
    merge_scores("mvd_results.txt", "../results/gnina_ranked_hits.csv")

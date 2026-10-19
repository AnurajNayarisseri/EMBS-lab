#!/usr/bin/env python3
"""Reproducible scaffold-split audit for the CDK2 project.

This script does NOT claim to train or validate a CDK2 activity predictor. It:
1) standardizes SMILES with RDKit;
2) groups molecules by Bemis-Murcko scaffold;
3) creates a leakage-resistant train/validation/test split;
4) verifies scaffold separation and writes auditable CSV/JSON outputs.

Input CSV must contain a SMILES column. Optional ID columns are retained.
"""
import argparse, json, random, sys
from pathlib import Path
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold


def canonicalize(s):
    if not isinstance(s, str) or not s.strip(): return None
    mol = Chem.MolFromSmiles(s.strip())
    if mol is None: return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def scaffold_for(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    # Acyclic compounds have an empty Murcko scaffold; keep them in one explicit group
    # rather than accidentally treating every acyclic molecule as a different scaffold.
    return Chem.MolToSmiles(scaffold, canonical=True) if scaffold.GetNumAtoms() else "__ACYCLIC__"


def scaffold_split(df, fractions=(0.8, 0.1, 0.1), seed=2026):
    groups = {}
    for idx, scaf in zip(df.index, df["scaffold"]): groups.setdefault(scaf, []).append(idx)
    items = list(groups.items())
    random.Random(seed).shuffle(items)
    # Assign whole scaffold groups; approximate requested molecule proportions without splitting a scaffold.
    targets = [fractions[0]*len(df), (fractions[0]+fractions[1])*len(df)]
    buckets, counts = [[], [], []], [0, 0, 0]
    for scaf, indices in sorted(items, key=lambda x: len(x[1]), reverse=True):
        # choose the bucket that best fills the next target while preserving order
        if counts[0] < targets[0]: b = 0
        elif counts[1] < targets[1]: b = 1
        else: b = 2
        buckets[b].extend(indices); counts[b] += len(indices)
    out = df.copy()
    label = {}
    for name, inds in zip(["train", "valid", "test"], buckets):
        for i in inds: label[i] = name
    out["split"] = out.index.map(label)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV with SMILES column")
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--id-col", default=None)
    p.add_argument("--outdir", default="results/scaffold_validation")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    inp = Path(args.input); outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(inp)
    if args.smiles_col not in raw.columns:
        raise SystemExit(f"ERROR: SMILES column '{args.smiles_col}' not found. Columns: {list(raw.columns)}")
    raw["canonical_smiles"] = raw[args.smiles_col].map(canonicalize)
    invalid = raw[raw["canonical_smiles"].isna()].copy()
    valid = raw[raw["canonical_smiles"].notna()].copy()
    valid = valid.drop_duplicates("canonical_smiles", keep="first").reset_index(drop=True)
    valid["scaffold"] = valid["canonical_smiles"].map(scaffold_for)
    valid = scaffold_split(valid, seed=args.seed)

    sets = {n: set(valid.loc[valid.split == n, "scaffold"]) for n in ["train", "valid", "test"]}
    overlaps = {"train_valid": sorted(sets['train'] & sets['valid']),
                "train_test": sorted(sets['train'] & sets['test']),
                "valid_test": sorted(sets['valid'] & sets['test'])}
    no_leakage = all(len(v) == 0 for v in overlaps.values())
    for split in ["train", "valid", "test"]:
        valid[valid.split == split].to_csv(outdir / f"{split}.csv", index=False)
    invalid.to_csv(outdir / "invalid_smiles.csv", index=False)
    valid.to_csv(outdir / "all_standardized_with_splits.csv", index=False)
    report = {
        "input_file": str(inp), "input_rows": int(len(raw)),
        "invalid_smiles_rows": int(len(invalid)),
        "valid_rows_before_deduplication": int(raw["canonical_smiles"].notna().sum()),
        "unique_valid_molecules": int(len(valid)),
        "unique_scaffolds": int(valid.scaffold.nunique()),
        "split_molecule_counts": {n: int((valid.split == n).sum()) for n in ["train", "valid", "test"]},
        "split_scaffold_counts": {n: int(valid.loc[valid.split == n, "scaffold"].nunique()) for n in ["train", "valid", "test"]},
        "scaffold_overlap_counts": {k: len(v) for k, v in overlaps.items()},
        "scaffold_leakage_check_passed": no_leakage,
        "seed": args.seed, "rdkit_version": rdBase.rdkitVersion,
        "note": "This is a scaffold-split integrity audit, not evidence of CDK2 activity prediction or generative-model superiority. External activity validation requires independent, experimentally labelled compounds."
    }
    (outdir / "scaffold_validation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not no_leakage:
        print("ERROR: scaffold overlap detected; do not use this split for claims of scaffold generalization.", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__": main()

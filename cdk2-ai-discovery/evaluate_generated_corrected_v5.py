#!/usr/bin/env python3
"""Canonicalized structural-novelty evaluation for generated SMILES.

Novelty is relative only to the supplied training CSV. It does not establish
biological activity, selectivity, or synthesizability.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def canonicalize(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return FPGEN.GetFingerprint(mol) if mol is not None else None


def truthy(value):
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated SMILES novelty against a training set."
    )
    parser.add_argument("--generated", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-generated", type=int, default=2000)
    args = parser.parse_args()

    generated_path = Path(args.generated)
    train_path = Path(args.train)
    out_path = Path(args.out)
    if not generated_path.is_file():
        raise FileNotFoundError(f"Generated CSV not found: {generated_path}")
    if not train_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")

    generated = pd.read_csv(generated_path)
    train = pd.read_csv(train_path)
    if "smiles" not in train.columns:
        raise ValueError("Training CSV must contain a 'smiles' column.")
    gen_col = "canonical_smiles" if "canonical_smiles" in generated.columns else (
        "smiles" if "smiles" in generated.columns else None
    )
    if gen_col is None:
        raise ValueError("Generated CSV must contain 'canonical_smiles' or 'smiles'.")

    train_canonical = []
    invalid_train = 0
    for smi in train["smiles"]:
        can = canonicalize(smi)
        if can is None:
            invalid_train += 1
        else:
            train_canonical.append(can)
    train_unique = sorted(set(train_canonical))
    if not train_unique:
        raise ValueError("No valid training molecules were found.")
    train_set = set(train_unique)
    train_fps = [fingerprint(s) for s in train_unique]
    train_fps = [fp for fp in train_fps if fp is not None]

    valid_records = []
    invalid_generated = 0
    for _, row in generated.iterrows():
        can = canonicalize(row.get(gen_col))
        if can is None:
            invalid_generated += 1
            continue
        if "valid" in generated.columns and not truthy(row["valid"]):
            continue
        record = row.to_dict()
        record["canonical_smiles"] = can
        valid_records.append(record)

    evaluated = valid_records[:max(args.max_generated, 0)]
    rows = []
    for record in evaluated:
        can = record["canonical_smiles"]
        fp = fingerprint(can)
        similarities = (
            [DataStructs.TanimotoSimilarity(fp, train_fp) for train_fp in train_fps]
            if fp is not None else []
        )
        max_similarity = max(similarities) if similarities else None
        record["exact_novel_vs_train"] = can not in train_set
        record["max_train_tanimoto"] = max_similarity
        record["novel_lt_0_4_tanimoto"] = (
            can not in train_set and max_similarity is not None and max_similarity < 0.4
        )
        rows.append(record)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    unique_generated = sorted({record["canonical_smiles"] for record in rows})
    unique_novel = [smiles for smiles in unique_generated if smiles not in train_set]
    unique_low_similarity = 0
    for smiles in unique_generated:
        fp = fingerprint(smiles)
        sims = (
            [DataStructs.TanimotoSimilarity(fp, train_fp) for train_fp in train_fps]
            if fp is not None else []
        )
        if smiles not in train_set and sims and max(sims) < 0.4:
            unique_low_similarity += 1

    n_rows = len(rows)
    summary = {
        "generated_input_rows": int(len(generated)),
        "valid_generated_rows_evaluated": int(n_rows),
        "unique_valid_generated_structures_evaluated": int(len(unique_generated)),
        "invalid_generated_rows_detected": int(invalid_generated),
        "training_input_rows": int(len(train)),
        "invalid_training_rows": int(invalid_train),
        "unique_valid_training_structures": int(len(train_unique)),
        "exact_novel_count_rows": int(sum(bool(row["exact_novel_vs_train"]) for row in rows)),
        "exact_novel_pct_rows": round(100 * sum(bool(row["exact_novel_vs_train"]) for row in rows) / n_rows, 2) if n_rows else None,
        "unique_exact_novel_count": int(len(unique_novel)),
        "unique_exact_novel_pct": round(100 * len(unique_novel) / len(unique_generated), 2) if unique_generated else None,
        "novel_lt_0_4_tanimoto_count_rows": int(sum(bool(row["novel_lt_0_4_tanimoto"]) for row in rows)),
        "novel_lt_0_4_tanimoto_pct_rows": round(100 * sum(bool(row["novel_lt_0_4_tanimoto"]) for row in rows) / n_rows, 2) if n_rows else None,
        "unique_novel_lt_0_4_tanimoto_count": int(unique_low_similarity),
        "rdkit_version": rdBase.rdkitVersion,
        "fingerprint": "Morgan radius 2, 2048 bits; Tanimoto similarity",
        "warning": "Novelty is relative only to the supplied training CSV and does not establish biological activity, selectivity, synthesizability, or generalizability."
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Molecule-level results: {out_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()

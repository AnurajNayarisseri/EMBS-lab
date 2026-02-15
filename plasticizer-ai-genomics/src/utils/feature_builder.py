"""
feature_builder.py

Utility functions for feature engineering used in the plasticizer hydrolase ML pipeline.

Feature order (MUST stay consistent across training + scoring):
  1) length (1)
  2) amino acid composition (20) in AA order: ACDEFGHIKLMNPQRSTVWY
  3) motif features (1): gxsxg-like motif (as implemented in your train_model.py)
  4) Pfam domain binary vector (len(domain_vocab))

Inputs supported:
  - FASTA protein sequences (.faa/.fasta)
  - Pfam domtblout output (hmmscan --domtblout ...)

Notes:
  - parse_pfam() uses an E-value cutoff (default 1e-5)
  - build_domain_vocab() selects top_n most frequent domains across pos+neg domtblout files
"""

from __future__ import annotations

import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Set, Optional

import numpy as np


# ------------------------
# Amino Acid alphabet
# ------------------------
AA: str = "ACDEFGHIKLMNPQRSTVWY"
AA_SET: Set[str] = set(AA)


# -------------------------
# FASTA reader
# -------------------------
def read_fasta(path: str | Path) -> Dict[str, str]:
    """
    Read a FASTA file and return a dict {protein_id: sequence}.

    Protein ID is the first token after '>' (same behavior as your train_model.py).
    """
    path = Path(path)
    seqs: Dict[str, List[str]] = {}
    current: Optional[str] = None

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            current = line[1:].split()[0]
            seqs[current] = []
        else:
            if current is None:
                raise ValueError(f"FASTA parse error: sequence line before header in {path}")
            seqs[current].append(line)

    return {k: "".join(v) for k, v in seqs.items()}


# ----------------------
# Amino acid composition
# ----------------------
def aac(seq: str) -> List[float]:
    """
    Amino acid composition (AAC) for the 20 standard amino acids in AA order.
    Returns a list of length 20.
    """
    seq = seq.upper()
    counts = Counter([x for x in seq if x in AA_SET])
    total = sum(counts.values()) or 1
    return [counts.get(a, 0) / total for a in AA]


# ----------------------------
# Motif features
# ----------------------------
def motif_features(seq: str) -> List[int]:
    """
    Motif features used in your model.

    This reproduces your existing implementation:
      gxsxg = 1 if re.search(r"G.A.G", seq) else 0

    Returns a list of length 1.
    """
    seq = seq.upper()
    gxsxg = 1 if re.search(r"G.A.G", seq) else 0
    return [gxsxg]


# -------------------------
# Parse Pfam domtblout
# -------------------------
def parse_pfam(domtbl_path: str | Path, evalue_cutoff: float = 1e-5) -> Dict[str, Set[str]]:
    """
    Parse a Pfam domtblout file and return mapping {protein_id: set(domains)}.

    This follows your train_model.py logic:
      - Skip comment lines starting with '#'
      - domain = parts[0]
      - protein = parts[3]
      - evalue = float(parts[6])
    """
    domtbl_path = Path(domtbl_path)
    prot2dom: Dict[str, Set[str]] = {}

    with domtbl_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 7:
                # Domtblout lines should be long; skip malformed lines safely
                continue

            domain = parts[0]
            protein = parts[3]

            try:
                evalue = float(parts[6])
            except ValueError:
                continue

            if evalue <= evalue_cutoff:
                prot2dom.setdefault(protein, set()).add(domain)

    return prot2dom


# -----------------------
# Build domain vocabulary
# -----------------------
def build_domain_vocab(
    pos_dom: str | Path,
    neg_dom: str | Path,
    top_n: int = 300,
    evalue_cutoff: float = 1e-5,
) -> List[str]:
    """
    Build domain vocabulary from positive and negative Pfam domtblout files.

    Returns top_n domains by frequency across (pos + neg).
    """
    pos_map = parse_pfam(pos_dom, evalue_cutoff=evalue_cutoff)
    neg_map = parse_pfam(neg_dom, evalue_cutoff=evalue_cutoff)

    counter = Counter()
    for doms in list(pos_map.values()) + list(neg_map.values()):
        counter.update(doms)

    vocab = [d for d, _ in counter.most_common(top_n)]
    return vocab


# ---------------------------
# Build feature matrix
# ---------------------------
def build_features(
    fasta: str | Path,
    domtbl: str | Path,
    domain_vocab: List[str],
    evalue_cutoff: float = 1e-5,
) -> Tuple[List[str], np.ndarray]:
    """
    Build feature matrix X and list of protein IDs.

    Features per protein (in order):
      1) length
      2) AAC (20)
      3) motif_gxsxg (1)
      4) Pfam domain vector (len(domain_vocab))

    Returns:
      ids: list[str]
      X: np.ndarray shape (n_proteins, n_features)
    """
    seqs = read_fasta(fasta)
    prot2dom = parse_pfam(domtbl, evalue_cutoff=evalue_cutoff)

    dom_index = {d: i for i, d in enumerate(domain_vocab)}

    X: List[List[float]] = []
    ids: List[str] = []

    for pid, seq in seqs.items():
        row: List[float] = []

        # 1) Length
        row.append(float(len(seq)))

        # 2) Amino acid composition
        row.extend(aac(seq))

        # 3) Motif
        row.extend(motif_features(seq))

        # 4) Domain binary vector
        dom_vector = [0.0] * len(domain_vocab)
        for d in prot2dom.get(pid, []):
            if d in dom_index:
                dom_vector[dom_index[d]] = 1.0
        row.extend(dom_vector)

        X.append(row)
        ids.append(pid)

    return ids, np.asarray(X, dtype=float)


# ---------------------------
# Feature name helper (optional)
# ---------------------------
def build_feature_names(domain_vocab: List[str]) -> List[str]:
    """
    Construct feature names in the same order as build_features().
    Useful for feature importance plots.
    """
    names = ["length"]
    names += [f"aac_{a}" for a in AA]
    names += ["motif_gxsxg"]
    names += [f"pfam_{d}" for d in domain_vocab]
    return names

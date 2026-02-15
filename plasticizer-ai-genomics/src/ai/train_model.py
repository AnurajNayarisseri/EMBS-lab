import re
import joblib
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    accuracy_score,
    f1_score
)
from sklearn.ensemble import RandomForestClassifier

# ------------------------
# Amino Acid alphabets
# ------------------------
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)

#-------------------------
# Fasta Header
#------------------------
def read_fasta(path):
    seqs = {}
    current = None
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            current = line[1:].split()[0]
            seqs[current] = []
        else:
            seqs[current].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}

#----------------------
# Amino Acid Composition
#----------------------
def aac(seq):
    seq = seq.upper()
    counts = Counter([x for x in seq if x in AA_SET])
    total = sum(counts.values()) or 1
    return [counts.get(a, 0) / total for a in AA]

#----------------------------
# Motif features
#----------------------------
def motif_features(seq):
    seq = seq.upper()
    gxsxg = 1 if re.search(r"G.A.G", seq) else 0
    return [gxsxg]

#-------------------------
# Parse Pfam domtblout
#-------------------------
def parse_pfam(domtbl_path, evalue_cutoff=1e-5):
    prot2dom = {}
    with open(domtbl_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            domain = parts[0]
            protein = parts[3]
            evalue = float(parts[6])
            if evalue <= evalue_cutoff:
                prot2dom.setdefault(protein, set()).add(domain)
    return prot2dom

#-----------------------
# Build domain vocabulary
#------------------------
def build_domain_vocab(pos_dom, neg_dom, top_n=300):
    pos_map = parse_pfam(pos_dom)
    neg_map = parse_pfam(neg_dom)

    counter = Counter()
    for doms in list(pos_map.values()) + list(neg_map.values()):
        counter.update(doms)

    vocab = [d for d, _ in counter.most_common(top_n)]
    return vocab

#---------------------------
# Build feature matrix
#----------------------------
def build_features(fasta, domtbl, domain_vocab):
    seqs = read_fasta(fasta)
    prot2dom = parse_pfam(domtbl)

    dom_index = {d: i for i, d in enumerate(domain_vocab)}

    X = []
    ids = []

    for pid, seq in seqs.items():
        row = []

        # Length
        row.append(len(seq))

        # Amino acid composition
        row.extend(aac(seq))

        # Motif
        row.extend(motif_features(seq))

        # Domain binary vector
        dom_vector = [0] * len(domain_vocab)
        for d in prot2dom.get(pid, []):
            if d in dom_index:
                dom_vector[dom_index[d]] = 1
        row.extend(dom_vector)

        X.append(row)
        ids.append(pid)

    return ids, np.array(X)

#--------------------------
# Main
#--------------------------
def main():

    POS_FASTA = "pos.faa"
    NEG_FASTA = "neg_clean.faa"
    POS_DOM = "pos.pfam.domtblout"
    NEG_DOM = "neg_clean.pfam.domtblout"

    print("Building domain vocabulary...")
    domain_vocab = build_domain_vocab(POS_DOM, NEG_DOM)

    print("Extracting features...")
    pos_ids, X_pos = build_features(POS_FASTA, POS_DOM, domain_vocab)
    neg_ids, X_neg = build_features(NEG_FASTA, NEG_DOM, domain_vocab)

    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*len(pos_ids) + [0]*len(neg_ids))

    print("Total samples:", len(y))
    print("Positive samples:", len(pos_ids))
    print("Negative samples:", len(neg_ids))

    #-----------------------------------
    # 80/20 Train -Test Split
    #----------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    rf.fit(X_train, y_train)

    probs = rf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print("\n=== 80/20 Split Performance ===")
    print("ROC-AUC:", roc_auc_score(y_test, probs))
    print("PR-AUC :", average_precision_score(y_test, probs))
    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1-score:", f1_score(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    #--------------------------
    # 5-fold Cross Validation
    #-------------------------
    print("\n=== 5-Fold Cross Validation ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    roc_scores = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    print("Mean ROC-AUC:", roc_scores.mean())
    print("Std ROC-AUC :", roc_scores.std())

    #---------------------
    # Save Model
    #---------------------
    joblib.dump({
        "model": rf,
        "domain_vocab": domain_vocab
    }, "rf_plasticizer_model.joblib")

    print("\nModel saved as rf_plasticizer_model.joblib")

if __name__ == "__main__":
    main()

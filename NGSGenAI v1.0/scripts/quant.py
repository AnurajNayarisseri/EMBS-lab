#!/usr/bin/env python3

import sys
import pysam
import pandas as pd
import numpy as np
import json
from collections import defaultdict

if len(sys.argv) != 4:
    print("Usage: quant.py aligned.bam annotation.gtf output_prefix")
    sys.exit(1)

bam_file = sys.argv[1]
gtf_file = sys.argv[2]
out = sys.argv[3]

print("[NGSGenAI] Loading annotation...")

genes = {}
lengths = {}

with open(gtf_file) as f:
    for line in f:
        if line.startswith("#"): continue
        fields = line.strip().split("\t")
        if fields[2] == "exon":
            info = fields[8]
            gene = info.split("gene_id")[1].split('"')[1]
            length = int(fields[4]) - int(fields[3]) + 1
            lengths[gene] = lengths.get(gene, 0) + length

print("[NGSGenAI] Processing alignments...")

bam = pysam.AlignmentFile(bam_file, "rb")

counts = defaultdict(int)
total_reads = 0

for read in bam.fetch():
    if not read.is_unmapped:
        gene = read.get_tag("XT") if read.has_tag("XT") else None
        if gene:
            counts[gene] += 1
            total_reads += 1

bam.close()

print("[NGSGenAI] Calculating TPM...")

tpm = {}
for gene in counts:
    rpk = counts[gene] / (lengths.get(gene,1)/1000)
    tpm[gene] = rpk

scale = sum(tpm.values()) / 1e6
for gene in tpm:
    tpm[gene] /= scale

# Bayesian posterior variance (Poisson approx)
variance = {}
for gene in counts:
    variance[gene] = 1.0 / (counts[gene] + 1)

df = pd.DataFrame({
    "Gene": list(tpm.keys()),
    "TPM": list(tpm.values()),
    "PosteriorVariance": [variance[g] for g in tpm]
})

df.to_csv(out + "_tpm.tsv", sep="\t", index=False)

meta = {
    "module": "NGSGenAI Quantification",
    "bam": bam_file,
    "gtf": gtf_file,
    "total_reads": total_reads
}

with open(out + "_meta.json", "w") as f:
    json.dump(meta, f, indent=4)

print("[NGSGenAI] TPM written to", out + "_tpm.tsv")

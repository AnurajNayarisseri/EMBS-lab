#!/usr/bin/env python3

import sys
import pysam
import pandas as pd
from collections import defaultdict

if len(sys.argv) != 3:
    print("Usage: fusion.py aligned.bam fusion_results.tsv")
    sys.exit(1)

bam_file = sys.argv[1]
out_file = sys.argv[2]

print("[NGSGenAI] Loading BAM...")
bam = pysam.AlignmentFile(bam_file, "rb")

# Track split reads
split_reads = defaultdict(list)
discordant = defaultdict(list)

for read in bam.fetch():
    if read.is_unmapped:
        continue

    # Split reads (supplementary alignments)
    if read.has_tag("SA"):
        chrom1 = read.reference_name
        chrom2 = read.get_tag("SA").split(",")[0]
        key = chrom1 + "--" + chrom2
        split_reads[key].append(read.query_name)

    # Discordant paired-end
    if read.is_paired and not read.is_proper_pair:
        if read.reference_name != read.next_reference_name:
            key = read.reference_name + "--" + read.next_reference_name
            discordant[key].append(read.query_name)

bam.close()

fusions = []

for key in set(split_reads.keys()).union(discordant.keys()):
    sr = len(set(split_reads[key]))
    dr = len(set(discordant[key]))
    score = sr + dr

    if score >= 5:  # confidence threshold
        fusions.append((key.split("--")[0], key.split("--")[1], sr, dr, score))

df = pd.DataFrame(fusions, columns=["GeneA","GeneB","SplitReads","DiscordantPairs","Score"])
df = df.sort_values("Score", ascending=False)

df.to_csv(out_file, sep="\t", index=False)

print("[NGSGenAI] Fusion results written to", out_file)

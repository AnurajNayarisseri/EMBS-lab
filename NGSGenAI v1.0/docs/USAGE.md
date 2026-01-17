# Using NGSGenAI

This document describes how to run **NGSGenAI** for end-to-end RNA-seq analysis, including alignment, quantification, isoform and fusion detection, differential expression, and visualization.

---

## Basic Command

NGSGenAI is executed through its command-line interface:

```bash
python ngsgenai.py run \
  --r1 sample_R1.fastq \
  --r2 sample_R2.fastq \
  --ref genome.fa \
  --gtf genes.gtf \
  --out results/

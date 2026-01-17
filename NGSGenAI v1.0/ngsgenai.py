#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import shutil
from datetime import datetime

VERSION = "NGSGenAI v1.0"

# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def banner():
    print("=" * 70)
    print("NGSGenAI — Generative AI RNA-Seq Analysis Platform")
    print("Version:", VERSION)
    print("Eminent Biosciences")
    print("=" * 70)

def die(msg):
    print("[ERROR]", msg)
    sys.exit(1)

def check_program(name):
    if shutil.which(name) is None:
        die(f"Required program not found: {name}")

def check_dependencies():
    print("[INFO] Checking system dependencies...")
    for p in ["nextflow", "docker"]:
        check_program(p)
    print("[OK] All dependencies satisfied.")

# --------------------------------------------------
# Run pipeline
# --------------------------------------------------

def run_pipeline(args):

    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)

    logfile = os.path.join(outdir, "ngsgenai.log")

    with open(logfile, "w") as f:
        f.write("NGSGenAI run started: " + str(datetime.now()) + "\n")

    cmd = [
        "nextflow", "run", args.pipeline,
        "--r1", os.path.abspath(args.r1),
        "--r2", os.path.abspath(args.r2),
        "--ref", os.path.abspath(args.ref),
        "--gtf", os.path.abspath(args.gtf),
        "--out", outdir
    ]

    if args.cancer:
        cmd.append("--cancer")

    print("\n[INFO] Launching NGSGenAI pipeline:")
    print(" ".join(cmd))

    with open(logfile, "a") as log:
        result = subprocess.run(cmd, stdout=log, stderr=log)

    if result.returncode != 0:
        die(f"Pipeline failed. See log file: {logfile}")

    print("\n[OK] NGSGenAI analysis completed successfully.")
    print(f"[INFO] Results directory: {outdir}")
    print(f"[INFO] Log file: {logfile}")

# --------------------------------------------------
# Info
# --------------------------------------------------

def show_info():
    print("""
NGSGenAI
========
End-to-end generative AI RNA-seq analysis system.

Modules
-------
• Quality control and preprocessing
• Learned alignment and indexing
• Bayesian transcript quantification
• Isoform reconstruction
• Fusion gene detection
• Differential expression
• Visualization and reporting

Citation
--------
Nayarisseri A. et al.
NGSGenAI: A Generative AI Pipeline for RNA-Seq Transcriptome Analysis
""")

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    banner()

    parser = argparse.ArgumentParser(
        description="NGSGenAI — Generative AI RNA-Seq analysis"
    )

    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run RNA-seq analysis")
    run.add_argument("--r1", required=True, help="Read 1 FASTQ")
    run.add_argument("--r2", required=True, help="Read 2 FASTQ")
    run.add_argument("--ref", required=True, help="Reference genome (FASTA)")
    run.add_argument("--gtf", required=True, help="Annotation (GTF)")
    run.add_argument("--out", required=True, help="Output directory")
    run.add_argument("--pipeline", default="main.nf", help="Nextflow pipeline")
    run.add_argument("--cancer", action="store_true", help="Enable cancer mode")

    info = sub.add_parser("info", help="Show system information")

    args = parser.parse_args()

    if args.command == "run":
        check_dependencies()
        run_pipeline(args)
    elif args.command == "info":
        show_info()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

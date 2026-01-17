#!/usr/bin/env python3

import sys
import subprocess
import json
import os
from datetime import datetime

if len(sys.argv) != 5:
    print("Usage: preprocess.py R1.fastq R2.fastq clean_R1.fastq clean_R2.fastq")
    sys.exit(1)

r1, r2, out1, out2 = sys.argv[1:]

report_dir = "qc_reports"
os.makedirs(report_dir, exist_ok=True)

json_report = os.path.join(report_dir, "fastp.json")
html_report = os.path.join(report_dir, "fastp.html")

print("[NGSGenAI] Starting generative preprocessing...")
print("[NGSGenAI] Input:", r1, r2)

cmd = [
    "fastp",
    "-i", r1,
    "-I", r2,
    "-o", out1,
    "-O", out2,
    "--detect_adapter_for_pe",
    "--cut_front",
    "--cut_tail",
    "--cut_window_size", "4",
    "--cut_mean_quality", "20",
    "--length_required", "30",
    "--thread", "4",
    "--json", json_report,
    "--html", html_report
]

subprocess.run(cmd, check=True)

# Save pipeline metadata
meta = {
    "module": "NGSGenAI Preprocessing",
    "date": str(datetime.now()),
    "input_R1": r1,
    "input_R2": r2,
    "output_R1": out1,
    "output_R2": out2,
    "qc_json": json_report,
    "qc_html": html_report
}

with open("preprocess_metadata.json", "w") as f:
    json.dump(meta, f, indent=4)

print("[NGSGenAI] Preprocessing completed successfully.")
print("[NGSGenAI] Clean reads:", out1, out2)
print("[NGSGenAI] QC reports:", json_report, html_report)

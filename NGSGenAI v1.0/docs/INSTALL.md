# NGSGenAI Installation Guide

This document describes how to install and run **NGSGenAI**, a generative AI-powered RNA-seq analysis platform.

NGSGenAI is distributed as a Dockerized Nextflow pipeline, ensuring reproducible execution across Linux, macOS, and cloud or HPC systems.

---

## System Requirements

Minimum recommended specifications:

- Linux or macOS
- Docker (version ≥ 20.10)
- Nextflow (version ≥ 22.10)
- Python (version ≥ 3.8)
- 16 GB RAM (32 GB recommended)
- 8 CPU cores (GPU optional)

---

## Step 1 — Install Docker

Follow the official Docker installation guide:

https://docs.docker.com/get-docker/

After installation, verify:

```bash
docker --version

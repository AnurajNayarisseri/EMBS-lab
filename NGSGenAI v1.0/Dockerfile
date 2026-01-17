FROM continuumio/miniconda3:latest

LABEL maintainer="Dr. Anuraj Nayarisseri, Eminent Biosciences"
LABEL description="NGSGenAI – Generative AI RNA-Seq Analysis Platform"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    curl \
    openjdk-11-jre \
    && rm -rf /var/lib/apt/lists/*

# Install bioinformatics tools
RUN conda install -y -c bioconda -c conda-forge \
    star \
    fastp \
    subread \
    samtools \
    pysam \
    nextflow \
    && conda clean -a

# Install Python scientific stack
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scipy \
    matplotlib \
    seaborn \
    statsmodels

# Create app directory
WORKDIR /app

# Copy NGSGenAI code
COPY . /app

# Make scripts executable
RUN chmod +x ngsgenai.py && chmod +x scripts/*.py

# Set default command
ENTRYPOINT ["python", "ngsgenai.py"]

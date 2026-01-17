nextflow.enable.dsl = 2

// -------------------------------
// Parameters
// -------------------------------
params.r1      = null
params.r2      = null
params.ref     = null
params.gtf     = null
params.out     = "results"
params.cancer  = false

// -------------------------------
// Workflow
// -------------------------------
workflow {

    if( !params.r1 || !params.r2 || !params.ref || !params.gtf ) {
        error "Missing required inputs. Use --r1 --r2 --ref --gtf"
    }

    Channel.fromPath(params.r1).set { r1_ch }
    Channel.fromPath(params.r2).set { r2_ch }
    Channel.fromPath(params.ref).set { ref_ch }
    Channel.fromPath(params.gtf).set { gtf_ch }

    PREPROCESS(r1_ch, r2_ch)
    ALIGN(PREPROCESS.out1, PREPROCESS.out2, ref_ch)
    QUANT(ALIGN.out, gtf_ch)
    FUSION(ALIGN.out)
    DE(QUANT.out)
    PLOT(DE.out)
}

// -------------------------------
// PREPROCESSING
// -------------------------------
process PREPROCESS {
    tag "Preprocess"

    input:
    path r1
    path r2

    output:
    path "clean_R1.fastq", emit: out1
    path "clean_R2.fastq", emit: out2

    script:
    """
    python scripts/preprocess.py $r1 $r2 clean_R1.fastq clean_R2.fastq
    """
}

// -------------------------------
// ALIGNMENT
// -------------------------------
process ALIGN {
    tag "Align"

    input:
    path r1
    path r2
    path ref

    output:
    path "aligned.bam", emit: out

    script:
    """
    STAR --runThreadN 8 \
         --genomeDir genome_index \
         --readFilesIn $r1 $r2 \
         --outSAMtype BAM SortedByCoordinate

    mv Aligned.sortedByCoord.out.bam aligned.bam
    """
}

// -------------------------------
// QUANTIFICATION
// -------------------------------
process QUANT {
    tag "Quant"

    input:
    path bam
    path gtf

    output:
    path "expression_tpm.tsv", emit: out

    script:
    """
    python scripts/quant.py $bam $gtf expression
    mv expression_tpm.tsv expression_tpm.tsv
    """
}

// -------------------------------
// FUSION DETECTION
// -------------------------------
process FUSION {
    tag "Fusion"

    input:
    path bam

    output:
    path "fusion_results.tsv"

    script:
    """
    python scripts/fusion.py $bam fusion_results.tsv
    """
}

// -------------------------------
// DIFFERENTIAL EXPRESSION
// -------------------------------
process DE {
    tag "DE"

    input:
    path expr

    output:
    path "de_results.tsv", emit: out

    script:
    """
    python scripts/de.py $expr de_results.tsv
    """
}

// -------------------------------
// PLOTTING
// -------------------------------
process PLOT {
    tag "Plots"

    input:
    path de

    output:
    path "volcano.png"
    path "heatmap.png"

    script:
    """
    python scripts/plots.py $de results
    """
}

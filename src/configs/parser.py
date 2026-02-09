def get_parser_args():
    parser_args = {
        "model": "standard",
        "fasta_file": "/path/assembly.fasta.gz",
        "bigwig_file": "/path/atac_peaks.bw",
        "positives_bed": "/path/positives_2000.bed",
        "negatives_bed": "/path/negatives_2000.bed",
        "split": "/path/split_0.json",
        "checkpoint_path": "/checkpoints/experiment_name",
        "pretrained": None, #"/checkpoints/pretrained.ckpt",
        "methylation_bigwig": "/path/methylation.bw"
    }

    return parser_args

def get_parser_args_gene():
    parser_args = {
        "fasta_file": "/path/assembly_with_chromatin.fasta.gz", #"/data/assembly.fasta.gz"
        "chromatin_tokens_used": True,
        "positives_bed": "/path/gene_positive_100.bed",
        "negatives_bed": "/path/gene_negative_100.bed",
        "split": "/path/split_0.json",
        "checkpoint_path": "/checkpoints/experiment_name",
        "pretrained": None,
        "methylation_bigwig": "/path/methylation.bw"
    }

    return parser_args

def get_parser_base():
    parser_args = {
        "model": "standard",
        "fasta_file": "/path/assembly.fasta.gz",
        "intervals_bed": "/path/random_seqs_2000.bed",
        "checkpoint_path": "/checkpoints/experiment_name",
        "methylation_bigwig": "/path/methylation.bw"
    }

    return parser_args
import gzip
import random
from Bio import SeqIO

fasta_file = "/path/assembly.fasta.gz"
output_file = "/path/random_seqs_10000.bed"

seq_length = 10000
num_seqs_per_chrom = 35000

with open(output_file, "w") as out_handle:
    with gzip.open(fasta_file, "rt") as in_handle:
        for record in SeqIO.parse(in_handle, "fasta"):
            chrom = record.id
            chrom_len = len(record.seq)
            
            if chrom_len < seq_length:
                print(f"Warning: Chromosome {chrom} shorter than sequence length. Skipping.")
                continue
            
            max_start = chrom_len - seq_length
            total_unique_positions = max_start + 1
            
            if num_seqs_per_chrom > total_unique_positions:
                print(f"Warning: Requested {num_seqs_per_chrom} sequences for {chrom}, "
                      f"but only {total_unique_positions} unique positions available. Using all unique positions.")
                start_positions = list(range(total_unique_positions))
            else:
                start_positions = random.sample(range(total_unique_positions), num_seqs_per_chrom)
            
            for start in start_positions:
                end = start + seq_length

                out_handle.write(f"{chrom}\t{start}\t{end}\n")

print("Created BED file with random intervals.")


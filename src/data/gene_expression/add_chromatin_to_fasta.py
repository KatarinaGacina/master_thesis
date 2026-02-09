import numpy as np
import gzip
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pysam
import io

from data.load_utils.data_loading import load_intervals_as_dict

def modify_label(label, chrom_info):
    for start, end in chrom_info:
        label[start:end] = True
    return label

def lowercase_intervals(fasta_file, output_file, bed_file):
    chrom_dict = load_intervals_as_dict(bed_file)

    with gzip.open(fasta_file, "rt") as in_handle, pysam.BGZFile(output_file, "w") as out_handle:
        out_handle_text = io.TextIOWrapper(out_handle, encoding="ascii")

        for record in SeqIO.parse(in_handle, "fasta"):
            chrom = record.id
            seq_array = np.frombuffer(str(record.seq).encode("utf-8"), dtype=np.uint8).copy()
            
            label = np.zeros(len(record.seq), dtype=bool)
            modify_label(label, chrom_dict.get(chrom, []))

            is_uppercase = (seq_array >= 65) & (seq_array <= 90)
            mask = label & is_uppercase
            seq_array[mask] += 32

            modified_seq = Seq(seq_array.tobytes().decode("ascii"))
            print(set(modified_seq))

            new_record = SeqRecord(
                modified_seq, 
                id=record.id, 
                description=record.description
            )

            SeqIO.write(new_record, out_handle_text, "fasta")
            out_handle_text.flush()

            print(f"Finished with record {chrom}")
            
        out_handle_text.close()

if __name__ == "__main__":
    input_fasta = "/path/assembly.fasta.gz"
    input_bed = "/path/atac_peaks.bed"
    output_fasta = "/path/assembly_with_chromatin.fasta.gz"

    print("Process started...")
    lowercase_intervals(input_fasta, output_fasta, input_bed)
    print(f"\nSuccess! Masked file saved as: {output_fasta}")
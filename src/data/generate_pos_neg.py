import gzip
from Bio import SeqIO
from intervaltree import Interval, IntervalTree
import json

def load_split(json_name):
    with open(json_name) as f:
        chrom_data = json.load(f)

    test_list = chrom_data.get("test", [])
    valid_list = chrom_data.get("valid", [])
    train_list = chrom_data.get("train", [])

    return train_list, valid_list, test_list


def is_overlap(chrom, start, end):
    if chrom in chromatin_trees:
        return len(chromatin_trees[chrom][start:end]) > 0
    return False


if __name__ == "__main__":

    train_list, val_list, test_list = load_split("/path/split_0.json")

    fasta_file = "/path/assembly.fasta.gz"
    bed_file = "/path/atac_peaks.bed"
    positives_file = "/path/positives_10000.bed"
    negatives_file = "/path/negatives_10000.bed"

    target_size = 5000
    flank = 1000

    chromatin_trees = {}
    with open(bed_file) as bf:
        for line in bf:
            if line.startswith("#") or line.strip() == "":
                continue
            chrom, start, end = line.strip().split()[:3]
            start, end = int(start), int(end)
            if chrom not in chromatin_trees:
                chromatin_trees[chrom] = IntervalTree()
            chromatin_trees[chrom][start:end] = True

    print("Loaded intervals...")

    with open(positives_file, "w") as pos_out, open(negatives_file, "w") as neg_out:
        with gzip.open(fasta_file, "rt") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                chrom = record.id
                seq_len = len(record.seq)

                if chrom in train_list:
                    chunk_size = target_size + flank
                else:
                    chunk_size = target_size

                for start in range(0, seq_len, chunk_size):
                    end = min(start + chunk_size, seq_len)
                    if end - start < target_size:
                        continue

                    line = f"{chrom}\t{start}\t{end}\n"
                    if is_overlap(chrom, start, end):
                        pos_out.write(line)
                    else:
                        neg_out.write(line)

                print(f"Processed chrom {chrom}")
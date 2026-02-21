import torch
from torch.utils.data import Dataset

import pyfaidx
import pyBigWig
import numpy as np

from data.load_utils.data_loading import retrieve_spec_seq, load_genome_sizes, add_methylation_info2


class GeneDataset(Dataset):
    
    def __init__(self, fasta_name, bw_methyl_name, intervals, labels, tokenizer):
        
        self.fasta_name = fasta_name
        self.bw_methyl_name = bw_methyl_name

        self.fa = None
        self.m_bw = None

        fai_filename = fasta_name + ".fai"
        self.chrom_sizes = load_genome_sizes(fai_filename)

        self.intervals = intervals
        assert len(self.intervals) > 0

        self.labels = labels

        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.intervals)

    def _open_files(self):
        if self.fa is None:
            self.fa = pyfaidx.Fasta(self.fasta_name)
        if self.m_bw is None and self.bw_methyl_name is not None:
            self.m_bw = pyBigWig.open(self.bw_methyl_name)

    def __getitem__(self, idx):
        self._open_files()

        chrom, start, end, strand = self.intervals[idx]
        record = retrieve_spec_seq(self.fa, chrom, start, end)

        if strand == "-":
            record = record.complement

            if self.m_bw is None:
                record = record.reverse

        record = record.seq
        if self.m_bw is not None:
            seq_array = np.frombuffer(record.encode("utf-8"), dtype=np.uint8).copy()
            seq_array = add_methylation_info2(chrom, start, end, seq_array, strand, self.m_bw) 

            if strand == "-":
                seq_array = seq_array[::-1]

            record = seq_array.tobytes().decode('ascii')

        seq_tensor = self.tokenizer.encode(record, add_special_tokens=False)

        return seq_tensor, self.labels[idx]

    def close(self):
        if self.fa is not None:
            self.fa.close()
        if self.m_bw is not None:
            self.m_bw.close()
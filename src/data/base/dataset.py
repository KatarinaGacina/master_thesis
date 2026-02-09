import torch
from torch.utils.data import Dataset

import pyfaidx
import pyBigWig
import numpy as np

from data.load_utils.data_loading import retrieve_spec_seq, load_genome_sizes, add_methylation_info


class ReadsDataset(Dataset):
    
    def __init__(self, fasta_name, bw_methyl_name, intervals, tokenizer):
        
        self.fasta_name = fasta_name
        self.bw_methyl_name = bw_methyl_name

        self.fa = None
        self.m_bw = None

        fai_filename = fasta_name + ".fai"
        self.chrom_sizes = load_genome_sizes(fai_filename)

        self.intervals = intervals
        assert len(self.intervals) > 0

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

        chrom, start, end = self.intervals[idx]
        record = retrieve_spec_seq(self.fa, chrom, start, end).seq

        if self.m_bw is not None: #not preprocessed into dataset, because it is strand specific
            seq_array = np.frombuffer(record.encode("utf-8"), dtype=np.uint8).copy()
            seq_array = add_methylation_info(chrom, start, end, seq_array, "+", self.m_bw, only_m=True) 
            record = seq_array.tobytes().decode('ascii')

        tokenized_output = self.tokenizer( 
            record,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=True,
            #return_special_tokens_mask=True,
            padding=False
        )
        
        return tokenized_output

    def close(self):
        if self.fa is not None:
            self.fa.close()
        if self.m_bw is not None:
            self.m_bw.close()
import torch
from torch.utils.data import Dataset

import warnings
import random
import math

import pyfaidx
import pyBigWig
import numpy as np

from data.load_utils.data_loading import retrieve_spec_seq, retrieve_spec_label, load_genome_sizes, add_methylation_info

class ChromDataset(Dataset):
    
    def __init__(self, fasta_name, bw_name, bw_methyl_name, tokenizer, positives_intervals, only_m=False, negatives_intervals=None, train_mode=False, add_revcomp=False, outputlen=None, shift=0):

        self.fasta_name = fasta_name
        self.bw_name = bw_name
        self.bw_methyl_name = bw_methyl_name

        self.only_m = only_m

        self.fa = None
        self.bw = None
        self.m_bw = None

        fai_filename = fasta_name + ".fai"
        self.chrom_sizes = load_genome_sizes(fai_filename)

        self.intervals = []
        if positives_intervals is not None and len(positives_intervals) > 0:
            self.intervals.extend(positives_intervals)
        if negatives_intervals is not None and len(negatives_intervals) > 0:
            self.intervals.extend(negatives_intervals)

        assert len(self.intervals) > 0

        self.tokenizer = tokenizer
        
        self.train_mode = train_mode
        self.add_revcomp = add_revcomp
        self.outputlen = outputlen

        self.shift = shift #not in current use

        self.provjera = False
        
    def __len__(self):
        return len(self.intervals)

    def _open_files(self):
        if self.bw is None:
            self.bw = pyBigWig.open(self.bw_name)
        if self.m_bw is None and self.bw_methyl_name is not None:
            self.m_bw = pyBigWig.open(self.bw_methyl_name)
        if self.fa is None:
            self.fa = pyfaidx.Fasta(self.fasta_name, rebuild=False)

    def __getitem__(self, idx):
        self._open_files()

        if self.provjera == False:
            print(f"[Worker] torch.initial_seed(): {torch.initial_seed()}")
            self.provjera = True

        chrom, start, end = self.intervals[idx]
        strand="+"

        if self.train_mode:
            if self.outputlen is not None and (end - start + (2 * self.shift)) > self.outputlen:
                if self.shift > 0:
                    start = max(0, start - self.shift)
                    end = min(self.chrom_sizes[chrom], end + self.shift)

                max_shift = (end - start) - self.outputlen
                shift = random.randint(0, max_shift)

                start = start + shift
                end = min(self.chrom_sizes[chrom], start + self.outputlen)
                
        else: #centers val and test examples if they are longer than target length, if target length provided
            if self.outputlen is not None and (end - start) > self.outputlen:
                center = (start + end) // 2
                start = center - (self.outputlen // 2)
                end = center + (self.outputlen - self.outputlen // 2)
        
        seq = retrieve_spec_seq(self.fa, chrom, start, end)
        label, _ = retrieve_spec_label(self.bw, chrom, start, end)

        label_tensor = torch.from_numpy(label)

        if self.train_mode and self.add_revcomp and random.random() < 0.5:
            seq = seq.complement
            strand = "-"

        seq_array = np.frombuffer(seq.seq.encode("utf-8"), dtype=np.uint8)
        if self.m_bw is not None:
            seq_array = seq_array.copy()
            seq_array = add_methylation_info(chrom, start, end, seq_array, strand, self.m_bw, only_m=self.only_m)
        
        if strand == "-":
            seq_array = seq_array[::-1]
            label_tensor = label_tensor.flip(dims=[0])

        seq = seq_array.tobytes().decode('ascii')
        seq_tensor = self.tokenizer.encode(seq)
        
        return seq_tensor, label_tensor

    def close(self):
        if self.fa is not None:
            self.fa.close()
        if self.bw is not None:
            self.bw.close()
        if self.m_bw is not None:
            self.m_bw.close()
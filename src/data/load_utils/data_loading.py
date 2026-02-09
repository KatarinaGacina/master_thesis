import pyfaidx
import pyBigWig
import numpy as np
import json
import random
from collections import defaultdict

def load_splits(json_name):
    with open(json_name) as f:
        chrom_data = json.load(f)

    test_list = chrom_data.get("test", [])
    valid_list = chrom_data.get("valid", [])
    train_list = chrom_data.get("train", [])

    return train_list, valid_list, test_list


def load_genome_sizes(genome_sizes_file):
    genome_sizes = {}

    with open(genome_sizes_file, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            fields = line.strip().split()
            chrom, size = fields[:2]
            genome_sizes[chrom] = int(size)

    return genome_sizes


def load_intervals_as_dict(bed_name, sort=False):
    interval_dict = defaultdict(list)

    with open(bed_name) as bed_file:
        for line in bed_file:
            if line.startswith("#") or not line.strip():
                continue
            
            fields = line.strip().split("\t")
            chrom, start, end = fields[:3]

            interval_dict[chrom].append((int(start), int(end)))

    if sort:
        for chrom in interval_dict:
            interval_dict[chrom].sort()
    
    return interval_dict

def load_intervals_as_list(bed_name):
    intervals = []

    with open(bed_name) as bed_file:
        for line in bed_file:
            if line.startswith("#") or not line.strip():
                continue
            
            fields = line.strip().split("\t")
            chrom, start, end = fields[:3]

            intervals.append((chrom, int(start), int(end)))
    
    return intervals


def load_intervals_per_split(bed_filename, split_list):
    intervals = []

    with open(bed_filename) as bed_file:
        for line in bed_file:
            if line.startswith("#") or not line.strip():
                continue
            
            fields = line.strip().split("\t")
            chrom, start, end = fields[:3]

            if chrom in split_list:
                intervals.append((chrom, int(start), int(end)))
    
    return intervals


def load_intervals_per_all_splits(bed_filename, split_file):
    train_split, val_split, test_split = load_splits(split_file)

    split_interval_dict = {"train": [], "val": [], "test": []}
    if bed_filename is None:
        return split_interval_dict

    with open(bed_filename) as bed_file:
        for line in bed_file:
            if line.startswith("#") or not line.strip():
                continue
            
            fields = line.strip().split("\t")
            chrom, start, end = fields[:3]

            if chrom in train_split:
                split_interval_dict["train"].append((chrom, int(start), int(end)))
            elif chrom in val_split:
                split_interval_dict["val"].append((chrom, int(start), int(end)))
            elif chrom in test_split:
                split_interval_dict["test"].append((chrom, int(start), int(end)))
    
    return split_interval_dict

def load_dict_per_all_splits(bed_filename, split_file):
    train_split, val_split, test_split = load_splits(split_file)

    split_interval_dict = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
    if bed_filename is None:
        return split_interval_dict

    with open(bed_filename) as bed_file:
        for line in bed_file:
            if line.startswith("#") or not line.strip():
                continue
            
            fields = line.strip().split("\t")
            chrom, start, end = fields[:3]

            if chrom in train_split:
                split_interval_dict["train"][chrom].append((chrom, int(start), int(end)))
            elif chrom in val_split:
                split_interval_dict["val"][chrom].append((chrom, int(start), int(end)))
            elif chrom in test_split:
                split_interval_dict["test"][chrom].append((chrom, int(start), int(end)))
    
    return split_interval_dict


def load_dict_per_all_splits_gene(bed_filename, split_file):
    train_split, val_split, test_split = load_splits(split_file)

    split_interval_dict = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
    if bed_filename is None:
        return split_interval_dict

    with open(bed_filename) as bed_file:
        for line in bed_file:
            if line.startswith("#") or not line.strip():
                continue
            
            fields = line.strip().split("\t")
            chrom, start, end = fields[:3]
            strand = fields[3]

            if chrom in train_split:
                split_interval_dict["train"][chrom].append((chrom, int(start), int(end), strand))
            elif chrom in val_split:
                split_interval_dict["val"][chrom].append((chrom, int(start), int(end), strand))
            elif chrom in test_split:
                split_interval_dict["test"][chrom].append((chrom, int(start), int(end), strand))
    
    return split_interval_dict



def load_methyl_data_as_dict(methylation_file, threshold=50.0):
    m_dict = defaultdict(lambda: defaultdict(list))
    h_dict = defaultdict(lambda: defaultdict(list))

    with open(methylation_file) as bed:
        for line in bed:
            if line.startswith("#") or not line.strip():
                continue
            
            fields = line.strip().split("\t")
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])

            value = fields[3]
            score = float(fields[10])

            strand = fields[5].strip()
            assert strand in ("+", "-")

            if value == "m" and score >= threshold:
                m_dict[chrom][strand].append((start, end))
            elif value == "h" and score > threshold:
                h_dict[chrom][strand].append((start, end))

    for chrom in m_dict:
        for strand in m_dict[chrom]:
            m_dict[chrom][strand].sort(key=lambda x: x[0])
    for chrom in h_dict:
        for strand in h_dict[chrom]:
            h_dict[chrom][strand].sort(key=lambda x: x[0])

    return m_dict, h_dict

def add_methylation_info(chrom, start, end, sequence, strand, m_bw=None, only_m=False): #accept sequence as np.array
    assert m_bw is not None
    assert len(sequence) == end - start 
    assert strand in ("+", "-")
    
    m_values = np.nan_to_num(m_bw.values(chrom, start, end))

    if strand == "+":
        m_map = {1: "m", 2: "h"}
    elif strand == "-":
        m_map = {-1: "m", -2: "h"}

    for key, replacement_char in m_map.items():
        indices_to_replace = (m_values == key)
        if only_m:
            sequence[indices_to_replace] = ord("m")
        else:
            sequence[indices_to_replace] = ord(replacement_char)
        
    return sequence

def add_methylation_info2(chrom, start, end, sequence, strand, m_bw=None, only_m=False): #accept sequence as np.array
    assert m_bw is not None
    assert len(sequence) == end - start 
    assert strand in ("+", "-")
    
    m_values = np.nan_to_num(m_bw.values(chrom, start, end))

    #future: better to have two separate bigwig files
    if strand == "+":
        m_map = {1: "m", 2: "h"}
    elif strand == "-":
        m_map = {-1: "m", -2: "h"}

    upper_mask = (sequence >= ord("A")) & (sequence <= ord("Z"))
    lower_mask = (sequence >= ord("a")) & (sequence <= ord("z"))

    for key, replacement_char in m_map.items():
        indices_to_replace = (m_values == key)
        if only_m:
            sequence[indices_to_replace & upper_mask] = ord("m")
            sequence[indices_to_replace & lower_mask] = ord("M")
        else:
            sequence[indices_to_replace & upper_mask] = ord(replacement_char.lower())
            sequence[indices_to_replace & lower_mask] = ord(replacement_char.upper())
        
    return sequence

def retrieve_seqs(genome, intervals, chrom):
    seqs = []
    for start, end in intervals.get(chrom, []):
        seq = genome[chrom][start:end]
        seqs.append(seq)

    return seqs

def retrieve_labels(bw, intervals, chrom):
    labels = []
    counts = []
    for start, end in intervals.get(chrom, []):
        label = np.nan_to_num(bw.values(chrom, start, end))
        labels.append(label)
        counts.append(np.any(label))
    
    return labels, counts


def retrieve_spec_seq(genome, chrom, start, end):
    return genome[chrom][start:end]

def retrieve_spec_label(bw, chrom, start, end):
    label = np.nan_to_num(bw.values(chrom, start, end))
    count = np.any(label)
    
    return label, count
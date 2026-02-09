import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
import pytorch_lightning as pl

from functools import partial

from data.chromatin.dataset import ChromDataset
from data.load_utils.data_loading import load_intervals_per_all_splits, load_dict_per_all_splits
from data.tokenizer.tokenizer import DNATokenizer

import warnings

import torch.distributed as dist

import random
from collections import defaultdict
import numpy as np


def pad_collate_fn(batch, pad_index):
    assert pad_index is not None

    #sequences = [item[0] for item in batch]
    sequences = [
        item[0] if isinstance(item[0], torch.Tensor) else torch.tensor(item[0], dtype=torch.long) for item in batch
    ]
    #labels = [item[1] for item in batch]
    labels = [
        item[1] if isinstance(item[1], torch.Tensor) else torch.tensor(item[1], dtype=torch.long) for item in batch
    ]

    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=pad_index)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=pad_index)
    attention_mask = (sequences_padded != pad_index)

    return {
        "input_ids": sequences_padded,
        "attention_mask": attention_mask,
        "labels": labels_padded
    }


class ChromDataModule(pl.LightningDataModule):
    def __init__(self, config, parser_args, only_m=False, tokenizer=None, num_workers=2, data_seed=42):
        super().__init__()

        self.data_seed = data_seed

        self.config = config
        self.parser_args = parser_args
        self.num_workers = num_workers

        if tokenizer is None:
            self.tokenizer = DNATokenizer()
        else:
            self.tokenizer = tokenizer
        self.only_m = only_m
        print(f"Not using h: {only_m}")

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _subsample_dataset(self, positives_dict, negatives_dict):
        rng = random.Random(self.data_seed)

        positives = []
        balanced_negatives = []

        for chrom, pos_list in positives_dict.items():
            n_pos = len(pos_list)
            positives.extend(pos_list)

            neg_list = negatives_dict.get(chrom, [])

            if len(neg_list) == 0:
                warnings.warn(
                    f"No negative intervals available on {chrom}. "
                    f"Skipping negatives for this chromosome."
                )
                continue
            
            if len(neg_list) < n_pos:
                warnings.warn(
                    f"Not enough negatives on {chrom}: "
                    f"need {n_pos}, have {len(neg_list)}. "
                    f"Using all available negatives."
                )
                balanced_negatives.extend(neg_list)
            else:
                balanced_negatives.extend(
                    rng.sample(neg_list, n_pos)
                )

        return positives, balanced_negatives


    def setup(self, stage=None):
        positives_dict = load_dict_per_all_splits(self.parser_args["positives_bed"], self.parser_args["split"])
        negatives_dict = load_dict_per_all_splits(self.parser_args["negatives_bed"], self.parser_args["split"])
        
        positives_intervals_train, negatives_intervals_train = self._subsample_dataset(positives_dict["train"], negatives_dict["train"])
        positives_intervals_val, negatives_intervals_val = self._subsample_dataset(positives_dict["val"], negatives_dict["val"])

        positives_intervals_test, negatives_intervals_test = self._subsample_dataset(positives_dict["test"], negatives_dict["test"])

        #print(positives_dict["test"].keys())
        #print(negatives_dict["test"].keys())

        self.train_dataset = ChromDataset(
            self.parser_args["fasta_file"],
            self.parser_args["bigwig_file"],
            self.parser_args["methylation_bigwig"],
            self.tokenizer,
            positives_intervals_train,
            self.only_m,
            negatives_intervals_train,
            train_mode=True,
            add_revcomp=True,
            outputlen=self.config["outputlen"]
        )
        self.val_dataset = ChromDataset(
            self.parser_args["fasta_file"],
            self.parser_args["bigwig_file"],
            self.parser_args["methylation_bigwig"],
            self.tokenizer,
            positives_intervals_val,
            self.only_m,
            negatives_intervals_val,
            train_mode=False,
            add_revcomp=False,
            outputlen=self.config["outputlen"]
        )
        self.test_dataset = ChromDataset(
            self.parser_args["fasta_file"],
            self.parser_args["bigwig_file"],
            self.parser_args["methylation_bigwig"],
            self.tokenizer,
            positives_intervals_test,
            self.only_m,
            negatives_intervals_test,
            train_mode=False,
            add_revcomp=False,
            outputlen=self.config["outputlen"]
        )

    def train_dataloader(self):
        if dist.is_initialized():
            sampler = DistributedSampler(self.train_dataset, shuffle=True, drop_last=True)
            print("Distributed sampler initilized.")
        else:
            sampler = None

        train_g = torch.Generator().manual_seed(self.data_seed) 

        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            sampler=sampler,
            shuffle=(sampler is None),
            drop_last=True,
            collate_fn=partial(pad_collate_fn, pad_index=self.tokenizer.get_pad_index()),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            generator=train_g
        )

    def val_dataloader(self):
        if dist.is_initialized():
            sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            sampler = None

        val_g = torch.Generator().manual_seed(self.data_seed) 

        return DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            sampler=sampler,
            shuffle=False,
            drop_last=True,
            collate_fn=partial(pad_collate_fn, pad_index=self.tokenizer.get_pad_index()),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            generator=val_g
        )

    def test_dataloader(self):
        if dist.is_initialized():
            sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            sampler = None

        test_g = torch.Generator().manual_seed(self.data_seed) 

        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            sampler=sampler,
            shuffle=False,
            drop_last=False,
            collate_fn=partial(pad_collate_fn, pad_index=self.tokenizer.get_pad_index()),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            generator=test_g
        )

    def teardown(self, stage=None):
        if hasattr(self.train_dataset, "close"):
            self.train_dataset.close()
        if hasattr(self.val_dataset, "close"):
            self.val_dataset.close()
        if hasattr(self.test_dataset, "close"):
            self.test_dataset.close()
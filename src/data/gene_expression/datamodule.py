import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
import pytorch_lightning as pl

from functools import partial

from data.gene_expression.dataset import GeneDataset
from data.load_utils.data_loading import load_dict_per_all_splits_gene
from data.tokenizer.tokenizer import DNATokenizer

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
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=pad_index)
    attention_mask = (sequences_padded != pad_index)
    
    if isinstance(batch[0][1], torch.Tensor):
        labels = torch.stack([item[1] for item in batch]).float()
    else:
        labels = torch.tensor([item[1] for item in batch], dtype=torch.float)

    return {
        "input_ids": sequences_padded,
        "attention_mask": attention_mask,
        "labels": labels
    }


class GeneDataModule(pl.LightningDataModule):
    def __init__(self, config, parser_args, tokenizer=None, num_workers=2):
        super().__init__()

        self.config = config
        self.parser_args = parser_args
        self.num_workers = num_workers

        if tokenizer is None:
            self.tokenizer = DNATokenizer(chromatin_tokens=parser_args.get("chromatin_tokens_used", False))
        else:
            self.tokenizer = tokenizer

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        positives_dict = load_dict_per_all_splits_gene(self.parser_args["positives_bed"], self.parser_args["split"])
        negatives_dict = load_dict_per_all_splits_gene(self.parser_args["negatives_bed"], self.parser_args["split"])

        intervals_train = [pos for pos_list in positives_dict["train"].values() for pos in pos_list]
        pos_train_len = len(intervals_train)
        intervals_train.extend([neg for neg_list in negatives_dict["train"].values() for neg in neg_list])

        intervals_val = [pos for pos_list in positives_dict["val"].values() for pos in pos_list]
        pos_val_len = len(intervals_val)
        intervals_val.extend([neg for neg_list in negatives_dict["val"].values() for neg in neg_list])
        
        intervals_test = [pos for pos_list in positives_dict["test"].values() for pos in pos_list]
        pos_test_len = len(intervals_test)
        intervals_test.extend([neg for neg_list in negatives_dict["test"].values() for neg in neg_list])

        labels_train = np.zeros(len(intervals_train), dtype=np.uint8)
        labels_train[:pos_train_len] = 1
        labels_val = np.zeros(len(intervals_val), dtype=np.uint8)
        labels_val[:pos_val_len] = 1
        labels_test = np.zeros(len(intervals_test), dtype=np.uint8)
        labels_test[:pos_test_len] = 1

        self.train_dataset = GeneDataset(
            self.parser_args["fasta_file"],
            self.parser_args["methylation_bigwig"],
            intervals_train,
            labels_train,
            self.tokenizer
        )
        self.val_dataset = GeneDataset(
            self.parser_args["fasta_file"],
            self.parser_args["methylation_bigwig"],
            intervals_val,
            labels_val,
            self.tokenizer
        )
        self.test_dataset = GeneDataset(
            self.parser_args["fasta_file"],
            self.parser_args["methylation_bigwig"],
            intervals_test,
            labels_test,
            self.tokenizer
        )

    def train_dataloader(self):
        if dist.is_initialized():
            sampler = DistributedSampler(self.train_dataset, shuffle=True, drop_last=True)
            print("Distributed sampler initilized.")
        else:
            sampler = None

        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            sampler=sampler,
            shuffle=(sampler is None),
            drop_last=True,
            collate_fn=partial(pad_collate_fn, pad_index=self.tokenizer.pad_token_id),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        if dist.is_initialized():
            sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            sampler = None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            sampler=sampler,
            shuffle=False,
            drop_last=True,
            collate_fn=partial(pad_collate_fn, pad_index=self.tokenizer.pad_token_id),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        if dist.is_initialized():
            sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            sampler = None

        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            sampler=sampler,
            shuffle=False,
            drop_last=False,
            collate_fn=partial(pad_collate_fn, pad_index=self.tokenizer.pad_token_id),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def teardown(self, stage=None):
        if hasattr(self.train_dataset, "close"):
            self.train_dataset.close()
        if hasattr(self.val_dataset, "close"):
            self.val_dataset.close()
        if hasattr(self.test_dataset, "close"):
            self.test_dataset.close()
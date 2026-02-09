import torch
from torch.utils.data import DataLoader, DistributedSampler
import pytorch_lightning as pl
import torch.distributed as dist

from transformers import DataCollatorForLanguageModeling
#from data.base.data_collator import RareClassMLMDataCollator

from data.tokenizer.tokenizer import DNATokenizerHF
from data.base.dataset import ReadsDataset

from data.load_utils.data_loading import load_intervals_as_list

#from datasets import load_from_disk

import numpy as np


class RepresentationsDataModule(pl.LightningDataModule):
    def __init__(self, config, parser_args, tokenizer=None, num_workers=4, data_seed=42):
        super().__init__()

        self.data_seed = data_seed

        self.config = config
        self.parser_args = parser_args

        self.num_workers = num_workers

        if tokenizer is None:
            assert self.config["specified_len"] is not None
            self.tokenizer = DNATokenizerHF(padding_side="right", truncation_side="right", model_max_length=config["specified_len"])
        else:
            self.tokenizer = tokenizer

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _train_val_test_split(self, intervals, train_size=0.7, val_size=0.15, seed=None):
        n = len(intervals)
        indices = np.arange(n)

        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

        train_end = int(n * train_size)
        val_end = train_end + int(n * val_size)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        return (
            [intervals[i] for i in train_idx],
            [intervals[i] for i in val_idx],
            [intervals[i] for i in test_idx],
        )


    def setup(self, stage=None):
        #support for HuggingFace Dataset
        """dataset = load_from_disk(self.parser_args["hf_dataset"])

        def encode(batch):
            return self.tokenizer(
                batch["sequences"], 
                truncation=True,
                add_special_tokens=False,
                return_token_type_ids=False,
                return_special_tokens_mask=True,
                padding="longest",
                return_tensors="pt"
            )

        self.train_dataset = dataset["train"]
        self.train_dataset.set_transform(encode)
        self.val_dataset = dataset["val"]
        self.val_dataset.set_transform(encode)
        self.test_dataset = dataset["test"]
        self.test_dataset.set_transform(encode)"""   

        intervals = load_intervals_as_list(self.parser_args["intervals_bed"])
        train_intervals, val_intervals, test_intervals = self._train_val_test_split(intervals, seed=self.data_seed)

        self.train_dataset = ReadsDataset(
            self.parser_args["fasta_file"],
            self.parser_args["methylation_bigwig"],
            train_intervals,
            self.tokenizer
        )
        self.val_dataset = ReadsDataset(
            self.parser_args["fasta_file"],
            self.parser_args["methylation_bigwig"],
            val_intervals,
            self.tokenizer
        )
        self.test_dataset = ReadsDataset(
            self.parser_args["fasta_file"],
            self.parser_args["methylation_bigwig"],
            test_intervals,
            self.tokenizer
        )

    def train_dataloader(self):
        if dist.is_initialized():
            sampler = DistributedSampler(self.train_dataset, shuffle=True, drop_last=True)
            print("Distributed sampler initilized.")
        else:
            sampler = None

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            sampler=sampler,
            shuffle=(sampler is None),
            drop_last=True,
            collate_fn=data_collator,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        if dist.is_initialized():
            sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            sampler = None

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        return DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            sampler=sampler,
            shuffle=False,
            drop_last=False,
            collate_fn=data_collator,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        if dist.is_initialized():
            sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            sampler = None

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            sampler=sampler,
            shuffle=False,
            drop_last=False,
            collate_fn=data_collator,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
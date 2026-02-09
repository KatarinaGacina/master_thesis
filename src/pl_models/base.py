import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

import pytorch_lightning as pl

from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR

from data.tokenizer.tokenizer import DNATokenizerHF
from model.base_model import BaseModel, BaseModelLongContext

from torchmetrics.text import Perplexity


def update_multiclass_cm(preds, ground_truth, num_classes, cm, device=None):
    if cm is None:
        if device is None:
            device = preds.device
        cm = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    indices = torch.stack((ground_truth, preds), dim=1)
    indices, counts = torch.unique(indices, return_counts = True, dim=0)
    cm[indices[:, 0], indices[:, 1]] += counts

    return cm

def compute_metrics_from_multiclass_cm(cm, name):
    assert cm is not None
    assert name is not None

    eps = 1e-8
    num_classes = cm.shape[0]

    metrics = {}

    accuracy = torch.trace(cm).float() / (cm.sum() + eps)
    metrics[f"{name}/accuracy"] = accuracy.item()

    for c in range(num_classes):
        tp = cm[c, c].float()
        fp = cm[:, c].sum().float() - tp
        fn = cm[c, :].sum().float() - tp

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        metrics[f"{name}/class{c}_precision"] = precision
        metrics[f"{name}/class{c}_recall"] = recall
        metrics[f"{name}/class{c}_f1"] = f1

    return metrics


class BaseLightningModule(pl.LightningModule):
    def __init__(self, config, model_name, tokenizer=None):
        super().__init__()

        if tokenizer is None:
            if config["specified_len"] is not None:
                self.tokenizer = DNATokenizerHF(padding_side="right", truncation_side="right", model_max_length=config["specified_len"])
            else:
                self.tokenizer = DNATokenizerHF(padding_side="right", truncation_side="right")

        self.output_vocab_size = config.get("output_vocab_size", self.tokenizer.vocab_size)
        
        config["output_vocab_size"] = self.output_vocab_size
        config["vocab_size"] = self.tokenizer.vocab_size
        config["pad_index"] = self.tokenizer.pad_token_id
        self.save_hyperparameters(config)

        if model_name == "longconetxt":
            self.model = BaseModelLongContext(config)
        else:
            self.model = BaseModel(config)

        weight = config.get("weight", None)
        if weight is not None:
            weight = torch.tensor(weight)
        self.ignore_index = -100 #default
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=self.ignore_index,
            reduction="mean"
        )

        self.lr = config.get("learning_rate", 1e-4)
        self.weight_decay = config.get("weight_decay", 1e-5)
        self.num_epochs = config.get("num_epochs", 10)

        self.ppl_metric = Perplexity(ignore_index=-100)
    
    def setup(self, stage=None):
        pass


    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"].bool()[:, None, None, :]

        output = self(input_ids, attention_mask)

        logits = output["logits"].flatten(0, 1)
        labels = labels.flatten()

        loss = self.criterion(logits, labels)
        self.log("train/loss", loss, sync_dist=True)
        self.log("train/loss_epoch", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss
    
    def on_train_epoch_end(self):
        if self.global_rank == 0:
            mem_alloc = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[Epoch {self.current_epoch}] Alloc: {mem_alloc:.2f} GB | Reserved: {mem_reserved:.2f} GB")


    def reset_metrics(self):
        self.cm = torch.zeros((self.output_vocab_size, self.output_vocab_size), dtype=torch.int64, device=self.device)
        self.ppl_metric.reset()
    
    def _on_end_eval_step(self, name):
        assert name is not None

        cm_agg = self.cm.clone()

        if self.trainer.world_size > 1:
            torch.distributed.all_reduce(cm_agg, op=torch.distributed.ReduceOp.SUM)

        metrics_tokens = compute_metrics_from_multiclass_cm(cm_agg, name=f"{name}")
        ppl_metric = self.ppl_metric.compute()

        self.log_dict({**metrics_tokens, f"{name}/perplexity": ppl_metric.item()}, sync_dist=True)


    def on_validation_epoch_start(self):
        self.reset_metrics()

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"].bool()[:, None, None, :]

        output = self(input_ids, attention_mask)

        logits = output["logits"]

        logits_flat = logits.flatten(0, 1)
        labels_flat = labels.flatten()

        if self.trainer.sanity_checking:
            print(attention_mask.shape)

        loss = self.criterion(logits_flat, labels_flat)
        self.log("val/loss", loss, sync_dist=True)

        preds = logits_flat.argmax(dim=-1)

        mask = labels_flat != self.ignore_index
        update_multiclass_cm(preds[mask].detach(), labels_flat[mask].detach(), self.output_vocab_size, self.cm)

        self.ppl_metric.update(logits.detach(), labels.detach())

        return loss

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.reset_metrics()
            return

        self._on_end_eval_step("val")


    def on_test_epoch_start(self):
        self.reset_metrics()

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"].bool()[:, None, None, :]

        output = self(input_ids, attention_mask)

        logits = output["logits"]

        logits_flat = logits.flatten(0, 1)
        labels_flat = labels.flatten()

        preds = logits_flat.argmax(dim=-1) #softmax

        mask = labels_flat != self.ignore_index
        update_multiclass_cm(preds[mask].detach(), labels_flat[mask].detach(), self.output_vocab_size, self.cm)

        self.ppl_metric.update(logits.detach(), labels.detach())

        return

    def on_test_epoch_end(self):
        self._on_end_eval_step("test")


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        warmup_scheduler = {
            "scheduler": LinearLR(
                optimizer,
                start_factor=1e-6,
                end_factor=1.0,
                total_iters=2500
            ),
            "interval": "step",
            "frequency": 1,
        }
        plateau_scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.9,
                patience=3,
                verbose=True
            ),
            "monitor": "train/loss_epoch",
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer], [warmup_scheduler, plateau_scheduler]

    def on_save_checkpoint(self, checkpoint):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    
        if rank == 0:
            checkpoint['base_model_weights'] = self.model.dna_model.state_dict()
        
        return checkpoint
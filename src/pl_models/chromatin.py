import torch
import torch.optim as optim
import torch.distributed as dist

import pytorch_lightning as pl

from model.model import ChromatinModel, ChromatinModelLongContext

from utils.loss import BCELossMasked
from data.tokenizer.tokenizer import DNATokenizer

from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchmetrics

import random
import math
import os

import torch._dynamo
#torch._dynamo.config.suppress_errors = True
#torch._dynamo.config.disable = True


def update_binary_cm(pred, true, cm, device=None):
    if cm is None:
        if device is None:
            device = pred.device
        cm = torch.zeros((2, 2), dtype=torch.int64, device=device)

    tp = ((pred == 1) & (true == 1)).sum()
    tn = ((pred == 0) & (true == 0)).sum()
    fp = ((pred == 1) & (true == 0)).sum()
    fn = ((pred == 0) & (true == 1)).sum()

    cm[0, 0] += tn
    cm[0, 1] += fp
    cm[1, 0] += fn
    cm[1, 1] += tp

    return cm

def compute_metrics_from_cm(cm, name):
    assert name is not None
    assert cm is not None

    eps = 1e-8

    tn, fp = cm[0]
    fn, tp = cm[1]

    print(f"tn: {tn}")
    print(f"fp: {fp}")
    print(f"fn: {fn}")
    print(f"tp: {tp}")

    precision_1 = tp / (tp + fp + eps)
    recall_1 = tp / (tp + fn + eps)
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1 + eps)

    precision_0 = tn / (tn + fn + eps)
    recall_0 = tn / (tn + fp + eps)
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0 + eps)

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    """numerator = (tp * tn) - (fp * fn)
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator.item() == 0:
        mcc = torch.tensor(0.0)
    else:
        mcc = numerator / denominator"""

    return {
        f"{name}_metrics/accuracy": accuracy,
        f"{name}_metrics/class1_precision": precision_1,
        f"{name}_metrics/class1_recall": recall_1,
        f"{name}_metrics/class1_f1": f1_1,
        f"{name}_metrics/class0_precision": precision_0,
        f"{name}_metrics/class0_recall": recall_0,
        f"{name}_metrics/class0_f1": f1_0
    }


class ChromatinLightningModule(pl.LightningModule):
    def __init__(self, config, model_name, tokenizer=None, pretrained=None):
        super().__init__()

        if tokenizer is None:
            self.tokenizer = DNATokenizer()
        else:
            self.tokenizer = tokenizer
        config["vocab_size"] = self.tokenizer.vocab_size
        config["pad_index"] = self.tokenizer.pad_token_id
        self.save_hyperparameters(config)

        if model_name == "longcontext":
            model = ChromatinModelLongContext(config, pretrained)
        else:
            model = ChromatinModel(config, pretrained)
        self.model = model

        #for testing finetuning with frozen pretrained part of the model
        """for p in self.model.dna_model.parameters():
            p.requires_grad = False"""

        self.criterion = BCELossMasked(pos_weight=torch.tensor(config["pos_weight"]))

        self.lr = config.get("learning_rate", 1e-3)
        self.weight_decay = config.get("weight_decay", 1e-4)

        self.num_epochs = config["num_epochs"]
        self.batch_size = config["batch_size"]

        #potentially move metric calculation to cpu
        self.auroc = torchmetrics.classification.AUROC(task='binary', compute_on_cpu=False)
        self.ap = torchmetrics.classification.AveragePrecision(task='binary', compute_on_cpu=False)
        self.mcc = torchmetrics.classification.MatthewsCorrCoef(task='binary', compute_on_cpu=False)

    def setup(self, stage=None):
        #self.model = torch.compile(self.model)
        pass

    def reset_metrics(self):
        self.cm = torch.zeros((2, 2), dtype=torch.int64, device=self.device)

        self.auroc.reset()
        self.ap.reset()
        self.mcc.reset()
    

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)


    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"][:, None, None, :]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)

        loss = self.criterion.compute_loss(logits, labels, attention_mask)
        self.log("train/loss", loss, sync_dist=True)
        self.log("train/loss_epoch", loss, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss

    def on_train_epoch_end(self):
        if self.global_rank == 0:
            mem_alloc = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[Epoch {self.current_epoch}] Alloc: {mem_alloc:.2f} GB | Reserved: {mem_reserved:.2f} GB")


    def _on_end_eval_step(self, name):
        assert name is not None

        cm_tokens_agg = self.cm.clone()

        if self.trainer.world_size > 1:
            torch.distributed.all_reduce(cm_tokens_agg, op=torch.distributed.ReduceOp.SUM)

        metrics_tokens = compute_metrics_from_cm(cm_tokens_agg, name=f"{name}_tokens")
        self.log_dict({**metrics_tokens}, sync_dist=True)
    

    def on_validation_epoch_start(self):
        self.reset_metrics()

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"][:, None, None, :]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)

        if self.trainer.sanity_checking:
            print(input_ids.shape)
            print(logits.shape)

        loss = self.criterion.compute_loss(logits, labels, attention_mask)
        self.log("val/loss", loss, sync_dist=True)

        preds = (torch.sigmoid(logits) > 0.5).long()

        mask = attention_mask[:, 0, 0, :].detach().bool()
        preds_masked = preds[mask].detach()
        y_masked = labels[mask].detach().long()

        update_binary_cm(preds_masked, y_masked, self.cm)

        return loss
    
    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.reset_metrics()
            return

        self._on_end_eval_step("val")


    def on_test_epoch_start(self):
        print("Test metrics reset")
        self.reset_metrics()

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"][:, None, None, :]
        labels = batch["labels"]
        
        logits = self(input_ids, attention_mask)
        preds = (torch.sigmoid(logits) > 0.5).long()

        mask = attention_mask[:, 0, 0, :].detach().bool()
        preds_masked = preds[mask].detach()
        y_masked = labels[mask].detach().long()

        update_binary_cm(preds_masked, y_masked, self.cm)

        logits_masked = logits[mask].detach()
    
        self.auroc.update(logits_masked, y_masked)
        self.ap.update(logits_masked, y_masked)
        self.mcc.update(preds_masked, y_masked)
    
    def on_test_epoch_end(self):
        self._on_end_eval_step("test")

        auroc_score = self.auroc.compute()
        ap_score = self.ap.compute()
        mcc_score = self.mcc.compute()
        
        self.log("test/auroc", auroc_score, on_epoch=True, sync_dist=True)
        self.log("test/ap", ap_score, on_epoch=True, sync_dist=True)
        self.log("test/mcc", mcc_score, on_epoch=True, sync_dist=True)
    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.weight_decay)

        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=3, verbose=True),
            'monitor': 'train/loss_epoch',
            'interval': 'epoch',
            'frequency': 1
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
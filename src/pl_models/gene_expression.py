import torch
import torch.nn as nn
import pytorch_lightning as pl

from model.model import GeneExpressionModelLongContext
from pl_models.chromatin import update_binary_cm, compute_metrics_from_cm

from data.tokenizer.tokenizer import DNATokenizer

from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchmetrics


class GeneLightningModule(pl.LightningModule):
    def __init__(self, config, parser_args, tokenizer=None, pretrained=None):
        super().__init__()

        if tokenizer is None:
            self.tokenizer = DNATokenizer(chromatin_tokens=parser_args.get("chromatin_tokens_used", False))
        else:
            self.tokenizer = tokenizer
        config["vocab_size"] = self.tokenizer.vocab_size
        config["pad_index"] = self.tokenizer.pad_token_id
        self.save_hyperparameters(config)

        self.model = GeneExpressionModelLongContext(config, pretrained)

        self.criterion =  nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config["pos_weight"]))
        #self.criterion = nn.PoissonNLLLoss(log_input=False, full=False) #for future use on coverage tracks
 
        self.lr = config.get("learning_rate", 1e-3)
        self.weight_decay = config.get("weight_decay", 1e-4)

        self.num_epochs = config["num_epochs"]
        self.batch_size = config["batch_size"]

        #potentially move metric calculation to cpu
        self.auroc = torchmetrics.classification.AUROC(task='binary', compute_on_cpu=False)
        self.ap = torchmetrics.classification.AveragePrecision(task='binary', compute_on_cpu=False)
        self.mcc = torchmetrics.classification.MatthewsCorrCoef(task='binary', compute_on_cpu=False)

        #self.pcc = torchmetrics.PearsonCorrCoef(num_outputs=1)

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

        #assert (labels >= 0).all(), "Poisson labels must be >= 0"

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)

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

        loss = self.criterion(logits, labels)
        self.log("val/loss", loss, sync_dist=True)

        preds = (torch.sigmoid(logits) > 0.5).long()

        update_binary_cm(preds.detach(), labels.detach(), self.cm)

        return loss
    
    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.reset_metrics()
            return

        self._on_end_eval_step("val")


    def on_test_epoch_start(self):
        print("Is model in train mode?", self.training)

        self.reset_metrics()

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"][:, None, None, :]
        labels = batch["labels"]
        
        logits = self(input_ids, attention_mask)

        preds = (torch.sigmoid(logits) > 0.5).long()
        probs = torch.sigmoid(logits.detach())

        update_binary_cm(preds.detach(), labels.detach(), self.cm)

        labels = labels.long()
    
        self.auroc.update(probs.detach(), labels.detach())
        self.ap.update(probs.detach(), labels.detach())
        self.mcc.update(preds.detach(), labels.detach())
    
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
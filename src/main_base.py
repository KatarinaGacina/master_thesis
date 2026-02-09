import torch
import torch.nn as nn
import torch.optim as optim

import torch._dynamo

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger

from configs.config import get_config_base
from configs.parser import get_parser_base

from data.base.datamodule import RepresentationsDataModule
from pl_models.base import BaseLightningModule

from pytorch_lightning import seed_everything

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser_args = get_parser_base()
    config_params = get_config_base()

    custom_seed=42
    seed_everything(custom_seed, workers=True, verbose=True) #for comparison

    datamodule = RepresentationsDataModule(config_params, parser_args, data_seed=42)
    model = BaseLightningModule(config_params, parser_args["model"])

    if torch.cuda.is_available():
        accelerator = "gpu"

        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            devices = list(range(n_gpus))
            strategy = "ddp"
        else:
            devices = 1 
            strategy = "auto"
    else:
        accelerator = "cpu"
        devices = 1
        strategy = "auto"

    wandb_logger = WandbLogger(
            name=config_params["wandb_name"],
            save_dir="/wandb",
            project=config_params["wandb_project"],
    )
    wandb_logger.watch(model.model, log="all")
    
    checkpoint_model_callback = ModelCheckpoint(
        dirpath=parser_args["checkpoint_path"],
        filename="epoch-{epoch:04d}",
        save_top_k=-1,
        every_n_epochs=10,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        patience=3,
        verbose=True,
        mode='min',
        check_on_train_epoch_end=True
    )

    trainer = pl.Trainer(
        max_epochs=config_params["num_epochs"],
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=wandb_logger,
        log_every_n_steps=1,
        enable_checkpointing=True,
        callbacks=[checkpoint_model_callback, early_stop_callback],
        gradient_clip_val=1.0,
        use_distributed_sampler=False
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
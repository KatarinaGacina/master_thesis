import torch
import torch.nn as nn
import torch.optim as optim

import torch._dynamo

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger

from configs.config import get_config_train_gene
from configs.parser import get_parser_args_gene

from data.gene_expression.datamodule import GeneDataModule
from pl_models.gene_expression import GeneLightningModule

from pytorch_lightning import seed_everything

import warnings
#warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser_args = get_parser_args_gene()
    config_params = get_config_train_gene()

    custom_seed=42
    seed_everything(custom_seed, workers=True, verbose=True) #for comparison

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    pretrained = parser_args.get("pretrained", None)

    datamodule = GeneDataModule(config_params, parser_args)
    model = GeneLightningModule(config_params, parser_args, pretrained=pretrained)

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
            #save_code=True
    )
    wandb_logger.watch(model.model, log="all")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=parser_args["checkpoint_path"],
        filename="epoch-{epoch:04d}",
        save_top_k=-1,
        every_n_epochs=2,
        save_last=True
    )
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        patience=2,
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
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=1.0,
        use_distributed_sampler=False
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
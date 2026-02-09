import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger

from configs.config import get_config_train
from configs.parser import get_parser_args

from data.chromatin.datamodule import ChromDataModule
from pl_models.chromatin import ChromatinLightningModule

from pytorch_lightning import seed_everything

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser_args = get_parser_args()
    config_params = get_config_train()

    custom_seed=42
    seed_everything(custom_seed, workers=True, verbose=True) #for comparison

    pretrained = parser_args.get("pretrained", None)

    datamodule = ChromDataModule(config_params, parser_args, data_seed=custom_seed, only_m=bool(pretrained))

    model = ChromatinLightningModule.load_from_checkpoint(
        "/checkpoints/experiment_name/last.ckpt",
        config=config_params,
        model_name=parser_args["model"], 
        pretrained=pretrained,
    )

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

    trainer = pl.Trainer(
        max_epochs=config_params["num_epochs"],
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=wandb_logger,
        log_every_n_steps=1,
        enable_checkpointing=False,
        gradient_clip_val=1.0,
        use_distributed_sampler=False
    )

    trainer.test(model, datamodule=datamodule)
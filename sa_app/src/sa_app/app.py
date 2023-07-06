import argparse

import pytorch_lightning as pl
import torch
import wandb
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from sa_app.common.utils import init_model_loggers
from sa_app.data.data import SentimentIterableDataset
from sa_app.models.model import Model
from sa_app.training.lightning_model_wrapper import LightningModelWrapper
from transformers import AutoConfig, AutoTokenizer


def init_model_callbacks(training_params: dict) -> list:
    model_checkpoint = ModelCheckpoint(
        dirpath=training_params["callbacks"]["dirpath"],
        filename="best_model",
        monitor=training_params["callbacks"]["monitor_var"],
        mode=training_params["callbacks"]["monitor_var_mode"],
        save_top_k=3,  # Save the top 3 models based on the monitored metric
    )
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        model_checkpoint,
    ]

    return callbacks


def train(config: dict, device: str, training_params: dict, dataset_params: dict, seed: int):
    # Global seeding
    pl.seed_everything(seed=seed)

    # load HF configs
    model_config = AutoConfig.from_pretrained(training_params.get("base-model-name"))

    # Load model (downloads from hub for the first time)
    model = Model.from_config(training_params.get("train_mode"))
    model = model.from_pretrained(training_params.get("base-model-name"), config=model_config)
    for params in model.parameters():
        params.requires_grad = False

    for params in model.classifier.parameters():
        params.requires_grad = True

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(training_params.get("base-model-name"))

    # Load streaming dataset
    train_dataset = SentimentIterableDataset(
        dataset_params.get("raw_dataset_file"),
        tokenizer,
        split_type="train",
        preprocessors=dataset_params.get("preprocessors"),
    )
    valid_dataset = SentimentIterableDataset(
        dataset_params.get("raw_dataset_file"),
        tokenizer,
        split_type="valid",
        preprocessors=dataset_params.get("preprocessors"),
    )

    # Loggers
    loggers = init_model_loggers(training_params["logging"]["log_dir"])

    # PL callbacks
    callbacks = init_model_callbacks(training_params)

    # PL wrapper
    model_wrapped = LightningModelWrapper(
        model=model,
        optimizer_params=training_params["optimizer"],
        lr_scheduler_params=training_params["lr_scheduler"] if "lr_scheduler" in training_params else None,
        unique_config=config,
    )

    # Get available devices
    try:
        devices = [int(device.split(":")[-1])]
        accelerator = "gpu"
    except ValueError:
        devices = "auto"
        accelerator = "cpu"

    # Trainer initialization
    trainer = pl.Trainer(
        logger=loggers,
        devices=devices,
        accelerator=accelerator,
        callbacks=callbacks,
        **training_params["trainer"],
    )

    # Initiate trainer
    trainer.fit(model=model_wrapped, train_dataloaders=train_dataset, val_dataloaders=valid_dataset, ckpt_path="last")

    wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="sa_app/app_cfg.yml",
        type=str,
        help="Path to config",
    )
    parser.add_argument(
        "--mode",
        default="train",
        type=str,
        help="mode of execution",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == "train":
        train(config=config, device=device, **config)
    # else:
    #     inference(config=config, device=device, **config)


if __name__ == "__main__":
    main()

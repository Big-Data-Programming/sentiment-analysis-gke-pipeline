import argparse

import pytorch_lightning as pl
import torch
import wandb
import yaml
from transformers import AutoTokenizer, BertConfig

from common.utils import init_model_loggers
from sa_train.sa_train_module.data.data import SentimentIterableDataset
from sa_train.sa_train_module.models.model import Model
from sa_train.sa_train_module.training.lightning_model_wrapper import (
    LightningModelWrapper,
)


def train(device: str, training_params: dict, dataset_params: dict, seed: int, config: dict):
    # Global seeding
    pl.seed_everything(seed=seed)

    # load HF configs
    model_config = BertConfig.from_pretrained(training_params.get("base-model-name"))

    # Load model (downloads from hub for the first time)
    model = Model.from_config(training_params.get("train_mode"))
    model = model.from_pretrained(training_params.get("base-model-name"), model_config)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(training_params.get("base-model-name"))

    # Load streaming dataset
    train_dataset = SentimentIterableDataset(dataset_params.get("raw_dataset_file"), tokenizer, split_type="train")
    valid_dataset = SentimentIterableDataset(dataset_params.get("raw_dataset_file"), tokenizer, split_type="valid")

    # Loggers
    loggers = init_model_loggers(training_params["logging"]["log_dir"])

    model.to(device)

    model_wrapped = LightningModelWrapper(
        model=model,
        optimizer_params=training_params["optimizer"],
        lr_scheduler_params=training_params["lr_scheduler"] if "lr_scheduler" in training_params else None,
    ).to(device)

    trainer = pl.Trainer(
        logger=loggers,
        devices=device,
        **training_params["trainer"],
    )

    trainer.fit(model=model_wrapped, train_dataloaders=train_dataset, val_dataloaders=valid_dataset, ckpt_path="last")

    wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="sa_train/train_cfg.yml",
        type=str,
        help="Path to config",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(device=device, **config)


if __name__ == "__main__":
    main()

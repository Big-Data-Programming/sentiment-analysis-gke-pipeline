import os
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import wandb
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sa_app.common.utils import init_model_loggers, parse_args
from sa_app.data.data import InitializeDataset, SentimentIterableDataset
from sa_app.models.model import Model
from sa_app.training.lightning_model_wrapper import LightningModelWrapper
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def init_model_callbacks(training_params: dict) -> List[LearningRateMonitor | ModelCheckpoint]:
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


def load_model(training_params: Dict) -> Tuple[Dict, AutoModelForSequenceClassification]:
    # load HF configs
    model_config = AutoConfig.from_pretrained(training_params.get("base-model-name"))

    # Load model (downloads from hub for the first time)
    model = Model.from_config(training_params.get("train_mode"))
    model = model.from_pretrained(training_params.get("base-model-name"), config=model_config)
    for params in model.parameters():
        params.requires_grad = False

    for params in model.classifier.parameters():
        params.requires_grad = True
    return model_config, model


def get_dataset(
    dataset_params: Dict, tokenizer: AutoTokenizer
) -> Tuple[SentimentIterableDataset, SentimentIterableDataset]:
    dataset_obj = InitializeDataset(dataset_params)
    raw_dataset_file, _ = dataset_obj()
    # Load streaming dataset
    train_dataset = SentimentIterableDataset(
        raw_dataset_file,
        tokenizer,
        split_type="train",
        preprocessors=dataset_params.get("preprocessors"),
    )
    valid_dataset = SentimentIterableDataset(
        raw_dataset_file,
        tokenizer,
        split_type="valid",
        preprocessors=dataset_params.get("preprocessors"),
    )
    return train_dataset, valid_dataset


def get_trainer(
    loggers: List[WandbLogger], devices: List[str] | str, accelerator: str, callbacks: List, training_params: Dict
) -> pl.Trainer:
    trainer = pl.Trainer(
        logger=loggers,
        devices=devices,
        accelerator=accelerator,
        callbacks=callbacks,
        **training_params["trainer"],
    )
    return trainer


def train(
    config: dict, device: list[str], training_params: Dict, dataset_params: Dict, seed: int, inference_params: Dict
):
    # wandb login
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    # Global seeding
    pl.seed_everything(seed=seed)
    model_config, model = load_model(training_params)
    tokenizer = AutoTokenizer.from_pretrained(training_params.get("base-model-name"))
    train_dataset, valid_dataset = get_dataset(dataset_params, tokenizer)
    loggers = init_model_loggers(dataset_params, training_params)
    callbacks = init_model_callbacks(training_params)

    # PL wrapper
    model_wrapped = LightningModelWrapper(
        model=model,
        optimizer_params=training_params["optimizer"],
        lr_scheduler_params=training_params["lr_scheduler"] if "lr_scheduler" in training_params else None,
        unique_config=config,
    ).to(device)

    # Get available devices
    if "cuda" in device:
        devices = [int(device.split(":")[-1])]
        accelerator = "gpu"
    else:
        devices = "auto"
        accelerator = "cpu"

    # Trainer initialization
    trainer = get_trainer(loggers, devices, accelerator, callbacks, training_params)

    # Initiate trainer
    trainer.fit(model=model_wrapped, train_dataloaders=train_dataset, val_dataloaders=valid_dataset, ckpt_path="last")

    with wandb.init(project=dataset_params["wandb_storage"]["wandb_project_name"]) as run:
        best_model = wandb.Artifact(
            f"{training_params['wandb_storage']['artifact_name']}_{run.id}",
            type=training_params["wandb_storage"]["artifact_type"],
        )
        best_model.add_file(callbacks[1].best_model_path)
        run.log_artifact(best_model)
        run.link_artifact(best_model, training_params["wandb_storage"]["register_to"])

    wandb.finish()


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device = "cuda:3"
    if args.mode == "train":
        train(config=config, device=device, **config)
    else:
        raise NotImplementedError(f"Method {args.mode} not implemented")


if __name__ == "__main__":
    main()

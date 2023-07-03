import argparse

import torch
import yaml
from transformers import AutoTokenizer

from sa_train.sa_train_module.models.model import Model


def train(device: str, training_params: dict, config: dict):
    model = Model.from_config(training_params.get("train_mode"))
    model = model.from_pretrained(training_params.get("base-model-name"))

    tokenizer = AutoTokenizer.from_pretrained(training_params.get("base-model-name"))

    model.to(device)


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

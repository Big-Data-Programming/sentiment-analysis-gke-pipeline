import argparse
import csv
from typing import Dict, List

from pytorch_lightning.loggers import WandbLogger


def init_model_loggers(dataset_params: Dict, training_params: Dict) -> List[WandbLogger]:
    wandb_project_name = dataset_params.get("wandb_storage").get("wandb_project_name")
    log_dir = training_params.get("logging").get("log_dir")
    wandb_logger = WandbLogger(project=wandb_project_name, save_dir=log_dir)
    return [
        wandb_logger,
    ]


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


def load_mapping(label_files_path) -> List[str]:
    with open(label_files_path) as labels_file:
        html = labels_file.read().split("\n")
        csvreader = csv.reader(html, delimiter="\t")
    return [row[1] for row in csvreader if len(row) > 1]

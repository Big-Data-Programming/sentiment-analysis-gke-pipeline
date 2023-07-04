from pytorch_lightning.loggers import WandbLogger


def init_model_loggers(log_dir: str) -> list:
    wandb_logger = WandbLogger(project="sa-roberta", save_dir=log_dir)
    return [
        wandb_logger,
    ]

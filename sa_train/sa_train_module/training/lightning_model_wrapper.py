from enum import Enum
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
from torchmetrics import MeanMetric
from transformers import PreTrainedModel

from sa_train.sa_train_module.training.optmizer import LearningRateScheduler, Optimizer


class Split(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

    def __str__(self):
        return self.value


class LightningModelWrapper(pl.LightningModule):
    def __init__(
        self,
        model: PreTrainedModel,
        optimizer_params: Optional[dict] = None,
        lr_scheduler_params: Optional[dict] = None,
        unique_config: Optional[dict] = None,
    ):
        super().__init__()

        self.model = model

        # Optional: For Training only
        self.optimizer_params = optimizer_params
        self.lr_scheduler_params = lr_scheduler_params
        self.unique_config = unique_config

        # metrics
        # self.train_acc = Accuracy(task="binary", ignore_index=-100)
        self.train_loss = MeanMetric()
        self.valid_loss = MeanMetric()

        self.train_seq_order_acc = MeanMetric()  # Accuracy(task="multiclass", ignore_index=-100, num_classes=10)
        self.train_mlm_acc = MeanMetric()
        self.train_total_loss = MeanMetric()
        self.train_mlm_loss = MeanMetric()
        self.train_seq_order_loss = MeanMetric()
        self.train_nsp_loss = MeanMetric()
        self.train_nsp_acc = MeanMetric()

        self.valid_seq_order_acc = MeanMetric()  # Accuracy(task="multiclass", ignore_index=-100, num_classes=10)
        self.valid_mlm_acc = MeanMetric()
        self.valid_total_loss = MeanMetric()
        self.valid_mlm_loss = MeanMetric()
        self.valid_seq_order_loss = MeanMetric()
        self.valid_nsp_loss = MeanMetric()
        self.valid_nsp_acc = MeanMetric()

        self.valid_global_step = 0
        self.evaluator = None

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Optimizer.from_config(params=self.parameters(), **self.optimizer_params)

        scheduler = None
        if self.lr_scheduler_params is not None:
            self.trainer.fit_loop.setup_data()
            # total_devices = self.trainer.num_devices * self.trainer.num_nodes
            # train_batches = len(self.trainer.train_dataloader) // total_devices
            # train_steps = (self.trainer.max_epochs * train_batches) // self.trainer.accumulate_grad_batches
            train_steps = self.trainer.max_steps
            lr_warmup = self.lr_scheduler_params.pop("lr_warmup", 0.0)
            interval = self.lr_scheduler_params.pop("interval", "epoch")
            lr_scheduler = LearningRateScheduler.from_config(
                optimizer=optimizer,
                num_warmup_steps=lr_warmup * train_steps,
                num_training_steps=train_steps,
                **self.lr_scheduler_params,
            )

            scheduler = {
                "scheduler": lr_scheduler,
                "interval": interval,
                "frequency": 1,
                "strict": False,
                "monitor": "loss",
            }

        if scheduler:
            return [optimizer], [scheduler]
        else:
            return optimizer

    def save_hf_checkpoint(self, path: Union[str, Path]) -> None:
        """Save the model using the original HF AutoModel.

        This is useful for when you'd like to export the model to the hub.
        Args:
            path: Path to save the model to.
        """
        self.model.save_pretrained(path)

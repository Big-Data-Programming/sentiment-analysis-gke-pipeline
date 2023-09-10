from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
import torch
from sa_app.training.optmizer import LearningRateScheduler, Optimizer
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy, MeanMetric, Metric
from transformers import AutoModelForSequenceClassification


class Split(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

    def __str__(self):
        return self.value


class LightningModelWrapper(pl.LightningModule):
    def __init__(
        self,
        model: AutoModelForSequenceClassification,
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
        self.num_classes = unique_config.get("training_params").get("num_classes")
        self.task_name = "multiclass" if self.num_classes > 2 else "binary"
        self.loss_fn = CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=1)

        self.train_loss = MeanMetric()
        self.train_acc = Accuracy(task=self.task_name, num_classes=self.num_classes)

        self.valid_loss = MeanMetric()
        self.valid_acc = Accuracy(task=self.task_name, num_classes=self.num_classes)

    def forward(self, x):
        labels, sentence_batch = x
        [sentence_batch[inp_key].to(self.model.device) for inp_key, value in sentence_batch.items()]
        output = self.model(**sentence_batch)
        output.logits = self.softmax(output.logits)
        output.loss = self.loss_fn(output.logits, labels)
        return {"logits": output.logits, "loss": output.loss}

    def training_step(self, batch: dict, batch_idx: int) -> Dict:
        return self.forward(batch)

    def on_train_batch_end(self, step_output: Dict, batch: Any, batch_idx: int) -> None:
        self.log_batch_end(
            step_output=step_output, batch=batch, loss_fn=self.train_loss, acc_fn=self.train_acc, split_type=Split.TRAIN
        )

    def validation_step(self, batch: dict, batch_idx: int) -> Dict:
        return self.forward(batch)

    def on_validation_batch_end(self, step_output: Dict, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.log_batch_end(
            step_output=step_output, batch=batch, loss_fn=self.valid_loss, acc_fn=self.valid_acc, split_type=Split.VALID
        )

    def on_train_epoch_end(self) -> None:
        self.log_epoch_end(loss_fn=self.train_loss, split_type=Split.TRAIN)

    def on_validation_epoch_end(self) -> None:
        self.log_epoch_end(loss_fn=self.valid_loss, split_type=Split.VALID)

    def log_batch_end(self, step_output: dict, batch: dict, loss_fn: Metric, acc_fn: Metric, split_type: Split):
        loss = step_output.get("loss")
        loss_fn.update(loss)
        self.log(f"{split_type}_loss_step", loss, batch_size=self.trainer.train_dataloader.batch_size)

        labels, sentence_batch = batch
        predicted_labels = torch.argmax(step_output.get("logits"), dim=1)
        self.log(
            f"{split_type}_acc_step",
            acc_fn(predicted_labels, labels),
            batch_size=self.trainer.train_dataloader.batch_size,
        )

    def log_epoch_end(self, loss_fn: Metric, split_type: Split):
        loss_fn = loss_fn.compute() if loss_fn.mean_value != 0 else None
        if loss_fn is not None:
            self.log(f"{split_type}-loss-epoch", loss_fn)

    def configure_optimizers(self):
        optimizer = Optimizer.from_config(params=self.parameters(), **self.optimizer_params)

        scheduler = None
        if self.lr_scheduler_params is not None:
            self.trainer.fit_loop.setup_data()
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

    def save_model(self, path: Union[str, Path]) -> None:
        """Save the model using the original HF AutoModel.

        This is useful for when you'd like to export the model to the hub.
        Args:
            path: Path to save the model to.
        """
        self.model.save_pretrained(path)

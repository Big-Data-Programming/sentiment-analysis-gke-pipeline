import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

MODEL = {"fine_tune_base_model": AutoModelForSequenceClassification}


class CustomClassificationHead(nn.Module):
    def __init__(self, input_dim: int = 768, num_labels: int = 3):
        super(CustomClassificationHead, self).__init__()
        self.dense = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.out_proj = nn.Linear(input_dim, num_labels)

    def forward(self, features):
        x = self.dense(features)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model:
    def __init__(self):
        super().__init__()

    @classmethod
    def from_config(cls, mode: str):
        try:
            class_ = MODEL[mode]
        except KeyError:
            raise KeyError(f'Mode type "{mode}" is not implemented.')

        return class_

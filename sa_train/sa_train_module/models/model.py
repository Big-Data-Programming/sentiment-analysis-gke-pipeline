from transformers import AutoModelForSequenceClassification

MODEL = {"fine_tune_base_model": AutoModelForSequenceClassification}


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

from typing import Dict

import torch
import yaml
from sa_app.common.utils import load_mapping, parse_args
from sa_app.data.data import InitializeDataset
from sa_app.data.data_cleaner import StackedPreprocessor
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class InferenceEngine:
    def __init__(self, inference_params: Dict, training_params: Dict, dataset_params: Dict, device: str):
        # TODO : Download artifacts from wandb
        # Set the device to CPU or GPU
        self.device = device

        # Load the trained model and tokenizer
        self.model_path = inference_params["model_dir"]
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(training_params["tokenizer"])
        self.preprocessors = StackedPreprocessor(dataset_params["preprocessors"])
        dataset_obj = InitializeDataset(dataset_params)
        _, label_mapping_path = dataset_obj()
        self.label_mapping = load_mapping(label_mapping_path)

    def perform_inference(self, sentence):
        # Preprocess the input sentence
        sentence = self.preprocessors([sentence])[0]

        # Tokenize the input sentence
        inputs = self.tokenizer(sentence, truncation=True, padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)

        # Forward pass through the model
        outputs = self.model(**inputs)

        # Get the predicted labels
        predicted_labels = torch.argmax(outputs.logits, dim=1)

        return self.label_mapping[predicted_labels.tolist()[0]]


if __name__ == "__main__":
    args = parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    device_in_use = "cuda" if torch.cuda.is_available() else "cpu"
    # Example usage
    ie_obj = InferenceEngine(
        inference_params=config["inference_params"],
        training_params=config["training_params"],
        dataset_params=config["dataset_params"],
        device=device_in_use,
    )
    input_sentence = "I feel so bad today . Such a bad day :( "
    predicted_labels = ie_obj.perform_inference(input_sentence)
    print("Predicted labels:", predicted_labels)

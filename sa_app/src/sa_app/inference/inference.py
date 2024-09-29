# import os
from typing import Dict

import torch

# import wandb
import yaml
from sa_app.common.utils import parse_args
from sa_app.data.data_cleaner import StackedPreprocessor
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


class InferenceEngine:
    def __init__(
        self,
        inference_params: Dict,
        training_params: Dict,
        dataset_params: Dict,
        device: str,
    ):
        # TODO : Download artifacts from wandb
        # Set the device to CPU or GPU
        self.device = device

        # Load the trained model and tokenizer
        self.model, self.model_config = self.get_wandb_model(inference_params)
        self.tokenizer = AutoTokenizer.from_pretrained(training_params["tokenizer"])

        # Preprecessor
        self.preprocessors = StackedPreprocessor(dataset_params["preprocessors"])

        # Dataset initialization
        # dataset_obj = InitializeDataset(dataset_params)
        # _, label_mapping_path = dataset_obj()
        # print(label_mapping_path)
        # self.label_mapping = load_mapping(label_mapping_path)
        self.label_mapping = ["negative", "positive"]

    def get_wandb_model(self, inference_params):
        # Download checkpoint from wandb
        # run = wandb.init()
        # artifact = run.use_artifact(inference_params["model_dir"], type="model")
        # artifact_dir = artifact.download()

        # Load checkpoint into the pretrained model
        # checkpoint_pth = os.path.join(artifact_dir, inference_params["default_model_name"])
        # state_dict = torch.load(checkpoint_pth)
        # new_state_dict = {}
        # for key in state_dict["state_dict"].keys():
        #     if key.startswith("model."):
        #         new_key = key.replace("model.", "")  # Remove 'model.' prefix
        #         new_state_dict[new_key] = state_dict["state_dict"][key]

        config = AutoConfig.from_pretrained(inference_params["base_model_name"])  # TODO: Remove hardcoding
        model = AutoModelForSequenceClassification.from_config(config).to(self.device)
        # model.load_state_dict(new_state_dict)

        return model, config

    def perform_inference(self, sentence):
        # Preprocess the input sentence
        sentence = self.preprocessors([sentence])[0]

        # Tokenize the input sentence
        inputs = self.tokenizer(sentence, truncation=True, padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)

        # Eval mode
        self.model.eval()

        # Forward pass through the model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get the predicted labels
        probs = torch.softmax(outputs[0][0], dim=-1)
        predicted_labels = torch.argmax(probs, dim=-1)

        return self.model_config.id2label[predicted_labels.tolist()]


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
    # input_sentence = "I feel so bad today . Such a bad day :( "
    # input_sentence = """
    # @swikey haha, okay, i feel much better now.
    # let's just dye our hair paramore red!
    # """
    input_sentence = "Covid cases are increasing fast!"
    predicted_labels = ie_obj.perform_inference(input_sentence)
    print("Predicted labels:", predicted_labels)

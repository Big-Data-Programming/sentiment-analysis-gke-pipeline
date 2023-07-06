import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the trained model and tokenizer
model_path = "path/to/your/trained/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Set the device to CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def perform_inference(sentence):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, truncation=True, padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    # Forward pass through the model
    outputs = model(**inputs)

    # Get the predicted labels
    predicted_labels = torch.argmax(outputs.logits, dim=1)

    return predicted_labels.tolist()


# Example usage
input_sentence = "This is a test sentence."
predicted_labels = perform_inference(input_sentence)
print("Predicted labels:", predicted_labels)

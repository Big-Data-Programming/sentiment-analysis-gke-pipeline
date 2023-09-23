import requests
import torch
import yaml
from flask import Flask, jsonify, request
from sa_app.common.utils import parse_args
from sa_app.inference.inference import InferenceEngine

app = Flask(__name__)
args = parse_args()
config = yaml.safe_load(open(args.config, "r"))
device_in_use = "cuda" if torch.cuda.is_available() else "cpu"
ie_obj = InferenceEngine(
    inference_params=config["inference_params"],
    training_params=config["training_params"],
    dataset_params=config["dataset_params"],
    device=device_in_use,
)


@app.route("/sentiment_analysis", methods=["POST"])
def get_sentiment():
    new_tweet = {"id": request.json["id"], "tweet_content": request.json["tweet_content"]}
    result_data = {"result": ie_obj.perform_inference(new_tweet["tweet_content"])}
    db_update_status = requests.post(
        "http://mongo-writer-service:5002/update_tweet_sentiment",
        json={"u_id": new_tweet["id"], "sentiment_value": result_data["result"]},
    )
    result_data["db_update_status"] = db_update_status.status_code
    return jsonify(result_data)


@app.route("/home")
def test_fn():
    return "Hello World!!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

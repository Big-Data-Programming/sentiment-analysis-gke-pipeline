import yaml
from flask import Flask, jsonify, request
from tweet_collection import service_insert_twitter_data, service_update_sentiment
from utils import mongo_global_init

app = Flask(__name__)
config = yaml.safe_load(open("container_src/app_cfg.yml", "r"))
mongo_global_init(**config["database_params"])


@app.route("/insert_tweet_data", methods=["POST"])
def insert_tweet_data():
    try:
        data = {"user_id": request.json["user_id"], "tweet_content": request.json["tweet_content"]}
        print(data)
        status_id = service_insert_twitter_data(data)
        if status_id:
            return jsonify({"result": status_id}), 200
        else:
            return jsonify({"result": "internal server error"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/update_tweet_sentiment", methods=["POST"])
def update_tweet_sentiment():
    try:
        u_id = request.json["u_id"]
        sentiment_value = request.json["sentiment_value"]
        status = service_update_sentiment(u_id, sentiment_value)
        if status["status"] == "success":
            return jsonify({"result": status["message"]}), 200
        else:
            return jsonify({"result": status["message"]}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/home")
def test_fn():
    return "Hello World!!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)

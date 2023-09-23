import datetime
from typing import Dict

import mongoengine


class TwitterDataCollection(mongoengine.Document):
    meta = {"db_alias": "core", "collection": "tweet_data"}
    user_id = mongoengine.StringField(required=True)
    tweet_content = mongoengine.StringField(required=True)
    created_at = mongoengine.DateTimeField(default=datetime.datetime.now)
    sentiment = mongoengine.StringField()


def service_insert_twitter_data(data: Dict):
    try:
        coll_obj = TwitterDataCollection()
        coll_obj.user_id = data["user_id"]
        coll_obj.tweet_content = data["tweet_content"]
        coll_obj.save()
        return str(coll_obj.id)
    except Exception as e:
        print(e)
        return False


def service_update_sentiment(u_id, sentiment_value):
    tweet_data = TwitterDataCollection.objects(id=u_id).first()
    if tweet_data:
        tweet_data.sentiment = sentiment_value
        tweet_data.save()

        return {"status": "success", "message": "Data updated successfully"}

    else:
        return {"status": "error", "message": "Record not found"}

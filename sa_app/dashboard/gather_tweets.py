import threading

import tweepy


class GatheringEngine(tweepy.StreamingClient):
    def __init__(self, secret_keys, rules_dict, tweet_thresh=5, sleep_after_tweet=1, wait_on_rate_limit=True):
        super().__init__(secret_keys["bearer_token"], wait_on_rate_limit=wait_on_rate_limit)
        self.secret_keys = secret_keys

        self.tweet_thresh = tweet_thresh
        self.tweet_counter = 0

        self.change_rule = False
        self.rules_dict = rules_dict
        self.sleep_after_tweet = sleep_after_tweet
        self.rules_names = list(self.rules_dict.keys())
        self.rule_counter = 0

    def monitor_thread(self):
        print("starting monitor thread to check threshold")
        th = threading.Thread(target=self.add_rule)
        th.start()
        return th

    def on_tweet(self, tweet):
        print(tweet.data["id"])
        data = {
            "tweet_id": str(tweet.data["id"]),
            "text": str(tweet.data["text"]),
            "tweet_timestamp": datetime.datetime.strftime(datetime.datetime.now(), "%d/%m/%Y %H:%M:%S"),
            "rule_name": self.rules_names[self.rule_counter],
        }
        if tweet.geo:
            print(f"location is : {tweet.geo}")
        string_data = json.dumps(data).encode("utf-8")
        self.publisher.publish(self.topic_path, string_data)
        print(f"Result published for : {tweet.data['id']}")

        # print("Publishing to confluent kafka")
        # self.confluent_producer.send(topic=self.topic_id, key=str(tweet.data['id']).encode('utf-8'),
        #                              value=string_data)
        # print("Published to confluent kafka")

        time.sleep(self.sleep_after_tweet)
        self.tweet_counter += 1
        if self.tweet_counter > self.tweet_thresh:
            print("Threshold reached, changing rule")
            self.change_rule = True
            self.tweet_counter = 0
            self.rule_counter += 1

    def add_rule(self):
        print("Adding the next rule\n")
        while True:
            if self.change_rule:
                self.clear_rules()
                rule = self.get_next_rule()
                if rule:
                    self.add_rules(rule)
                else:
                    print("End of all rules\n")
                    sys.exit(0)
            if self.rule_counter == "EXIT":
                print("End of all rules\n")
                sys.exit(0)

    def clear_rules(self):
        print("Removing existing rules")
        for rules_list in self.get_rules():
            if isinstance(rules_list, list):
                rule_ids = [rule.id for rule in rules_list]
                if rule_ids:
                    self.delete_rules(rule_ids)
                    print("deleted existing rules")
                    self.change_rule = False
                break

    def get_next_rule(self):
        print("Getting the next rule")
        if self.rule_counter >= len(self.rules_names):
            print("Resetting the next rule counter")
            return None
        return tweepy.StreamRule(self.rules_dict[self.rules_names[self.rule_counter]])

    def run(self):
        monitor_th = self.monitor_thread()
        self.clear_rules()

        print("\nAdding 1st rule")
        rule = self.get_next_rule()
        self.add_rules(rule)

        self.filter()
        monitor_th.join()

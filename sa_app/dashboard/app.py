import warnings

import pandas as pd  # read csv, df manipulation
import plotly.express as px
import requests
import streamlit as st  # 🎈 data web app development

warnings.filterwarnings("ignore", category=UserWarning, message="Can't initialize NVML")

# Initialize streamlit
st.set_page_config(
    page_title="Twitter Data Sentiment Analysis Dashboard (Demo)",
    page_icon="✅",
    layout="wide",
)


def get_sentiment(u_id, tweet_text):
    inference_url = "http://inference-service:5001/sentiment_analysis"
    data = {"id": u_id, "tweet_content": tweet_text}
    response = requests.post(inference_url, json=data)
    if response.status_code == 200:
        data = response.json()
        return data["result"]
    else:
        print(f"Request failed with status code {response.status_code}")


def insert_to_db(u_id, tweet_content):
    if u_id is not None:
        writer_url = "http://mongo-writer-service:5002/insert_tweet_data"
        db_update_status = requests.post(writer_url, json={"user_id": u_id, "tweet_content": tweet_content})
        if db_update_status.status_code == 200:
            return db_update_status.json()["result"]
        else:
            print(f"Request failed with status code {db_update_status.status_code}")
            return False
    else:
        print("Database update failed")


dataset_url = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vRp617o0JFqccSe3nDF90EyAtCwmshhtDfL"
    "-p4f60Gt63Fo2MEqrXWG2RIrN55RwmqRYFx16KsOPjR6/pub?gid=1359267733&single=true&output=csv"
)


def update_donut(sentiment_cnt_dict):
    sentiment_data = pd.DataFrame(
        {"Sentiment": list(sentiment_cnt_dict.keys()), "Count": list(sentiment_cnt_dict.values())}
    )
    fig = px.pie(sentiment_data, names="Sentiment", values="Count", hole=0.5)
    fig.update_traces(textinfo="percent+label", pull=[0.2, 0])
    st.plotly_chart(fig, use_container_width=False)


# read csv from a URL
@st.cache_data
def get_data_iterator() -> pd.DataFrame:
    df = pd.read_csv(dataset_url)
    # pick n sampled rows
    return df


# dashboard title
st.title("Sentiment Analysis Dashboard")

topic_limit = st.text_input(label="Number of tweets to be analysed", placeholder="Enter # of tweets to be analysed")
collect_btn = st.button("Start collecting")

# creating a single-element container
placeholder = st.empty()
tweet_count = 0
sentiment_cnt = {"positive": 0, "negative": 0}
label_mapping = {0: "negative", 4: "positive", 1: "positive"}

results = []

if collect_btn:
    df_iterator = get_data_iterator().sample(int(topic_limit))
    for _, row in df_iterator.iterrows():
        if row[5]:
            # Inserting the tweet to database
            u_id = insert_to_db(row[4], row[5])
            # Run model inference here
            sentiment_pred = get_sentiment(u_id, row[5])
            # sentiment_pred = label_mapping[row[0]]
            print(f"For {row[5]}, prediction is : {sentiment_pred}")
            tweet_count += 1

            if sentiment_pred is not None:
                sentiment_cnt[sentiment_pred] += 1

                with placeholder.container():
                    # create three columns
                    kpi1, kpi2, kpi3 = st.columns(3)

                    # fill in those three columns with respective metrics or KPIs
                    kpi1.metric(
                        label="Total tweets",
                        value=tweet_count,
                        delta=tweet_count,
                    )

                    kpi2.metric(label="Positive Count", value=sentiment_cnt["positive"])

                    kpi3.metric(label="Negative Count", value=sentiment_cnt["negative"])

                    results.append([row[5], label_mapping[row[0]]])

update_donut(sentiment_cnt)
st.table(results)

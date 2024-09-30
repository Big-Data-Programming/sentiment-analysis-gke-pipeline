from datetime import datetime

import pandas as pd  # read csv, df manipulation
import plotly.express as px

# import plotly.figure_factory as ff
import requests
import streamlit as st  # 🎈 data web app development

# from sklearn.metrics import confusion_matrix

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
        {
            "Sentiment": list(sentiment_cnt_dict.keys()),
            "Count": list(sentiment_cnt_dict.values()),
        }
    )
    fig = px.pie(
        sentiment_data,
        names="Sentiment",
        values="Count",
        hole=0.5,
        title="Sentiment Distribution",
    )
    fig.update_traces(textinfo="percent+label", pull=[0.2, 0])
    return fig


# read csv from a URL
@st.cache_data
def get_data_iterator() -> pd.DataFrame:
    df = pd.read_csv(dataset_url)
    # pick n sampled rows
    return df


# Function to convert timestamp to datetime
def convert_to_datetime(timestamp_str):
    return datetime.strptime(timestamp_str, "%a %b %d %H:%M:%S PDT %Y")


def update_user_sentiments_map(user_sentiments_df):
    # Create an interactive bar chart using Plotly
    fig = px.bar(
        user_sentiments_df,
        x="user_id",
        y="count",
        color="sentiment",
        title="User to Sentiments Map",
        barmode="group",
        labels={"user_id": "User ID", "count": "Count"},
        hover_data=["user_id", "count", "sentiment"],
    )
    fig.update_layout(
        xaxis={"categoryorder": "total descending"},
        yaxis_title="Number of Tweets",
        xaxis_title="User ID",
    )
    return fig


# dashboard title
st.title("Twitter Data Sentiment Analysis Dashboard (Demo)")

topic_limit = st.text_input(
    label="Number of tweets to be analysed",
    placeholder="Enter # of tweets to be analysed",
)
collect_btn = st.button("Start analysis")

# creating a single-element container
placeholder = st.empty()
tweet_count = 0
sentiment_cnt = {"positive": 0, "negative": 0, "neutral": 0}
label_mapping = {0: "negative", 1: "neutral", 2: "positive"}
label_mapping_raw_data = {0: "negative", 2: "neutral", 4: "positive"}
results = []
sentiment_time_series = []

if collect_btn:
    df_iterator = get_data_iterator().sample(int(topic_limit))
    for _, row in df_iterator.iterrows():
        if row[5]:
            # Inserting the tweet to database
            u_id = insert_to_db(row[4], row[5])
            # Run model inference here
            sentiment_pred = get_sentiment(u_id, row[5])

            print(f"For {row[5]}, prediction is : {sentiment_pred}")
            tweet_count += 1

            if sentiment_pred is not None:
                sentiment_cnt[sentiment_pred] += 1

                with placeholder.container():
                    # create three columns
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

                    # fill in those three columns with respective metrics or KPIs
                    kpi1.metric(
                        label="Total tweets",
                        value=tweet_count,
                        delta=tweet_count,
                    )

                    kpi2.metric(label="Positive Count", value=sentiment_cnt["positive"])

                    kpi3.metric(label="Negative Count", value=sentiment_cnt["negative"])

                    kpi4.metric(label="Neutral Count", value=sentiment_cnt["neutral"])

                    results.append([row[5], label_mapping_raw_data[row[0]], sentiment_pred])

            tweet_datetime = convert_to_datetime(row[2])
            sentiment_time_series.append([tweet_datetime.date(), sentiment_pred])


# Create columns for the donut and line charts
col1, col2 = st.columns(2)

with col1:
    donut_fig = update_donut(sentiment_cnt)
    st.plotly_chart(donut_fig, use_container_width=True)

with col2:
    if sentiment_time_series:
        df_sentiment_time_series = pd.DataFrame(sentiment_time_series, columns=["Date", "Sentiment"])
        sentiment_count_per_day = (
            df_sentiment_time_series.groupby(["Date", "Sentiment"]).size().reset_index(name="Count")
        )

        # Apply a moving average to smooth the 'Count' values
        sentiment_count_per_day["Smoothed Count"] = sentiment_count_per_day.groupby("Sentiment")["Count"].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )

        # Create a smoother line graph using the smoothed data
        line_fig = px.line(
            sentiment_count_per_day,
            x="Date",
            y="Smoothed Count",
            color="Sentiment",
            title="Smoothed Sentiment Count Over Time",
        )

        # Update the layout for better visualization
        line_fig.update_traces(mode="lines+markers")
        line_fig.update_layout(xaxis_title="Date", yaxis_title="Tweet Count", hovermode="x unified")

        # Display the plot
        st.plotly_chart(line_fig, use_container_width=True)

# if results:
#     # Create a DataFrame from the results list
#     df_results = pd.DataFrame(results, columns=["Tweet_Content", "Actual", "Prediction"])

#     # Compute the confusion matrix
#     cm = confusion_matrix(
#         df_results["Actual"],
#         df_results["Prediction"],
#         labels=["negative", "neutral", "positive"],
#     )

#     # Display labels
#     cm_labels = ["negative", "neutral", "positive"]

#     # Create a heatmap using plotly
#     fig_cm = ff.create_annotated_heatmap(
#         z=cm,
#         x=cm_labels,
#         y=cm_labels,
#         annotation_text=cm.astype(str),
#         colorscale="Blues",
#         showscale=True,
#     )

#     # Update layout for better visualization
#     fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")

#     # Display the confusion matrix using Streamlit
#     st.plotly_chart(fig_cm, use_container_width=True)


# For Debug
# st.table(pd.DataFrame(results, columns=["Tweet_Content", "Actual", "Prediction"]))

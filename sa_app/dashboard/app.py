import warnings

import pandas as pd  # read csv, df manipulation
import streamlit as st  # 🎈 data web app development
import torch
import yaml
from sa_app.inference.inference import InferenceEngine
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, message="Can't initialize NVML")


# Initialize streamlit
st.set_page_config(
    page_title="Twitter Data Sentiment Analysis Dashboard",
    page_icon="✅",
    layout="wide",
)


@st.cache_data
def initialize_inference_model():
    # Initialize the sentiment analysis application
    config = yaml.safe_load(
        open(
            "container_src/app_cfg.yml",
            "r",
        )
    )
    device_in_use = "cuda" if torch.cuda.is_available() else "cpu"
    ie_obj = InferenceEngine(
        inference_params=config["inference_params"],
        training_params=config["training_params"],
        dataset_params=config["dataset_params"],
        device=device_in_use,
    )
    return config, device_in_use, ie_obj


def get_sentiment():
    pass


config, device_in_use, ie_obj = initialize_inference_model()


dataset_url = "container_src/sample.csv"


# read csv from a URL
# @st.cache_data
def get_data_iterator() -> pd.DataFrame:
    return pd.read_csv(dataset_url, chunksize=1000)


df_iterator = get_data_iterator()

# dashboard title
st.title("Real-Time / Live Data Science Dashboard")

# top-level filters
job_filter = st.selectbox("Select topic", [pd.unique(df.iloc[:, 4]) for df in df_iterator][0])

cal_df_len = get_data_iterator()
cal_df_len = sum([len(df) for df in cal_df_len])

# creating a single-element container
placeholder = st.empty()

# dataframe filter
df_iterator = get_data_iterator()
tweet_count = 0
sentiment_cnt = {"positive": 0, "negative": 0}
sentiment_cnt_actual = {"positive": 0, "negative": 0}

pbar = tqdm(desc="Processing tweets", total=cal_df_len)

# near real-time / live feed simulation
for seconds in range(200):
    try:
        df = next(df_iterator)
        # df = df[df.iloc[:, 4] == job_filter]

        for _, row in df.iterrows():
            if row[5]:
                sentiment_pred = ie_obj.perform_inference(row[5])
                actual = ie_obj.label_mapping[row[0]]
                tweet_count += 1

                sentiment_cnt[sentiment_pred] += 1
                sentiment_cnt_actual[actual] += 1

                with placeholder.container():
                    # create three columns
                    kpi1, kpi2, kpi3 = st.columns(3)

                    # fill in those three columns with respective metrics or KPIs
                    kpi1.metric(
                        label="Total tweets",
                        value=tweet_count,
                        delta=tweet_count,
                    )

                    kpi2.metric(
                        label="Positive Count",
                        value=sentiment_cnt["positive"],
                        delta=-sentiment_cnt_actual["positive"],
                    )

                    kpi3.metric(
                        label="Negative Count",
                        value=sentiment_cnt["negative"],
                        delta=sentiment_cnt_actual["negative"],
                    )
            # time.sleep(1)

        pbar.update(len(df))

    except StopIteration:
        print("No more tweets!!!!")
        break

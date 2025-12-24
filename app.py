import json
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from model_utils import evaluate_models, save_metrics, load_dataset


METRICS_FILE = "metrics.json"
DATA_PATH = "data/heart.csv"


def get_models():
    return [
        ("LogisticRegression", LogisticRegression(max_iter=2000)),
        ("RandomForest", RandomForestClassifier(n_estimators=300, random_state=42)),
        ("GradientBoosting", GradientBoostingClassifier(random_state=42)),
        ("SVM_RBF", SVC(kernel="rbf", probability=True, random_state=42)),
    ]


def load_metrics():
    if not os.path.exists(METRICS_FILE):
        return None
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def metrics_to_df(metrics_json):
    return pd.DataFrame(metrics_json["metrics"]).set_index("model")


st.set_page_config(page_title="ML Model Comparison Dashboard", layout="wide")

st.title("ML Model Comparison Dashboard")
st.caption("Train multiple models, compare evaluation metrics, and retrain from the UI.")

with st.sidebar:
    st.header("Controls")
    st.write("Dataset:", DATA_PATH)

    retrain = st.button("Retrain Models", use_container_width=True)

    st.divider()
    st.subheader("Notes")
    st.write(
        "- Pipelines prevent data leakage.\n"
        "- Metrics are saved in `metrics.json`.\n"
        "- Trained models are saved in `models/`."
    )


# Retrain if button clicked OR metrics missing
if retrain or not os.path.exists(METRICS_FILE):
    with st.spinner("Training models and computing metrics..."):
        df = load_dataset(DATA_PATH)
        metrics_json = evaluate_models(df, models=get_models(), target_col=None)
        save_metrics(metrics_json, METRICS_FILE)
    st.success("Training complete. Metrics updated.")


metrics_json = load_metrics()
if metrics_json is None:
    st.error("No metrics found. Click 'Retrain Models' to generate metrics.")
    st.stop()

df_metrics = metrics_to_df(metrics_json)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Choose a model")
    model_names = df_metrics.index.tolist()
    selected = st.selectbox("Model", model_names)

    st.subheader("Evaluation metrics")
    st.dataframe(df_metrics.loc[[selected]].reset_index(), use_container_width=True)

with col2:
    st.subheader("Model comparison")
    metric = st.selectbox("Metric to compare", ["accuracy", "precision", "recall", "f1"], index=3)

    chart_df = df_metrics[[metric]].sort_values(metric, ascending=False)

    fig = plt.figure()
    plt.bar(chart_df.index, chart_df[metric])
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.title(f"{metric.upper()} by model")
    st.pyplot(fig, clear_figure=True)

st.divider()
st.subheader("All metrics table")
st.dataframe(df_metrics.reset_index(), use_container_width=True)
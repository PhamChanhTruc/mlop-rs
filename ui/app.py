import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_EVENTS_PATH = REPO_ROOT / "data/processed/events_retailrocket.parquet"
DEFAULT_CANDIDATE_LIMIT = 20


@st.cache_data(show_spinner=False)
def load_default_candidate_ids(limit: int = DEFAULT_CANDIDATE_LIMIT) -> list[int]:
    if not PROCESSED_EVENTS_PATH.exists():
        return [101, 102, 103]

    events = pd.read_parquet(PROCESSED_EVENTS_PATH, columns=["item_id"])
    top_items = (
        events["item_id"]
        .value_counts()
        .head(limit)
        .index.astype(int)
        .tolist()
    )
    return top_items or [101, 102, 103]


def post_json(path: str, payload: dict) -> tuple[int, dict]:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=30)
    try:
        body = response.json()
    except ValueError:
        body = {"raw_text": response.text}
    return response.status_code, body


def parse_candidate_ids(raw_value: str) -> list[int]:
    values = []
    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("Provide at least one candidate item ID.")
    return values


st.set_page_config(page_title="MLOps-RS Demo UI", page_icon=":chart_with_upwards_trend:", layout="wide")

st.title("MLOps-RS Demo UI")
st.caption("Lightweight thesis demo UI for purchase probability prediction and top-K recommendation.")

with st.sidebar:
    st.subheader("API")
    st.write(f"Base URL: `{API_BASE_URL}`")
    if st.button("Check API health"):
        try:
            health_response = requests.get(f"{API_BASE_URL}/", timeout=15)
            st.success(f"API reachable: HTTP {health_response.status_code}")
            try:
                st.json(health_response.json())
            except ValueError:
                st.code(health_response.text)
        except Exception as exc:
            st.error(f"API check failed: {exc}")

default_candidates = load_default_candidate_ids()
default_candidate_text = ",".join(str(item_id) for item_id in default_candidates)

predict_col, recommend_col = st.columns(2)

with predict_col:
    st.subheader("1. Purchase Probability Prediction")
    with st.form("predict_form"):
        predict_user_id = st.number_input("user_id", min_value=0, step=1, value=1)
        predict_item_id = st.number_input("item_id", min_value=0, step=1, value=101)
        predict_is_addtocart = st.selectbox("is_addtocart", options=[0, 1], index=0)
        predict_submit = st.form_submit_button("Predict")

    if predict_submit:
        predict_payload = {
            "user_id": int(predict_user_id),
            "item_id": int(predict_item_id),
            "is_addtocart": int(predict_is_addtocart),
        }
        try:
            status_code, body = post_json("/predict_proba", predict_payload)
            st.write(f"HTTP status: `{status_code}`")
            if status_code == 200:
                st.metric("Predicted probability", f"{body.get('proba', 0.0):.4f}")
                st.write(f"Feature source: `{body.get('feature_source', 'unknown')}`")
            else:
                st.error("Prediction request failed.")
            with st.expander("Response JSON", expanded=True):
                st.code(json.dumps(body, indent=2), language="json")
        except Exception as exc:
            st.error(f"Prediction request failed: {exc}")

with recommend_col:
    st.subheader("2. Top-K Recommendation")
    with st.form("recommend_form"):
        recommend_user_id = st.number_input("recommend user_id", min_value=0, step=1, value=1)
        recommend_top_k = st.number_input("top_k", min_value=1, max_value=100, step=1, value=5)
        recommend_is_addtocart = st.selectbox("recommend is_addtocart", options=[0, 1], index=0)
        candidate_ids_text = st.text_area(
            "Candidate item IDs",
            value=default_candidate_text,
            help="Comma-separated item IDs used for candidate scoring. Defaults to popular items from the processed dataset.",
        )
        recommend_submit = st.form_submit_button("Recommend")

    if recommend_submit:
        try:
            candidate_item_ids = parse_candidate_ids(candidate_ids_text)
            recommend_payload = {
                "user_id": int(recommend_user_id),
                "top_k": int(recommend_top_k),
                "is_addtocart": int(recommend_is_addtocart),
                "candidate_item_ids": candidate_item_ids,
            }
            status_code, body = post_json("/recommend", recommend_payload)
            st.write(f"HTTP status: `{status_code}`")
            if status_code == 200:
                items = body.get("items", [])
                if items:
                    results_df = pd.DataFrame(items)
                    st.dataframe(results_df, use_container_width=True)
                else:
                    st.info("No recommendation items were returned.")
            else:
                st.error("Recommendation request failed.")
            with st.expander("Response JSON", expanded=True):
                st.code(json.dumps(body, indent=2), language="json")
        except Exception as exc:
            st.error(f"Recommendation request failed: {exc}")

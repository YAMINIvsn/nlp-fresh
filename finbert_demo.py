
from pathlib import Path

import pandas as pd
import streamlit as st
from transformers import pipeline

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "finbert_v2"

st.set_page_config(
    page_title="FinBERT Compliance Analyzer",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ FinBERT Compliance Analyzer")
st.caption("Interactive inference using your local fine-tuned checkpoint")

if not MODEL_PATH.exists():
    st.error(f"Model directory not found: {MODEL_PATH}")
    st.stop()


@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model=str(MODEL_PATH),
        tokenizer=str(MODEL_PATH),
        device=-1,
    )


try:
    classifier = load_model()
except Exception as exc:
    st.error("The local model could not be loaded.")
    st.exception(exc)
    st.stop()

labels = classifier.model.config.id2label
st.caption(f"Model labels: {labels}")

tab1, tab2 = st.tabs(["Single Text", "CSV Batch Analysis"])

with tab1:
    text = st.text_area(
        "Enter a compliance statement",
        value=(
            "The customer failed to complete the "
            "mandatory KYC update within the prescribed deadline."
        ),
        height=150,
    )

    if st.button("Analyze Text", type="primary"):
        if not text.strip():
            st.warning("Enter some text first.")
        else:
            with st.spinner("Running FinBERT..."):
                prediction = classifier(
                    text,
                    truncation=True,
                    max_length=512,
                )[0]

            left, right = st.columns(2)
            left.metric("Predicted label", prediction["label"])
            right.metric(
                "Model confidence",
                f'{prediction["score"]:.2%}',
            )

            st.progress(float(prediction["score"]))
            st.caption(
                "Confidence is the model's output score, "
                "not a guarantee of correctness."
            )

with tab2:
    uploaded = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
    )

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            st.stop()

        if df.empty or len(df.columns) == 0:
            st.warning("The CSV has no data.")
        else:
            st.write("Preview of uploaded data")
            st.dataframe(df.head(10), use_container_width=True)

            text_column = st.selectbox(
                "Select the column containing text",
                options=list(df.columns),
            )

            if st.button("Analyze CSV", type="primary"):
                texts = df[text_column].fillna("").astype(str).tolist()
                valid_indices = [
                    i for i, value in enumerate(texts)
                    if value.strip()
                ]

                if not valid_indices:
                    st.warning("No non-empty text was found.")
                else:
                    with st.spinner(
                        f"Analyzing {len(valid_indices)} records..."
                    ):
                        predictions = classifier(
                            [texts[i] for i in valid_indices],
                            batch_size=8,
                            truncation=True,
                            max_length=512,
                        )

                    result = df.copy()
                    result["predicted_label"] = ""
                    result["confidence"] = None

                    for i, pred in zip(valid_indices, predictions):
                        result.at[i, "predicted_label"] = pred["label"]
                        result.at[i, "confidence"] = pred["score"]

                    st.subheader("Prediction Results")
                    st.dataframe(result, use_container_width=True)

                    st.subheader("Predicted Label Distribution")
                    st.bar_chart(
                        result.loc[
                            result["predicted_label"] != "",
                            "predicted_label",
                        ].value_counts()
                    )

                    st.download_button(
                        "Download Predictions CSV",
                        data=result.to_csv(index=False).encode("utf-8"),
                        file_name="finbert_predictions.csv",
                        mime="text/csv",
                    )

st.divider()
st.warning(
    "Research demonstration only. Verify model labels and "
    "performance before using predictions in compliance decisions."
)


import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from fpdf import FPDF

st.set_page_config(page_title="Churn Prediction", page_icon="🧪", layout="wide")

# Load model + column metadata
model = joblib.load("models/churn_pipeline2.pkl")
numeric_features, categorical_features = joblib.load("models/base_columns.pkl")

raw_data = pd.read_excel("data/E Commerce Dataset.xlsx", sheet_name="E Comm")

st.title("🧪 Customer Churn Prediction")

# --------------------------------------------------------------------
# SIDEBAR – SINGLE CUSTOMER INPUT
# --------------------------------------------------------------------
st.sidebar.header("Single Customer Input")

def user_input():
    data = {}

    st.sidebar.subheader("Numeric Fields")
    for col in numeric_features:
        min_val = float(raw_data[col].min())
        max_val = float(raw_data[col].max())
        default = float(raw_data[col].median())

        data[col] = st.sidebar.slider(
            col,
            min_value=min_val,
            max_value=max_val,
            value=default,
            step=0.1,         # float step size
        )

    st.sidebar.subheader("Categorical Fields")
    for col in categorical_features:
        options = raw_data[col].unique().tolist()
        data[col] = st.sidebar.selectbox(col, options)

    return pd.DataFrame([data])


single_df = user_input()

st.write("### 🧾 Single Customer Input Summary")
st.dataframe(single_df)

# --------------------------------------------------------------------
# SINGLE PREDICTION BUTTON
# --------------------------------------------------------------------
if st.button("🔍 Predict Single Customer", use_container_width=True, key="single_predict_btn"):
    pred = model.predict(single_df)[0]
    proba = model.predict_proba(single_df)[0][1] * 100

    st.markdown("---")
    st.subheader("📌 Risk Classification")

    if proba >= 70:
        risk = "🔴 HIGH RISK"
        st.error(f"**{risk} – {proba:.2f}% chance of churn**")
    elif proba >= 40:
        risk = "🟠 MEDIUM RISK"
        st.warning(f"**{risk} – {proba:.2f}% chance of churn**")
    else:
        risk = "🟢 LOW RISK"
        st.success(f"**{risk} – {proba:.2f}% chance of churn**")


# --------------------------------------------------------------------
# MAIN AREA – BATCH CSV UPLOAD
# --------------------------------------------------------------------
st.markdown("## 📂 Batch Prediction via CSV Upload")

uploaded_file = st.file_uploader(
    "Upload a CSV file for batch churn prediction",
    type=["csv"]
)

batch_df = None

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)

    st.write("### 📄 Uploaded Data Preview")
    st.dataframe(batch_df)

    # Check if required columns exist
    required_cols = numeric_features + categorical_features
    missing_cols = [col for col in required_cols if col not in batch_df.columns]

    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        batch_df = None

# --------------------------------------------------------------------
# BATCH PREDICTION BUTTON
# --------------------------------------------------------------------
if batch_df is not None:
    if st.button("📊 Predict Batch (CSV File)", use_container_width=True, key="batch_predict_btn"):

        preds = model.predict(batch_df)
        probas = model.predict_proba(batch_df)[:, 1] * 100

        results_df = batch_df.copy()
        results_df["Churn Prediction"] = preds
        results_df["Churn Probability (%)"] = probas

        # Risk labels
        def risk_label(p):
            if p >= 70:
                return "HIGH RISK"
            elif p >= 40:
                return "MEDIUM RISK"
            return "LOW RISK"

        results_df["Risk Level"] = results_df["Churn Probability (%)"].apply(risk_label)

        st.markdown("---")
        st.subheader("✅ Batch Prediction Results")
        st.dataframe(results_df)

        # Download CSV
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Batch Results as CSV",
            csv,
            file_name="batch_churn_predictions.csv",
            mime="text/csv"
        )


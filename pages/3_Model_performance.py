import streamlit as st
import joblib
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report
)

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Model Performance Dashboard")

# ----------------------------------------------------
# LOAD MODEL + METRICS
# ----------------------------------------------------
model = joblib.load("models/churn_pipeline2.pkl")

# Load stored metrics (accuracy, ROC AUC, report)
with open("models/metrics.json", "r") as f:
    metrics = json.load(f)

# Load dataset (for confusion matrix + ROC curve)
df = pd.read_excel('data/E Commerce Dataset.xlsx', sheet_name='E Comm')
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Predictions
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

# ----------------------------------------------------
# ACCURACY
# ----------------------------------------------------
st.subheader("✅ Overall Accuracy")
st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")

# ----------------------------------------------------
# CONFUSION MATRIX
# ----------------------------------------------------
st.subheader("📊 Confusion Matrix")
fig, ax = plt.subplots()
ConfusionMatrixDisplay(confusion_matrix(y, y_pred)).plot(ax=ax)
st.pyplot(fig)

# ----------------------------------------------------
# ROC CURVE
# ----------------------------------------------------
st.subheader("📈 ROC Curve")

# Use stored AUC value instead of recomputation
roc_auc = metrics["roc_auc"]

fpr, tpr, _ = roc_curve(y, y_proba)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_title("ROC Curve")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
st.pyplot(fig)

# ----------------------------------------------------
# CLASSIFICATION REPORT (stored)
# ----------------------------------------------------
# Extract the classification report from JSON
report_dict = metrics["classification_report"]

# Convert to DataFrame
report_df = pd.DataFrame(report_dict).transpose()

st.subheader("📄 Classification Report (Table View)")
st.dataframe(report_df)

# ----------------------------------------------------
# FEATURE IMPORTANCE
# ----------------------------------------------------
st.markdown("---")
st.subheader("🔍 Feature Importance (Random Forest)")

model2 = joblib.load("models/model.pkl")

clf = model2.named_steps["classifier"]
preprocessor = model.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()
importances = clf.feature_importances_

fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

st.bar_chart(fi_df.set_index("Feature"))

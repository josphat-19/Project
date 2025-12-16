import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Data Exploration", page_icon="📁", layout="wide")

st.title("📁 Customer Data Exploration")

df = pd.read_excel('data/E Commerce Dataset.xlsx', sheet_name='E Comm')

st.subheader("📌 Dataset Preview")
st.dataframe(df)

st.subheader("📈 Summary Statistics")
st.write(df.describe())

st.subheader("🔍 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.subheader("📊 Churn Distribution")
fig, ax = plt.subplots()
sns.countplot(x="Churn", data=df, ax=ax)
st.pyplot(fig)

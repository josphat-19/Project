# E-Commerce Customer Churn Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Model-Stacking%20Ensemble-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()

## 📌 Project Overview
Customer churn is a critical challenge in the e-commerce sector, directly impacting long-term profitability and customer acquisition costs. This project implements a high-performance predictive system designed to identify at-risk customers by analyzing behavioral patterns such as purchase frequency, complaint history, and platform engagement.

Developed as a final year capstone project for **BSc. Information Technology** at **Dedan Kimathi University of Technology (DeKUT)**, this system leverages advanced ensemble learning to provide business intelligence for data-driven retention strategies.

## 🚀 Key Performance Metrics
The system utilizes a **Stacking Ensemble** architecture, achieving state-of-the-art results:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **98%** |
| **Recall (Sensitivity)** | **94%** |
| **ROC-AUC Score** | **0.96** |
| **Precision** | **92%** |

## 🛠️ Technical Implementation
### 1. Data Balancing (SMOTE)
To address class imbalance (where loyal customers outnumber churners), **Synthetic Minority Over-sampling Technique (SMOTE)** was utilized. This significantly improved the model's ability to detect churners without sacrificing overall accuracy.

### 2. Stacking Ensemble Architecture
The model combines the predictive power of multiple algorithms:
* **Base Learners:** Random Forest Classifier & XGBoost.
* **Meta-Learner:** Logistic Regression (aggregates base predictions for final classification).

### 3. Interactive Dashboard
The system is deployed as a **Streamlit** web application, enabling:
* **Batch Inference:** Upload `.csv` datasets for immediate churn analysis.
* **Visualization:** Real-time generation of feature importance and churn distribution charts.
* **Risk Labeling:** Automated categorization of customers into "High Risk" and "Low Risk."

## 🏗️ System Architecture & Pipeline
1.  **EDA:** Identification of churn drivers (Tenure, Complaints, Cashback).
2.  **Preprocessing:** Handling missing values, categorical encoding, and feature scaling.
3.  **Modeling:** Training via Stacking Ensemble and validation using K-Fold cross-validation.
4.  **Deployment:** Serialization of models (`.pkl`) for real-time use in the Streamlit UI.

## 📁 Project Structure
```text
├── data/               # Datasets (Raw & Preprocessed)
├── notebooks/          # EDA & Model Training Notebooks
├── app.py              # Streamlit Application Source Code
├── requirements.txt    # List of Dependencies
├── models/             # Serialized Model Files (.pkl)
└── README.md           # Documentation

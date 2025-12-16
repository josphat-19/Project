import streamlit as st

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📉",
    layout="wide"
)

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>

.big-title {
    font-size: 42px !important;
    font-weight: 800;
    text-align: center;
    padding-top: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #888;
    margin-top: -10px;
    margin-bottom: 30px;
}

.feature-card {
    background-color: rgba(255,255,255,0.1);
    padding: 20px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.3);
    backdrop-filter: blur(4px);
    transition: 0.3s ease;
    height: 180px;
}

.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0px 4px 16px rgba(0,0,0,0.15);
}

.feature-card h3 {
    color: white;
    font-size: 20px;
}

.feature-card p {
    color: #ddd;
    font-size: 15px;
}

.footer {
    text-align: center;
    margin-top: 50px;
    color: #aaa;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# HERO SECTION
# ----------------------------
st.markdown("<h1 class='big-title'>📉 Customer Churn Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict, analyze, and prevent customer churn using advanced machine learning insights.</p>", unsafe_allow_html=True)

st.markdown("---")

# ----------------------------
# PROJECT SUMMARY
# ----------------------------
st.subheader("🎯 What This System Does")

st.write("""
This machine-learning platform helps businesses reduce customer churn by:

- Predicting churn probability  
- Highlighting high-risk customers  
- Providing personalized business recommendations  
- Visualizing key performance metrics  
- Offering detailed dataset exploration  

Use the sidebar to start predicting, view dashboards, or access insights.
""")

st.info("💡 Tip: Start with **Churn Prediction** to get a churn probability and risk level.")

st.markdown("---")

# ----------------------------
# FEATURE CARDS
# ----------------------------
st.subheader("🚀 Key Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='feature-card'>
        <h3>🔍 Churn Prediction</h3>
        <p>Generate real-time customer churn probabilities using machine learning models.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='feature-card'>
        <h3>📊 Performance Dashboard</h3>
        <p>View metrics like ROC curves, confusion matrices, accuracy, and feature importance.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='feature-card'>
        <h3>📂 Dataset Explorer</h3>
        <p>Interactively browse, filter, and analyze the customer dataset used for training.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("<div class='footer'>© 2025 • Customer Churn Prediction System • Built with Streamlit</div>", unsafe_allow_html=True)

import streamlit as st

st.set_page_config(
    page_title="Business Insights",
    page_icon="💡",
    layout="wide"
)

st.title("💡 Business Insights & Recommendations")

st.markdown("""
This page provides **general strategic insights and recommendations** for customers with different churn risk levels.
It is designed to help decision-makers understand customer behaviors and plan interventions.
""")

st.markdown("---")

# -----------------------------
# HIGH RISK
# -----------------------------
st.subheader("🔴 HIGH RISK CUSTOMERS")
st.write("""
Customers in this category are at **very high risk of churning**. Immediate interventions are recommended.
""")
st.markdown("""
**Recommendations:**
- Offer retention incentives (discounts, loyalty rewards)
- Assign customer success managers to reach out personally
- Review past complaints, support tickets, or service issues
- Provide personalized retention plans or upgrade offers
""")
st.write("""
**Strategic Interpretation:**  
High-risk customers often experience dissatisfaction in pricing, service, or communication. Quick and personalized retention strategies are critical.
""")
st.markdown("---")

# -----------------------------
# MEDIUM RISK
# -----------------------------
st.subheader("🟠 MEDIUM RISK CUSTOMERS")
st.write("""
Customers in this category show **warning signals** but can still be retained with proactive engagement.
""")
st.markdown("""
**Recommendations:**
- Send targeted engagement emails or in-app notifications
- Promote product features they are underusing
- Offer optional check-ins or customer education programs
- Monitor activity levels and intervene if declining
""")
st.write("""
**Strategic Interpretation:**  
Medium-risk customers may have minor dissatisfaction or engagement gaps. Focus on **personalized communication** and monitoring to reduce churn probability.
""")
st.markdown("---")

# -----------------------------
# LOW RISK
# -----------------------------
st.subheader("🟢 LOW RISK CUSTOMERS")
st.write("""
Customers in this category are **stable and satisfied**, but should be nurtured to maximize lifetime value.
""")
st.markdown("""
**Recommendations:**
- Maintain engagement with regular updates or newsletters
- Provide loyalty points, exclusive content, or referral incentives
- Encourage upsells or cross-sells based on their satisfaction
- Solicit testimonials or reviews to strengthen brand advocacy
""")
st.write("""
**Strategic Interpretation:**  
Low-risk customers are loyal but should still be nurtured to avoid slipping into higher risk categories. Reward loyalty and maintain engagement.
""")
st.markdown("---")

st.caption("Business Insights Module © 2025 – Provides actionable strategies for managing customer churn across risk categories.")

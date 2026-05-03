import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "clv_model.pkl")
model = joblib.load(model_path)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="CLV Prediction", layout="wide")

# =========================
# HEADER
# =========================
st.title("💰 Customer Lifetime Value Prediction")
st.markdown("### Predict customer value using Machine Learning")

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns(2)

# =========================
# INPUT SECTION
# =========================
with col1:
    st.subheader("📥 Customer Input")

    recency = st.number_input("Recency (days)", min_value=0, value=10)
    frequency = st.number_input("Frequency", min_value=0, value=5)
    lifetime = st.number_input("Customer Lifetime (days)", min_value=0, value=100)

# =========================
# FEATURE ENGINEERING
# =========================
purchase_intensity = frequency / (lifetime + 1)

recency_log = np.log1p(recency)
frequency_log = np.log1p(frequency)
lifetime_log = np.log1p(lifetime)
purchase_intensity_log = np.log1p(purchase_intensity)

# =========================
# PREDICTION + OUTPUT
# =========================
with col2:
    st.subheader("📊 Prediction Output")

    if st.button("🔍 Predict CLV"):

        input_data = np.array([[
            recency_log,
            frequency_log,
            lifetime_log,
            purchase_intensity_log
        ]])

        prediction_log = model.predict(input_data)[0]
        prediction_actual = np.expm1(prediction_log)

        # Classification
        if prediction_actual < 1000:
            category = "Low Value"
            color = "red"
        elif prediction_actual < 5000:
            category = "Medium Value"
            color = "orange"
        else:
            category = "High Value"
            color = "green"

        st.metric("Predicted CLV (log)", f"{prediction_log:.2f}")
        st.metric("Estimated Value ($)", f"{prediction_actual:.2f}")
        st.markdown(f"### Customer Segment: :{color}[{category}]")

        # =========================
        # CHART (FEATURE CONTRIBUTION)
        # =========================
        st.subheader("📊 Feature Values")

        features = ["Recency", "Frequency", "Lifetime", "Purchase Intensity"]
        values = [recency, frequency, lifetime, purchase_intensity]

        fig, ax = plt.subplots()
        ax.barh(features, values)
        ax.set_title("Customer Feature Overview")

        st.pyplot(fig)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.subheader("ℹ️ About Model")

st.write("""
- Model: Linear Regression  
- Method: RFM-based Customer Analysis  
- Features:
  - Recency
  - Frequency
  - Lifetime
  - Purchase Intensity

This model predicts customer value based on purchasing behavior.
""")
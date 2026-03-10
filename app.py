import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance

# -------------------------
# Load trained model + scaler + feature columns
# -------------------------
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")  # List of columns used in training

st.set_page_config(page_title="Diabetes Risk Predictor 🩺", layout="wide")
st.title("🩺 Diabetes Risk Prediction App")

# =========================
# Sidebar for User Inputs
# =========================
st.sidebar.header("Enter Your Health Data")

age = st.sidebar.number_input("Age", 0, 120, 40)
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 27.0)
glucose = st.sidebar.number_input("Blood Glucose Level", 50, 300, 120)
HbA1c = st.sidebar.number_input("HbA1c Level", 3.0, 15.0, 5.5)
gender = st.sidebar.selectbox("Gender", ["Female", "Male", "Other"])
smoking = st.sidebar.selectbox("Smoking History", ["never","current","former","ever","not current","No Info"])

# =========================
# Prepare DataFrame with all columns
# =========================
user_input = {
    "age": age,
    "bmi": bmi,
    "blood_glucose_level": glucose,
    "HbA1c_level": HbA1c,
    "gender_Male": 1 if gender=="Male" else 0,
    "gender_Other": 1 if gender=="Other" else 0,
    "smoking_history_current": 1 if smoking=="current" else 0,
    "smoking_history_ever": 1 if smoking=="ever" else 0,
    "smoking_history_former": 1 if smoking=="former" else 0,
    "smoking_history_never": 1 if smoking=="never" else 0,
    "smoking_history_not current": 1 if smoking=="not current" else 0,
    "smoking_history_No Info": 1 if smoking=="No Info" else 0
}

# Create a DataFrame with **all training columns**
user_df_full = pd.DataFrame(columns=feature_columns)
for col in feature_columns:
    if col in user_input:
        user_df_full.loc[0, col] = user_input[col]
    else:
        user_df_full.loc[0, col] = 0  # Missing columns filled with 0

# Scale the user input
user_scaled = scaler.transform(user_df_full)

# =========================
# Prediction
# =========================
if st.sidebar.button("Predict Risk"):
    prob = model.predict_proba(user_scaled)[0][1]

    # Risk Category
    if prob < 0.3:
        risk = "Low Risk"
        color = "green"
    elif prob < 0.6:
        risk = "Moderate Risk"
        color = "orange"
    else:
        risk = "High Risk"
        color = "red"

    # Display Metrics
    st.markdown(f"### Predicted Probability of Diabetes: {prob:.2f}")
    st.markdown(f"### Risk Level: <span style='color:{color};'>{risk}</span>", unsafe_allow_html=True)

    # Probability Bar
    st.progress(min(max(prob,0),1))

    # Feature Importance
    st.subheader("XGBoost Feature Importance")
    fig, ax = plt.subplots(figsize=(8,5))
    plot_importance(model, importance_type='weight', ax=ax)
    st.pyplot(fig)

# =========================
# Optional Info Section
# =========================
st.markdown("""
---
#### About This App
This app predicts your risk of diabetes using your health data.
- High blood glucose and HbA1c indicate higher risk.
- BMI, age, smoking history, and gender are also considered.
""")

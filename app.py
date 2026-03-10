# ===============================
# app.py - Diabetes Risk Predictor
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# 1️⃣ Load Model & Scaler
# -------------------------------
@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# -------------------------------
# 2️⃣ Streamlit UI
# -------------------------------
st.title("Diabetes Risk Predictor")

# User input (example)
age = st.number_input("Age", min_value=1, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
# Add other features like glucose_level, bp, etc.

# Predict button
if st.button("Predict Risk"):
    X_user = np.array([[age, bmi]])  # Adjust according to your model features
    X_user_scaled = scaler.transform(X_user)
    prob = model.predict_proba(X_user_scaled)[0][1]
    
    if prob < 0.3:
        risk = "Low Risk"
    elif prob < 0.6:
        risk = "Moderate Risk"
    else:
        risk = "High Risk"
    
    st.write(f"Predicted Diabetes Probability: {prob:.2f}")
    st.write(f"Risk Level: {risk}")

# ===============================
# app.py - Diabetes Risk Predictor
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier

# -------------------------------
# 1️⃣ Load Model & Scaler
# -------------------------------
@st.cache_resource
def load_model():
    # Load trained model
    model = XGBClassifier()
    model.load_model("xgb_model.json")
    # Load scaler
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

# -------------------------------
# 2️⃣ App Title
# -------------------------------
st.title("Diabetes Risk Predictor")
st.write("Enter your health details to predict diabetes risk.")

# -------------------------------
# 3️⃣ User Input
# -------------------------------
age = st.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
smoking_history = st.selectbox(
    "Smoking History", ["never", "current", "former", "ever", "not current", "No Info"]
)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=20.0, value=5.5)
blood_glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=500, value=100)

# -------------------------------
# 4️⃣ Encode Inputs
# -------------------------------
def encode_input():
    # Gender encoding
    gender_male = 1 if gender == "Male" else 0
    gender_other = 1 if gender == "Other" else 0
    
    # Smoking history encoding
    smoke_current = 1 if smoking_history=="current" else 0
    smoke_ever = 1 if smoking_history=="ever" else 0
    smoke_former = 1 if smoking_history=="former" else 0
    smoke_never = 1 if smoking_history=="never" else 0
    smoke_not_current = 1 if smoking_history=="not current" else 0

    # Create dataframe for model
    input_df = pd.DataFrame([[
        age, hypertension, heart_disease, bmi, hba1c, blood_glucose,
        gender_male, gender_other,
        smoke_current, smoke_ever, smoke_former, smoke_never, smoke_not_current
    ]], columns=[
        'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level',
        'gender_Male', 'gender_Other',
        'smoking_history_current', 'smoking_history_ever', 'smoking_history_former', 'smoking_history_never',
        'smoking_history_not current'
    ])
    return input_df

input_data = encode_input()
input_scaled = scaler.transform(input_data)

# -------------------------------
# 5️⃣ Predict
# -------------------------------
prob = model.predict_proba(input_scaled)[:,1][0]

def risk_category(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.6:
        return "Moderate Risk"
    else:
        return "High Risk"

risk = risk_category(prob)

# -------------------------------
# 6️⃣ Show Results
# -------------------------------
st.subheader("Results")
st.write(f"Diabetes Risk Probability: **{prob:.2f}**")
st.write(f"Risk Level: **{risk}**")

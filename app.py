import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

st.title("Diabetes Risk Prediction App 🩺")

# ===============================
# 1️⃣ Upload Dataset (Optional)
# ===============================
uploaded_file = st.file_uploader("Upload diabetes dataset (CSV inside ZIP)", type=["zip"])

if uploaded_file is not None:
    extract_dir = "archive_extracted/"
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    df = pd.read_csv(os.path.join(extract_dir, "diabetes_prediction_dataset.csv"))
    st.success("Dataset Loaded ✅")
    st.dataframe(df.head())

    # ===============================
    # 2️⃣ Data Cleaning & Encoding
    # ===============================
    df = df.drop_duplicates()
    if 'gender' in df.columns:
        df = pd.get_dummies(df, columns=["gender", "smoking_history"], drop_first=True)

    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Balance train data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train_res, y_train_res)

    st.success("Model Trained ✅")

    # ===============================
    # 3️⃣ Predict Risk for New User
    # ===============================
    st.header("Predict Your Risk Level")
    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(f"Enter {col}", value=float(X[col].mean()))

    if st.button("Predict"):
        user_df = pd.DataFrame([user_input])
        user_scaled = scaler.transform(user_df)
        prob = model.predict_proba(user_scaled)[0][1]

        if prob < 0.3:
            risk = "Low Risk"
        elif prob < 0.6:
            risk = "Moderate Risk"
        else:
            risk = "High Risk"

        st.metric("Predicted Probability of Diabetes", f"{prob:.2f}")
        st.metric("Risk Level", risk)

        st.success(f"Your risk level is: {risk}")

    # Optional: show feature importance
    st.header("Feature Importance")
    fig, ax = plt.subplots(figsize=(8,6))
    plot_importance(model, importance_type='weight', ax=ax)
    st.pyplot(fig)

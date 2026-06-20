"""
streamlit_app.py
----------------
Streamlit Community Cloud entry point for the churn prediction app.

Unlike app.py (the Flask version), this app is self-contained: if a
trained model isn't already present in model/, it trains one on cold
start from data/cleaned_churn_data.csv (which IS committed to the repo,
unlike the raw Kaggle CSV) and caches it for the life of the server
process. No manual training step is required before deploying.
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_churn_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

CAT_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


@st.cache_resource(show_spinner="Loading model (training on first run)...")
def load_or_train_model():
    """Load a previously-trained model from model/, or train one fresh
    from the bundled dataset and cache it for this server's lifetime."""
    model_path = os.path.join(MODEL_DIR, "churn_model.pkl")
    encoders_path = os.path.join(MODEL_DIR, "encoders.pkl")
    features_path = os.path.join(MODEL_DIR, "feature_names.pkl")

    if all(os.path.exists(p) for p in (model_path, encoders_path, features_path)):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(encoders_path, "rb") as f:
            encoders = pickle.load(f)
        with open(features_path, "rb") as f:
            feature_names = pickle.load(f)
        return model, encoders, feature_names

    # No cached model on disk — train one now from the bundled CSV.
    df = pd.read_csv(DATA_PATH)

    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    le_churn = LabelEncoder()
    df["Churn"] = le_churn.fit_transform(df["Churn"].astype(str))
    encoders["Churn"] = le_churn

    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "churn_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "wb") as f:
        pickle.dump(list(X.columns), f)

    return model, encoders, list(X.columns)


def encode_input(value, col, encoders):
    le = encoders.get(col)
    if le and value in le.classes_:
        return le.transform([value])[0]
    elif le:
        return le.transform([le.classes_[0]])[0]
    return 0


def predict_churn(inputs, model, encoders, feature_names):
    features = {col: encode_input(inputs[col], col, encoders) for col in CAT_COLS}
    features["SeniorCitizen"] = inputs["SeniorCitizen"]
    features["tenure"] = inputs["tenure"]
    features["MonthlyCharges"] = inputs["MonthlyCharges"]
    features["TotalCharges"] = inputs["TotalCharges"]

    input_df = pd.DataFrame([[features[c] for c in feature_names]], columns=feature_names)
    proba = model.predict_proba(input_df)[0]
    churn_proba = float(proba[1])

    if churn_proba >= 0.7:
        risk_level, risk_color = "High Risk", "#e74c3c"
    elif churn_proba >= 0.4:
        risk_level, risk_color = "Medium Risk", "#f39c12"
    else:
        risk_level, risk_color = "Low Risk", "#27ae60"

    return {
        "prediction": "Yes" if churn_proba >= 0.5 else "No",
        "churn_probability": round(churn_proba * 100, 1),
        "retain_probability": round((1 - churn_proba) * 100, 1),
        "risk_level": risk_level,
        "risk_color": risk_color,
    }


# ------------------------------------------------------------------- UI --
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")

st.title("📊 Customer Churn Predictor")
st.caption(
    "Random Forest model trained on 7,032 telecom customers · 78.32% test accuracy"
)

model, encoders, feature_names = load_or_train_model()

with st.form("churn_form"):
    st.subheader("Customer profile")

    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 1, 72, 12)
    with c2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])
        monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
        total = st.number_input("Total Charges ($)", 18.0, 8700.0, 500.0, step=10.0)
    with c3:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    c4, c5 = st.columns(2)
    with c4:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    with c5:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    submitted = st.form_submit_button("Predict churn risk", use_container_width=True)

if submitted:
    inputs = {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": multiple_lines,
        "InternetService": internet, "OnlineSecurity": security,
        "OnlineBackup": backup, "DeviceProtection": device_protection,
        "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }
    result = predict_churn(inputs, model, encoders, feature_names)

    st.divider()
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Churn Probability", f"{result['churn_probability']}%")
    col_b.metric("Retain Probability", f"{result['retain_probability']}%")
    col_c.metric("Prediction", result["prediction"])

    st.markdown(
        f"<div style='padding:14px;border-radius:8px;background-color:{result['risk_color']}20;"
        f"border:1px solid {result['risk_color']};text-align:center;font-size:18px;"
        f"font-weight:600;color:{result['risk_color']}'>{result['risk_level']}</div>",
        unsafe_allow_html=True,
    )

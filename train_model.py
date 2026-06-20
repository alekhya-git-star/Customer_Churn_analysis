"""
train_model.py
--------------
Run this script ONCE to train the model and save it to the model/ folder.
You need the cleaned dataset (cleaned_churn_data.csv) in the data/ folder.

Usage:
    python train_model.py

Dataset source:
    https://www.kaggle.com/datasets/blastchar/telco-customer-churn
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH  = "data/cleaned_churn_data.csv"
MODEL_DIR  = "model"
# ────────────────────────────────────────────────────────────────────────────

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape}")

    # Encode categorical columns
    cat_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  Encoded: {col}")

    # Target
    le_churn = LabelEncoder()
    df['Churn'] = le_churn.fit_transform(df['Churn'].astype(str))
    encoders['Churn'] = le_churn

    # Drop customerID, fix TotalCharges
    df = df.drop('customerID', axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    print(f"\nTraining Random Forest on {len(X)} records...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save artifacts
    with open(f"{MODEL_DIR}/churn_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{MODEL_DIR}/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open(f"{MODEL_DIR}/feature_names.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)

    print(f"\nSaved model artifacts to /{MODEL_DIR}/")
    print("Done! Now run: python app.py")


if __name__ == "__main__":
    train()

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load model and encoders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'model', 'churn_model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, 'model', 'encoders.pkl'), 'rb') as f:
    encoders = pickle.load(f)
with open(os.path.join(BASE_DIR, 'model', 'feature_names.pkl'), 'rb') as f:
    feature_names = pickle.load(f)


def encode_input(value, col):
    le = encoders.get(col)
    if le and value in le.classes_:
        return le.transform([value])[0]
    elif le:
        return le.transform([le.classes_[0]])[0]
    return 0


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        # Build feature vector in the same order as training
        features = {
            'gender': encode_input(data.get('gender', 'Male'), 'gender'),
            'SeniorCitizen': int(data.get('SeniorCitizen', 0)),
            'Partner': encode_input(data.get('Partner', 'No'), 'Partner'),
            'Dependents': encode_input(data.get('Dependents', 'No'), 'Dependents'),
            'tenure': int(data.get('tenure', 1)),
            'PhoneService': encode_input(data.get('PhoneService', 'Yes'), 'PhoneService'),
            'MultipleLines': encode_input(data.get('MultipleLines', 'No'), 'MultipleLines'),
            'InternetService': encode_input(data.get('InternetService', 'DSL'), 'InternetService'),
            'OnlineSecurity': encode_input(data.get('OnlineSecurity', 'No'), 'OnlineSecurity'),
            'OnlineBackup': encode_input(data.get('OnlineBackup', 'No'), 'OnlineBackup'),
            'DeviceProtection': encode_input(data.get('DeviceProtection', 'No'), 'DeviceProtection'),
            'TechSupport': encode_input(data.get('TechSupport', 'No'), 'TechSupport'),
            'StreamingTV': encode_input(data.get('StreamingTV', 'No'), 'StreamingTV'),
            'StreamingMovies': encode_input(data.get('StreamingMovies', 'No'), 'StreamingMovies'),
            'Contract': encode_input(data.get('Contract', 'Month-to-month'), 'Contract'),
            'PaperlessBilling': encode_input(data.get('PaperlessBilling', 'Yes'), 'PaperlessBilling'),
            'PaymentMethod': encode_input(data.get('PaymentMethod', 'Electronic check'), 'PaymentMethod'),
            'MonthlyCharges': float(data.get('MonthlyCharges', 50)),
            'TotalCharges': float(data.get('TotalCharges', 500)),
        }

        # Look up values via feature_names (the exact column order used at
        # training time) rather than dict order, so this stays correct even
        # if the dict above is ever reordered. Wrapped in a DataFrame (not a
        # bare ndarray) so column names match what the model was fit on.
        input_df = pd.DataFrame([[features[col] for col in feature_names]], columns=feature_names)
        proba = model.predict_proba(input_df)[0]
        churn_proba = float(proba[1])
        prediction = 'Yes' if churn_proba >= 0.5 else 'No'

        risk_level = 'High Risk' if churn_proba >= 0.7 else ('Medium Risk' if churn_proba >= 0.4 else 'Low Risk')
        risk_color = '#e74c3c' if churn_proba >= 0.7 else ('#f39c12' if churn_proba >= 0.4 else '#27ae60')

        return jsonify({
            'prediction': prediction,
            'churn_probability': round(churn_proba * 100, 1),
            'retain_probability': round((1 - churn_proba) * 100, 1),
            'risk_level': risk_level,
            'risk_color': risk_color
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)

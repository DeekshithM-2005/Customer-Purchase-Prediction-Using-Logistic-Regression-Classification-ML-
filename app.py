"""
Customer Purchase Prediction Using Logistic Regression Classification
Flask application with prediction interface.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import os

app = Flask(__name__)
CORS(app)

# --- Load Model Artifacts ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_DIR = os.path.join(BASE_DIR, 'data')

model = joblib.load(os.path.join(MODEL_DIR, 'logistic_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

with open(os.path.join(MODEL_DIR, 'metrics.json'), 'r') as f:
    metrics = json.load(f)

# Load dataset for basic stats
df = pd.read_csv(os.path.join(DATA_DIR, 'online_shoppers_intention.csv'))

FEATURE_COLUMNS = metrics['feature_columns']

MONTH_OPTIONS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
VISITOR_OPTIONS = ['Returning_Visitor', 'New_Visitor', 'Other']

MONTH_MAP = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}
VISITOR_MAP = {'Returning_Visitor': 0, 'New_Visitor': 1, 'Other': 2}


# ========================
# PAGE ROUTES
# ========================

@app.route('/')
def home():
    """Home page with project overview and model metrics."""
    # Basic dataset stats
    total = len(df)
    purchased = int(df['Revenue'].sum())
    not_purchased = total - purchased
    purchase_rate = round(purchased / total * 100, 1)

    return render_template('index.html',
        metrics=metrics,
        total_samples=total,
        purchased=purchased,
        not_purchased=not_purchased,
        purchase_rate=purchase_rate
    )


@app.route('/predict')
def predict_page():
    """Prediction form page."""
    return render_template('predict.html',
        months=MONTH_OPTIONS,
        visitors=VISITOR_OPTIONS
    )


# ========================
# API ENDPOINTS
# ========================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Predict whether a customer will make a purchase."""
    try:
        data = request.get_json()

        # Convert Month and VisitorType from string to encoded values
        if 'Month' in data and isinstance(data['Month'], str):
            data['Month'] = MONTH_MAP.get(data['Month'], 0)
        if 'VisitorType' in data and isinstance(data['VisitorType'], str):
            data['VisitorType'] = VISITOR_MAP.get(data['VisitorType'], 2)

        # Validate required fields
        missing = [col for col in FEATURE_COLUMNS if col not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400

        # Extract features in correct order
        features = [float(data[col]) for col in FEATURE_COLUMNS]
        features_array = np.array(features).reshape(1, -1)

        # Scale and predict
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        # Build response
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Will Purchase' if prediction == 1 else 'Will Not Purchase',
            'confidence': round(float(max(probability)) * 100, 2),
            'purchase_probability': round(float(probability[1]) * 100, 2),
            'no_purchase_probability': round(float(probability[0]) * 100, 2),
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """Return model performance metrics."""
    return jsonify(metrics)


# ========================
# RUN SERVER
# ========================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Customer Purchase Prediction Using Logistic Regression")
    print("=" * 60)
    print(f"Model Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"Dataset: {metrics['total_samples']} samples")
    print(f"Server: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

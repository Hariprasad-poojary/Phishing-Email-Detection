import pickle
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import os

app = Flask(__name__, static_folder='.')
CORS(app)

# Load model and vectorizer
model_path = "Phishing-Email-Detection-System-main/models/phishing_detector.pkl"
vectorizer_path = "Phishing-Email-Detection-System-main/models/vectorizer.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    email_text = data.get("email", "")
    if not email_text:
        return jsonify({"error": "No email provided"}), 400

    # Vectorize input
    X = vectorizer.transform([email_text])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0][prediction]

    result = {
        "prediction": "phishing" if prediction == 1 else "safe",
        "probability": float(proba),
        "label": "Phishing Email" if prediction == 1 else "Safe Email"
    }
    return jsonify(result)

@app.route('/')
def serve_index():
    return send_from_directory('.', 'Phishing-Email-Detection-System-main/index.html')

@app.route('/<path:path>')
def serve_file(path):
    if os.path.exists(path):
        return send_from_directory('.', path)
    else:
        return "File not found", 404

if __name__ == "__main__":
    app.run(debug=True)

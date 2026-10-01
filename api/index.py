import os
import joblib
import pandas as pd

from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "heart_model.pkl"
)

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
features = model_data["features"]


@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/style.css")
def style():
    return send_from_directory(BASE_DIR, "style.css")


@app.route("/script.js")
def script():
    return send_from_directory(BASE_DIR, "script.js")


@app.route("/api/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        required_columns = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal"
        ]

        missing = [
            col for col in required_columns
            if col not in data
        ]

        if missing:
            return jsonify({
                "error": "Missing fields: " + ", ".join(missing)
            }), 400

        input_df = pd.DataFrame([{
            col: float(data[col])
            for col in required_columns
        }])

        input_df = pd.get_dummies(input_df)

        input_df = input_df.reindex(
            columns=features,
            fill_value=0
        )

        prediction = model.predict(input_df)[0]

        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            probability = float(probabilities[1] * 100)

        if int(prediction) == 1:
            result = "Higher predicted likelihood of heart disease"
        else:
            result = "Lower predicted likelihood of heart disease"

        return jsonify({
            "prediction": int(prediction),
            "result": result,
            "probability": round(probability, 2)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500
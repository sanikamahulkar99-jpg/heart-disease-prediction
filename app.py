import os
import joblib
import pandas as pd

from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "heart_model.pkl"
)

try:
    model_data = joblib.load(MODEL_PATH)

    model = model_data["model"]
    features = model_data["features"]

    print("Model loaded successfully.")
    print("Expected features:", features)

except Exception as e:
    print("MODEL LOADING ERROR:", e)
    model = None
    features = []


# -----------------------------
# Serve website
# -----------------------------
@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/style.css")
def style():
    return send_from_directory(BASE_DIR, "style.css")


@app.route("/script.js")
def script():
    return send_from_directory(BASE_DIR, "script.js")


# -----------------------------
# Prediction API
# -----------------------------
@app.route("/api/predict", methods=["POST"])
def predict():

    try:
        # Check model
        if model is None:
            return jsonify({
                "error": "Model could not be loaded."
            }), 500

        # Get JSON data
        data = request.get_json()

        print("\n==============================")
        print("Prediction request received")
        print("==============================")
        print(data)

        if not data:
            return jsonify({
                "error": "No data received from the form."
            }), 400

        # Required columns
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

        # Check missing fields
        missing = [
            column
            for column in required_columns
            if column not in data
        ]

        if missing:
            return jsonify({
                "error": "Missing fields: " + ", ".join(missing)
            }), 400

        # Create dataframe
        input_df = pd.DataFrame([{
            column: float(data[column])
            for column in required_columns
        }])

        print("\nInput dataframe:")
        print(input_df)

        # Apply same preprocessing
        input_df = pd.get_dummies(input_df)

        # Match training features
        input_df = input_df.reindex(
            columns=features,
            fill_value=0
        )

        print("\nFinal model input:")
        print(input_df)

        # Prediction
        prediction = model.predict(input_df)[0]

        # Probability
        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            probability = float(probabilities[1] * 100)

        # Result
        if int(prediction) == 1:
            result = "Higher predicted likelihood of heart disease"
        else:
            result = "Lower predicted likelihood of heart disease"

        print("\nPrediction:", prediction)
        print("Probability:", probability)
        print("==============================\n")

        return jsonify({
            "prediction": int(prediction),
            "result": result,
            "probability": round(probability, 2)
            if probability is not None else None
        })

    except Exception as e:

        print("\n==============================")
        print("PREDICTION ERROR")
        print("==============================")
        print(str(e))
        print("==============================\n")

        return jsonify({
            "error": str(e)
        }), 500


# -----------------------------
# Run application
# -----------------------------
if __name__ == "__main__":

    print("")
    print("======================================")
    print("   HEART DISEASE PREDICTION APP")
    print("======================================")
    print("")
    print("Server running at:")
    print("http://127.0.0.1:5000")
    print("")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
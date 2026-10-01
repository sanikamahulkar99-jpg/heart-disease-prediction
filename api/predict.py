import os
import joblib
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "heart_model.pkl"
)

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
features = model_data["features"]


@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        input_df = pd.DataFrame([data])

        input_df = pd.get_dummies(input_df)

        input_df = input_df.reindex(
            columns=features,
            fill_value=0
        )

        prediction = model.predict(input_df)[0]

        probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]

            if len(probabilities) > 1:
                probability = float(probabilities[1] * 100)


        if int(prediction) == 1:

            result = "Higher predicted likelihood of heart disease"

        else:

            result = "Lower predicted likelihood of heart disease"


        return jsonify({
            "prediction": int(prediction),
            "result": result,
            "probability": round(probability, 2)
            if probability is not None else None
        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
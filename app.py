from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return "Iris ML API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = np.array([
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"]
    ]).reshape(1, -1)

    prediction = model.predict(features)[0]

    return jsonify({
        "prediction": int(prediction)
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
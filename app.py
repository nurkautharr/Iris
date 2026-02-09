from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("iris_model.pkl")

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
    app.run(debug=True)
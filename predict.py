import pickle
from flask import Flask, request, jsonify


with open("model.bin", "rb") as f_model:
    model = pickle.load(f_model)

with open("dv.bin", "rb") as f_dv:
    dv = pickle.load(f_dv)

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()

    if customer is None:
        return jsonify({"error": "Missing JSON body"}), 400

    X = dv.transform([customer])
    churn_proba = float(model.predict_proba(X)[0, 1])
    churn = churn_proba >= 0.5

    return jsonify({
        "churn_probability": churn_proba,
        "churn": bool(churn)
    })


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Streaming Churn Prediction API. Use POST /predict."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=True)

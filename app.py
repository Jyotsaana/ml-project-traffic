from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "ML API is running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    input_df = pd.DataFrame([{
        "curvature": data["curvature"],
        "speed_limit": data["speed_limit"],
        "lighting": data["lighting"],
        "weather": data["weather"],
        "num_reported_accidents": data["num_reported_accidents"]
    }])


# Because the dictionary itself cannot transform anything.


    input_df["lighting"] = preprocessor["lighting"].transform(input_df["lighting"])
    input_df["weather"] = preprocessor["weather"].transform(input_df["weather"])

    prediction = model.predict(input_df)


    return jsonify({
        "prediction": float(prediction[0])
    })



if __name__ == "__main__":
    app.run(port=5001, debug=True)
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# 1. Load model, column order, scaler and numeric columns
model = joblib.load("Data Folder/gradient_boosting.pkl")
columns = joblib.load("Data Folder/columns.pkl")
scaler = joblib.load("Data Folder/scaler.pkl")
num_cols = joblib.load("Data Folder/num_cols.pkl")

@app.route("/")
def home():
    return jsonify({"message": "Churn Prediction API is running", "model": "Gradient Boosting"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # raw customer data, e.g. tenure=5

    # 2. Rebuild the engineered features exactly like in preprocess.py
    tenure = data.get("tenure", 0)
    total_charges = data.get("TotalCharges", data.get("MonthlyCharges", 0) * tenure)
    data["TotalCharges"] = total_charges
    data["AvgChargesPerMonth"] = total_charges / (tenure + 1)

    service_cols = ["PhoneService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    data["TotalServices"] = sum(1 for c in service_cols if data.get(c) == "Yes")

    # 3. One-hot encode text fields into the same columns used in training
    raw_df = pd.DataFrame([data])
    raw_df["TenureGroup"] = pd.cut(
        raw_df["tenure"], bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"], include_lowest=True,
    ).astype(str)
    encoded = pd.get_dummies(raw_df)
    input_df = encoded.reindex(columns=columns, fill_value=0).astype(float)

    # 4. Scale numeric columns using the saved scaler
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # 5. Predict
    probability = float(model.predict_proba(input_df)[0][1])
    prediction = int(probability >= 0.5)

    return jsonify({
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "result": "Customer likely to churn" if prediction == 1 else "Customer likely to stay",
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
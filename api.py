from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Customer Churn Prediction API")

# Load model, scaler, and feature names at startup
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

class CustomerData(BaseModel):
    # Using a flexible dict-based input since we have many one-hot encoded columns
    data: dict

@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    # Build a DataFrame with all expected feature columns, defaulting missing ones to 0
    input_df = pd.DataFrame([customer.data])
    
    # Ensure all expected columns exist, fill missing with 0
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training data
    input_df = input_df[feature_names]
    
    # Scale and predict
    input_scaled = scaler.transform(input_df)
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]
    
    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "risk_level": "High" if probability > 0.5 else "Low"
    }

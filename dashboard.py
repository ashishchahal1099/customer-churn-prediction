import streamlit as st
import pandas as pd
import joblib

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("Customer Churn Prediction Dashboard")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

if st.button("Predict Churn Risk"):
    input_data = {col: 0 for col in feature_names}
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly_charges
    input_data['TotalCharges'] = monthly_charges * max(tenure, 1)
    if contract == "One year":
        input_data['Contract_One year'] = 1
    elif contract == "Two year":
        input_data['Contract_Two year'] = 1
    if internet == "Fiber optic":
        input_data['InternetService_Fiber optic'] = 1
    elif internet == "No":
        input_data['InternetService_No'] = 1

    input_df = pd.DataFrame([input_data])[feature_names]
    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[0][1]

    st.metric("Churn Probability", f"{prob*100:.1f}%")
    st.write("Risk Level:", "🔴 High" if prob > 0.5 else "🟢 Low")

# Customer Churn Prediction

End-to-end ML pipeline to predict customer churn using the Telco Customer Churn dataset, with model explainability and a deployed REST API + interactive dashboard.

## Results
- **Model:** Logistic Regression (class_weight='balanced', scaled features)
- **ROC-AUC:** 0.84
- **Recall (churners):** 79% — prioritized over precision since missing a churner costs more than a false alarm
- Benchmarked against XGBoost (SMOTE and scale_pos_weight variants) — Logistic Regression outperformed on recall and ROC-AUC for this dataset

## Pipeline
1. Data cleaning (handling TotalCharges edge cases, encoding target)
2. EDA (churn drivers: contract type, tenure, monthly charges, internet service)
3. Feature engineering (tenure buckets, total services subscribed, avg monthly spend)
4. Train/test split with stratification + SMOTE (train-only, to avoid leakage)
5. Model comparison: Logistic Regression vs XGBoost (SMOTE and scale_pos_weight variants)
6. SHAP explainability (global feature importance + individual prediction breakdowns)
7. Deployed as a REST API (FastAPI) and an interactive dashboard (Streamlit)

## Tech Stack
Python, pandas, scikit-learn, XGBoost, SHAP, imbalanced-learn, FastAPI, Streamlit

## Usage
API: `uvicorn api:app --reload` → visit `/docs` for Swagger UI
Dashboard: `streamlit run dashboard.py`

## Example Results
High-risk profile (new customer, month-to-month, fiber optic, no add-ons): 92.95% churn probability
Low-risk profile (long tenure, two-year contract, DSL, auto-pay): 2.28% churn probability

import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

# Load model, scaler, and data
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

X_test = pd.read_csv("X_test.csv")
X_test_scaled = scaler.transform(X_test)

# SHAP explainer for linear models
explainer = shap.LinearExplainer(model, X_test_scaled)
shap_values = explainer.shap_values(X_test_scaled)

# 1. Summary plot - which features drive churn overall
plt.figure()
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.close()
print("Saved shap_summary.png")

# 2. Bar plot - feature importance ranked
plt.figure()
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_importance_bar.png")
plt.close()
print("Saved shap_importance_bar.png")

# 3. Explain one specific customer (first row in test set) - useful for interview demo
customer_idx = 0
plt.figure()
shap.force_plot(
    explainer.expected_value, shap_values[customer_idx], X_test.iloc[customer_idx],
    feature_names=feature_names, matplotlib=True, show=False
)
plt.tight_layout()
plt.savefig("shap_single_customer.png")
plt.close()
print("Saved shap_single_customer.png")

print("\nActual churn label for this customer:", pd.read_csv("y_test.csv").iloc[customer_idx].values[0])

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# Scale features - fixes convergence warning and improves LR performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Final model: Logistic Regression, class_weight balanced, more iterations
final_model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
final_model.fit(X_train_scaled, y_train)

preds = final_model.predict(X_test_scaled)
probs = final_model.predict_proba(X_test_scaled)[:, 1]

print("=== Final Model: Logistic Regression (scaled, class_weight=balanced) ===")
print(classification_report(y_test, preds))
print("ROC-AUC:", roc_auc_score(y_test, probs))

# Save model, scaler, and feature names together
joblib.dump(final_model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X_train.columns.tolist(), "feature_names.pkl")
print("\nSaved final model, scaler, and feature names.")

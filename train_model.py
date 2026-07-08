import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load splits
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

X_train_smote = pd.read_csv("X_train_smote.csv")
y_train_smote = pd.read_csv("y_train_smote.csv").values.ravel()

results = {}

# --- Model 1: Logistic Regression (baseline, class_weight balanced) ---
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_probs = lr.predict_proba(X_test)[:, 1]

print("=== Logistic Regression (class_weight='balanced') ===")
print(classification_report(y_test, lr_preds))
print("ROC-AUC:", roc_auc_score(y_test, lr_probs))
results['logreg'] = roc_auc_score(y_test, lr_probs)

# --- Model 2: XGBoost trained on SMOTE-balanced data ---
xgb_smote = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_smote.fit(X_train_smote, y_train_smote)
xgb_smote_preds = xgb_smote.predict(X_test)
xgb_smote_probs = xgb_smote.predict_proba(X_test)[:, 1]

print("\n=== XGBoost (trained on SMOTE data) ===")
print(classification_report(y_test, xgb_smote_preds))
print("ROC-AUC:", roc_auc_score(y_test, xgb_smote_probs))
results['xgb_smote'] = roc_auc_score(y_test, xgb_smote_probs)

# --- Model 3: XGBoost with scale_pos_weight (handles imbalance without SMOTE) ---
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_weighted = XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
xgb_weighted.fit(X_train, y_train)
xgb_weighted_preds = xgb_weighted.predict(X_test)
xgb_weighted_probs = xgb_weighted.predict_proba(X_test)[:, 1]

print("\n=== XGBoost (scale_pos_weight, no SMOTE) ===")
print(classification_report(y_test, xgb_weighted_preds))
print("ROC-AUC:", roc_auc_score(y_test, xgb_weighted_probs))
results['xgb_weighted'] = roc_auc_score(y_test, xgb_weighted_probs)

print("\n=== Summary (ROC-AUC) ===")
for name, score in results.items():
    print(f"{name}: {score:.4f}")

# Save best model (we'll pick based on your results, but saving xgb_weighted as default for now)
best_model = xgb_weighted
joblib.dump(best_model, "churn_model.pkl")
print("\nSaved best model to churn_model.pkl")

# Save feature names for later use in deployment
joblib.dump(X_train.columns.tolist(), "feature_names.pkl")

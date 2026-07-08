import pandas as pd
import numpy as np

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Fix TotalCharges: convert to numeric, coerce blanks to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check how many became NaN
print("Rows with NaN TotalCharges after conversion:", df['TotalCharges'].isnull().sum())

# These are usually customers with tenure = 0 (new customers, no bill yet)
print(df[df['TotalCharges'].isnull()][['tenure', 'MonthlyCharges', 'TotalCharges']])

# Fill NaN TotalCharges with 0 (since tenure=0 means no charges yet)
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Drop customerID - not predictive, just an identifier
df = df.drop('customerID', axis=1)

# Convert target to binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Confirm everything is clean now
print("\nFinal dtypes:")
print(df.dtypes)
print("\nAny remaining nulls:", df.isnull().sum().sum())

# Save cleaned data for next step
df.to_csv("churn_cleaned.csv", index=False)
print("\nSaved cleaned data to churn_cleaned.csv")

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Basic inspection
print("Shape:", df.shape)
print("\nColumn info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Check target distribution
print("\nChurn distribution:")
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True) * 100)

# Check for missing/weird values
print("\nMissing values:")
print(df.isnull().sum())
print("\nTotalCharges dtype:", df['TotalCharges'].dtype)

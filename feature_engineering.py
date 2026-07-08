import pandas as pd
import numpy as np

df = pd.read_csv("churn_cleaned.csv")

# 1. Derived feature: tenure buckets
def tenure_bucket(t):
    if t <= 12:
        return 'new'
    elif t <= 36:
        return 'medium'
    else:
        return 'long_term'

df['tenure_bucket'] = df['tenure'].apply(tenure_bucket)

# 2. Derived feature: total number of services subscribed
service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

def count_services(row):
    count = 0
    for col in service_cols:
        if row[col] not in ['No', 'No internet service', 'No phone service']:
            count += 1
    return count

df['total_services'] = df.apply(count_services, axis=1)

# 3. Derived feature: average monthly spend so far (avoids div by zero for tenure=0)
df['avg_monthly_spend'] = df['TotalCharges'] / df['tenure'].replace(0, 1)

# 4. Identify categorical columns for encoding
categorical_cols = df.select_dtypes(include='object').columns.tolist()
print("Categorical columns to encode:", categorical_cols)

# 5. One-hot encode categoricals (drop_first avoids multicollinearity)
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nShape after encoding:", df_encoded.shape)
print("\nColumns after encoding:")
print(df_encoded.columns.tolist())

# Save for modeling step
df_encoded.to_csv("churn_features.csv", index=False)
print("\nSaved to churn_features.csv")

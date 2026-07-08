import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

df = pd.read_csv("churn_features.csv")

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Stratified split - keeps the same churn ratio in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("\nTrain churn distribution:")
print(y_train.value_counts(normalize=True))
print("\nTest churn distribution:")
print(y_test.value_counts(normalize=True))

# Apply SMOTE only on training data (never on test data - that would leak info)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE - Train shape:", X_train_smote.shape)
print("After SMOTE - churn distribution:")
print(y_train_smote.value_counts(normalize=True))

# Save all splits for modeling step
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
X_train_smote.to_csv("X_train_smote.csv", index=False)
y_train_smote.to_csv("y_train_smote.csv", index=False)

print("\nAll splits saved.")

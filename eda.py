import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("churn_cleaned.csv")

sns.set_style("whitegrid")

# 1. Churn by Contract type
plt.figure(figsize=(7,5))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title("Churn by Contract Type")
plt.savefig("plot1_contract.png")
plt.close()

# 2. Churn by tenure (distribution)
plt.figure(figsize=(7,5))
sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30)
plt.title("Churn by Tenure")
plt.savefig("plot2_tenure.png")
plt.close()

# 3. Churn by Monthly Charges
plt.figure(figsize=(7,5))
sns.histplot(data=df, x='MonthlyCharges', hue='Churn', multiple='stack', bins=30)
plt.title("Churn by Monthly Charges")
plt.savefig("plot3_monthlycharges.png")
plt.close()

# 4. Churn by Internet Service type
plt.figure(figsize=(7,5))
sns.countplot(data=df, x='InternetService', hue='Churn')
plt.title("Churn by Internet Service")
plt.savefig("plot4_internet.png")
plt.close()

# 5. Correlation heatmap (numeric features only)
plt.figure(figsize=(6,5))
numeric_df = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plot5_correlation.png")
plt.close()

print("Saved 5 plots: plot1_contract.png, plot2_tenure.png, plot3_monthlycharges.png, plot4_internet.png, plot5_correlation.png")

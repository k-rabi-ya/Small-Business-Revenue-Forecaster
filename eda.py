# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("data/cleaned_panel_data.csv")

# Basic info
print("Shape:", data.shape)
print("\nColumns:\n", data.columns)
print("\nMissing Values:\n", data.isnull().sum())
print("\nSummary Statistics:\n", data.describe())

# Income Distribution
plt.figure(figsize=(6,4))
sns.histplot(data["income"], kde=True)
plt.title("Income Distribution")
plt.show()

# Correlation Matrix
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Income vs Employee Number
plt.figure(figsize=(6,4))
sns.scatterplot(x="employee_num", y="income", data=data)
plt.title("Employee Number vs Income")
plt.show()

# Income vs Firm Size
plt.figure(figsize=(6,4))
sns.scatterplot(x="firm_size", y="income", data=data)
plt.title("Firm Size vs Income")
plt.show()

print("\nEDA Completed Successfully")
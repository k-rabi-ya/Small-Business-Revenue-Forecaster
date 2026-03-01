import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("data/panel_data_digital_finance_clean.csv")

print("Initial Shape:", data.shape)

# Remove duplicate rows
duplicates = data.duplicated().sum()
print("Duplicate Rows:", duplicates)
data = data.drop_duplicates()

# Check missing values
print("\nMissing Values Before Cleaning:\n", data.isnull().sum())

# Fill numeric missing values with median
for col in data.select_dtypes(include=['int64', 'float64']).columns:
    data[col] = data[col].fillna(data[col].median())

# Drop remaining missing values if any
data = data.dropna()

print("\nMissing Values After Cleaning:\n", data.isnull().sum())

# Ensure important columns are numeric
numeric_columns = [
    "firm_size",
    "region_gdp",
    "high_tech",
    "digital_finance",
    "employee_num"
]

for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Remove negative values if not valid
for col in numeric_columns:
    data = data[data[col] >= 0]

# Outlier removal using IQR method
for col in numeric_columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

print("\nFinal Shape After Cleaning:", data.shape)

# Save cleaned dataset
data.to_csv("data/cleaned_panel_data.csv", index=False)

print("\nData Cleaning Completed Successfully")
import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("startup_failure_prediction.csv")

print("Initial Shape:", data.shape)

# Remove duplicate rows
data = data.drop_duplicates()

# Handle missing values
data = data.dropna()

# Automatically detect numeric columns
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

# Ensure numeric data types
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Drop rows that became NaN after conversion
data = data.dropna()

# Validate ranges (remove negative values)
for col in numeric_cols:
    data = data[data[col] >= 0]

# Outlier detection using IQR method
for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    data = data[(data[col] >= lower) & (data[col] <= upper)]

# Save cleaned dataset
data.to_csv("cleaned_data.csv", index=False)

print("Final Shape:", data.shape)
print("Cleaning completed successfully")
import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("dataset_micro.csv")

# Remove duplicates
data = data.drop_duplicates()

# Handle missing values
data = data.dropna()

# Ensure numeric types
numeric_cols = ["Annual Revenue ($M)", "Total Funding ($M)", "Number of Employees"]

for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna()

# Validate ranges (remove negative values)
data = data[
    (data["Annual Revenue ($M)"] >= 0) &
    (data["Total Funding ($M)"] >= 0) &
    (data["Number of Employees"] >= 0)
]

# Outlier detection using IQR
for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    data = data[(data[col] >= lower) & (data[col] <= upper)]

# Save cleaned dataset
data.to_csv("cleaned_data.csv", index=False)

print("Cleaning completed successfully")
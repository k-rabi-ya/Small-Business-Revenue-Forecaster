import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("data/cleaned_panel_data.csv")


# Feature Validation

print("Missing Values:\n", data.isnull().sum())

print("\nData Types:\n", data.dtypes)

print("\nBasic Statistics:\n", data.describe())

print("\nUnique Values per Column:")
for col in data.columns:
    print(col, ":", data[col].nunique())

print("\nDuplicate Rows:", data.duplicated().sum())

# Select features
X = data[[
    "firm_size",
    "region_gdp",
    "high_tech",
    "digital_finance",
    "employee_num"
]]

# Target
y = data["income"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.preprocessing import StandardScaler

# Standardization
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Train Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
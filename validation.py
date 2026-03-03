# ============================================
# EVALUATION OF SAVED MODEL (growth_model.joblib)
# ============================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. LOAD DATA
# -----------------------------

# adjust path to use dataset that actually exists in repository
# (located inside the data/ directory)
data = pd.read_csv("data/cleaned_panel_data.csv")


# -----------------------------
# 2. TRAIN-TEST SPLIT
# -----------------------------

# NOTE: X and y will be defined after bundle is loaded (see below)


# -----------------------------
# 4. LOAD SAVED MODEL
# -----------------------------

# compatibility: the saved model references numpy._core, which may not
# exist in newer numpy versions. create a shim module pointing to
# numpy.core so joblib can unpickle successfully.
import importlib, sys
import numpy as _np
if not hasattr(_np, "_core"):
    _np._core = importlib.import_module("numpy.core")
    sys.modules["numpy._core"] = _np._core

bundle = joblib.load("growth_model.joblib")
# training.py saved a dictionary containing model, scaler, features, etc.
# extract the model object for prediction.
model = bundle.get("model", bundle)

# determine feature columns and prepare dataset
if isinstance(bundle, dict) and "features" in bundle:
    feature_cols = bundle["features"]
else:
    feature_cols = [
        "digital_finance", "employee_num", "firm_size", "founder_edu", 
        "financing_difficulty", "region_gdp", "urban", "high_tech",
        "Finance_x_GDP", "Employee_Sq"
    ]

# replicate logic used during training to engineer features

def engineer_features(df, feature_cols):
    df = df.copy()
    if "Finance_x_GDP" in feature_cols and "Finance_x_GDP" not in df.columns:
        df['Finance_x_GDP'] = df['digital_finance'] * df['region_gdp']
    if "Employee_Sq" in feature_cols and "Employee_Sq" not in df.columns:
        df['Employee_Sq'] = df['employee_num'] ** 2
    return df[feature_cols]

X = engineer_features(data, feature_cols)
y = data["income"]

# split for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scaling using bundle's scaler if available
if isinstance(bundle, dict) and "scaler" in bundle:
    scaler = bundle["scaler"]
else:
    scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. PREDICTIONS
# -----------------------------

y_pred = model.predict(X_test_scaled)

# -----------------------------
# 6. EVALUATION METRICS
# -----------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("========== SAVED MODEL EVALUATION ==========")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# -----------------------------
# 7. ADJUSTED R2
# -----------------------------

n = X_test.shape[0]
p = X_test.shape[1]

adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("Adjusted R2:", adjusted_r2)

# -----------------------------
# 8. OVERFITTING CHECK
# -----------------------------

train_r2 = r2_score(y_train, model.predict(X_train_scaled))
print("\nTrain R2:", train_r2)
print("Test R2:", r2)

# -----------------------------
# 9. CROSS VALIDATION
# -----------------------------

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(model, X_train_scaled, y_train,
                            scoring='r2', cv=kf)

print("\nCross Validation R2:", cv_scores.mean())

# -----------------------------
# 10. RESIDUAL PLOT
# -----------------------------

residuals = y_test - y_pred

plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot - Saved Model")
plt.show()

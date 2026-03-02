import pandas as pd
import numpy as np
import os
import joblib  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#We us classes to make the program more modular
class ModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.model = LinearRegression()
        self.feature_cols = [
            "digital_finance", "employee_num", "firm_size", "founder_edu", 
            "financing_difficulty", "region_gdp", "urban", "high_tech",
            "Finance_x_GDP", "Employee_Sq"
        ]

    
    def engineer_features(self, df):
        df = df.copy()
        df['Finance_x_GDP'] = df['digital_finance'] * df['region_gdp']  #something like a banking app is more usefull when in a place that has the infrastructure to use it
        df['Employee_Sq'] = df['employee_num'] ** 2  #we square employees to simulate diminishing returns
        return df[self.feature_cols]

    def train_and_save(self, model_name="growth_model.joblib"):
        # Load and Filter data to less than 100 employees
        df = pd.read_csv(self.data_path)
        df = df[df['employee_num'] <= 100].copy()
        
        X = self.engineer_features(df)
        y = df["income"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        
        # Metrics Calculation
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        metrics_dict = {
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "sample_size": len(df)
        }
        

        print("\n" + "="*50)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Dataset Size:      {len(df)} micro-enterprises")
        print(f"R-Squared (R²):    {r2:.4f} (Reliability)")
        print(f"Mean Abs Error:    {mae:.4f} (Log Units)")
        print(f"Root Mean Sq Err:  {rmse:.4f}")
        print("-" * 50)
        
        # Displaying most important parameters
        importance = pd.DataFrame({
            'Feature': self.feature_cols,
            'Weight': self.model.coef_
        }).sort_values(by='Weight', ascending=False)
        
        print("Main growth (Coefficients):")
        print(importance.to_string(index=False))
        print("="*50 + "\n")

        # Save the bundle
        model_bundle = {
            "model": self.model,
            "scaler": self.scaler,
            "features": self.feature_cols,
            "metrics": metrics_dict
        }
        
        joblib.dump(model_bundle, model_name)
        print(f"SUCCESSFULLY SAVED BUNDLE AS: {model_name}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "data", "panel_data_augmented.csv")
    trainer = ModelTrainer(data_file)
    trainer.train_and_save()
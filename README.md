# Small Business Revenue Forecaster

## Project Overview
This project predicts small business income using regression techniques.  
The goal is to analyze financial and firm-level features to estimate revenue.



## Dataset
Panel dataset containing:

- firm_id
- year
- income (target variable)
- digital_finance
- firm_size
- employee_num
- founder_edu
- financing_difficulty
- region_gdp
- urban
- high_tech

No missing values were found in the dataset.



##  Data Cleaning
- Checked for missing values
- Verified numeric data types
- Removed invalid values
- Applied IQR method for outlier detection
- Saved cleaned dataset



## Exploratory Data Analysis (EDA)
- Summary statistics
- Distribution plots
- Correlation matrix
- Feature relationship analysis



## Model Used
Linear Regression

Train-Test Split: 80% - 20%


## Evaluation Metrics

- MAE: 0.83  
- MSE: 1.10  
- RMSE: 1.05  
- R² Score: 0.59  

The model explains approximately 59% of the variance in income.



## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Conda (Environment Management)



## Conclusion
The regression model performs reasonably well for a mini project.  
Future improvements could include advanced models like Random Forest or Gradient Boosting.
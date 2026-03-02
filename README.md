# Small Business Revenue Forecaster

## Project Overview
This project implements an econometric machine learning pipeline to predict the annual revenue of micro-enterprises. It utilizes an object-oriented structure to analyze how firm-level attributes and macro-economic factors influence income growth.

The goal is to provide a strategic tool for forecasting revenue based on financial inclusion, labor scale, and regional infrastructure.

---

## Dataset Specifications
The model is trained on a panel dataset containing the following features:

- **firm_id**: Unique identifier for each enterprise.
- **year**: Temporal marker for longitudinal analysis.
- **income**: Target variable (Log-transformed annual revenue).
- **digital_finance**: Index of digital banking and payment tool usage.
- **firm_size**: Relative scale of physical assets and property.
- **employee_num**: Total workforce count.
- **founder_edu**: Ordinal scale of the founder's formal education.
- **financing_difficulty**: Index representing barriers to formal credit access.
- **region_gdp**: Proxy for local economic strength and infrastructure.
- **urban**: Binary indicator for geographic location (Urban vs. Rural).
- **high_tech**: Binary indicator for specialized technical operations.

---

## Data Cleaning and Feature Engineering

- **Missing Values**: Verified zero null entries in the primary dataset.
- **Outliers**: Applied the Interquartile Range (IQR) method for outlier detection and removal to ensure model stability.
- **Scaling**: Utilized `StandardScaler` to normalize features, ensuring consistent coefficient weighting.
- **Engineered Features**:
  - **Finance_x_GDP**: Interaction term modeling the synergy between digital tools and regional economic maturity.
  - **Employee_Sq**: Polynomial feature simulating non-linear labor returns and diminishing marginal productivity.

---

## Model Architecture and Evaluation

The system utilizes a standardized Linear Regression model implemented via a modular `ModelTrainer` class. This approach ensures code reusability and clear separation of concerns.

**Performance Metrics (Optimized Model):**

- **R² Score**: 0.6742 (The model explains approximately 67% of income variance.)
- **MAE**: 0.7412 (Mean Absolute Error.)
- **RMSE**: 0.9104 (Root Mean Squared Error.)

---

## Installation and Setup

### Environment Management

It is recommended to use a virtual environment to avoid dependency conflicts.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Dependency Installation

Install all required libraries using the provided `requirements.txt` file:

```powershell
pip install -r requirements.txt
```

---

## Execution Flow

### 1. Data Augmentation

Generate synthetic observations for new startups to expand the training dataset:

```powershell
python augment_data.py
```

### 2. Model Training

Execute the OOP-based trainer to perform feature engineering, scaling, training, and model serialization. This process generates:

```
growth_model.joblib
```

Run:

```powershell
python training.py
```

### 3. Application Launch

Run the interactive Streamlit dashboard for real-time revenue forecasting:

```powershell
streamlit run app.py
```

---

## Technologies Used

- **Python**: Core programming language.
- **Pandas and NumPy**: Data manipulation and numerical analysis.
- **Scikit-learn**: Machine learning and preprocessing.
- **Streamlit**: Dashboard deployment and user interface.
- **Joblib**: Model serialization and persistence.
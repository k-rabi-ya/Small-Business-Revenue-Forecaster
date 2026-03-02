import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# basic page config
st.set_page_config(page_title="Micro Enterprises Revenue Predictor", layout="wide")

# custom styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    [data-testid="stMetricValue"] { font-size: 26px; color: #1E3A8A; font-weight: bold; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #1E3A8A; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# loading the bundle from training.py
@st.cache_resource
def load_bundle():
    try:
        return joblib.load("growth_model.joblib")
    except:
        return None

bundle = load_bundle()

if not bundle:
    st.error("❌ Model file not found. Run 'train_model.py' first.")
    st.stop()

model, scaler, features = bundle["model"], bundle["scaler"], bundle["features"]
m = bundle.get("metrics", {"r2": 0.0, "mae": 0.0, "rmse": 0.0, "sample_size": "Unknown"})

st.title("Business Revenue & Growth Predictor")
st.markdown("### Strategic Forecasting Tool")

# metrics bar
st.divider()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Model Reliability", f"{m['r2']:.1%}", help="How much variance the model explains.")
k2.metric("Avg Error Margin", f"{m['mae']:.3f}")
k3.metric("Calculation Speed", "Instant")
k4.metric("Firms Studied", f"{m['sample_size']}")

st.divider()

col_internal, col_external = st.columns(2, gap="large")

with col_internal:
    st.subheader("Business Details")
    emp = st.slider("Total Number of Employees", 1, 100, 10)
    size = st.slider("Business Equipment & Property Scale", 0.1, 5.0, 1.2, step=0.1)
    
    edu_map = {"Primary School": 1, "High School": 2, "University Degree": 3, "Advanced Degree": 4}
    edu_label = st.selectbox("Owner's Education Level", options=list(edu_map.keys()), index=2)
    edu = edu_map[edu_label]
    
    tech = st.toggle("Is this a High-Tech Business?")

with col_external:
    st.subheader("🌐 External Economic Factors")
    digi = st.slider("Online Banking & Digital Tools Usage", 0.0, 1.0, 0.5, step=0.01)
    gdp = st.slider("Local Economic Strength", 4.0, 20.0, 11.0, step=0.1)
    diff = st.slider("Difficulty Obtaining Business Loans", 0.0, 1.0, 0.4, step=0.01)

    urban_loc = st.radio("Business Location", ["Rural / Countryside", "Urban / City"], index=1, horizontal=True)
    urban = 1 if "Urban" in urban_loc else 0

with st.sidebar:
    st.header("Live Model Inputs")
    st.write(f"**Digital Usage:** `{digi:.2f}`")
    st.write(f"**GDP Strength:** `{gdp:.1f}`")
    st.write(f"**Loan Friction:** `{diff:.2f}`")
    st.write(f"**Employee Sq:** `{emp**2}`")
    st.divider()
    interaction = digi * gdp 
    st.metric("Tech-Economy Score", f"{interaction:.2f}")

st.divider()

# Prediction Logic
if st.button("Calculate Projected Annual Revenue", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "digital_finance": digi, "employee_num": emp, "firm_size": size,
        "founder_edu": edu, "financing_difficulty": diff, "region_gdp": gdp,
        "urban": urban, "high_tech": 1 if tech else 0,
        "Finance_x_GDP": interaction, "Employee_Sq": emp**2
    }])

    scaled_input = scaler.transform(input_df[features])
    # np.exp reverses the log(income) target used during training
    prediction = np.exp(model.predict(scaled_input)[0])

    res_c1, res_c2 = st.columns([2, 1])
    with res_c1:
        st.markdown("### Estimated Annual Income")
        st.title(f":green[${prediction:,.2f}]")
    with res_c2:
        st.info("Results calculated based on the R² reliability above.")

st.divider()

# Visualization Section
with st.expander("View Model Insights & Drivers"):
    st.write("### Which factors drive growth in this model?")
    
    # capturing the weights from the linear regression model
    importance_df = pd.DataFrame({
        'Factor': features,
        'Impact (Weight)': model.coef_
    }).sort_values(by='Impact (Weight)', ascending=False)
    
    # interactive bar chart to show feature importance
    st.bar_chart(importance_df.set_index('Factor'), height=500, use_container_width=True)
    
    st.caption("Positive values increase revenue; negative values represent barriers to growth.")
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('best_carbon_model.pkl')

model = load_model()

# Page config
st.set_page_config(page_title="AgriCarbon Predictor", page_icon="🌱", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        border: none;
    }
    .stMetric {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 Agricultural Carbon Emission Predictor")
st.markdown("Estimate the environmental impact of your farming practices using Machine Learning.")

# Sidebar info
st.sidebar.header("About")
st.sidebar.info("This tool uses a Gradient Boosting Regressor trained on synthetic agricultural data to estimate carbon footprints.")

# Layout: Two columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("🚜 Farming Practices")
    
    farm_size = st.number_input("Farm Size (acres)", min_value=1.0, max_value=1000.0, value=50.0)
    
    crop_type = st.selectbox("Crop Type", ["Rice", "Wheat", "Maize", "Soybeans", "Cotton"])
    
    soil_type = st.selectbox("Soil Type", ["Clay", "Sandy", "Loamy", "Silty", "Peaty"])
    
    irrigation_type = st.selectbox("Irrigation Type", ["Drip", "Sprinkler", "Flood"])
    
    livestock_count = st.number_input("Livestock Count", min_value=0, max_value=5000, value=10)

with col2:
    st.subheader("⚡ Resource Usage")
    
    fertilizer = st.number_input("Fertilizer Usage (kg)", min_value=0.0, max_value=10000.0, value=200.0)
    
    pesticide = st.number_input("Pesticide Usage (liters)", min_value=0.0, max_value=1000.0, value=10.0)
    
    fuel = st.number_input("Fuel Consumption (liters)", min_value=0.0, max_value=10000.0, value=100.0)
    
    electricity = st.number_input("Electricity Consumption (kWh)", min_value=0.0, max_value=50000.0, value=500.0)
    
    transport = st.number_input("Transportation Distance (km)", min_value=0.0, max_value=1000.0, value=50.0)

# Prediction Logic
input_data = {
    'Farm_Size_acres': farm_size,
    'Crop_Type': crop_type,
    'Soil_Type': soil_type,
    'Fertilizer_Usage_kg': fertilizer,
    'Pesticide_Usage_liters': pesticide,
    'Fuel_Consumption_liters': fuel,
    'Irrigation_Type': irrigation_type,
    'Electricity_Consumption_kWh': electricity,
    'Livestock_Count': livestock_count,
    'Transportation_Distance_km': transport
}

if st.button("Calculate Carbon Score"):
    df_input = pd.DataFrame([input_data])
    prediction = model.predict(df_input)[0]
    
    # Classification
    if prediction < 50000:
        impact = "Low"
        color = "green"
    elif prediction < 200000:
        impact = "Medium"
        color = "orange"
    else:
        impact = "High"
        color = "red"
    
    st.divider()
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric("Estimated Emissions", f"{prediction:,.2f} kg CO2e")
    
    with res_col2:
        st.markdown(f"### Impact Level: <span style='color:{color}'>{impact}</span>", unsafe_allow_html=True)
    
    # Visualization or suggestion
    st.info("💡 Tip: To reduce emissions, consider switch to Drip irrigation or reducing synthetic fertilizer use.")

st.markdown("---")
st.caption("Developed by AI ML Engineering Assistant")

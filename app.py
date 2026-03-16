import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Absolute path for the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_carbon_model.pkl')

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except:
        return None

model = load_model()

# Page config
st.set_page_config(page_title="AgriCarbon AI", page_icon="🌍", layout="wide")

# Enhanced Premium CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f0f7f4 0%, #e2ece9 100%);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.5);
        border-radius: 10px 10px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        border: none;
        color: #2e7d32;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2e7d32 !important;
        color: white !important;
    }

    .prediction-card {
        padding: 30px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        text-align: center;
        margin-bottom: 20px;
    }

    .metric-label {
        color: #666;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        color: #2e7d32;
        font-size: 2.5em;
        font-weight: 700;
        margin: 10px 0;
    }

    .impact-low { color: #2ecc71; font-weight: 600; }
    .impact-medium { color: #f1c40f; font-weight: 600; }
    .impact-high { color: #e74c3c; font-weight: 600; }

    .stButton>button {
        background: linear-gradient(45deg, #2ecc71, #27ae60);
        color: white;
        border-radius: 12px;
        font-weight: 600;
        border: none;
        padding: 12px 24px;
        transition: transform 0.2s;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(46, 204, 113, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div style="text-align: center; padding: 20px 0 40px 0;">
        <h1 style="color: #1b5e20; font-size: 3em;">🌍 AgriCarbon AI Predictor</h1>
        <p style="color: #4caf50; font-size: 1.2em;">Advanced Climate & Agricultural Emission Analysis</p>
    </div>
    """, unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model file not found! Please wait while the model is being trained or run `python ml_pipeline.py` manually.")
    st.info("I am currently re-training the model with the new dataset schema...")
    st.stop()

# Tabs for different prediction modes
tab1, tab2 = st.tabs(["📝 Manual Input", "📊 Bulk Dataset Prediction"])

with tab1:
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
    
    with col1:
        st.markdown("### 🗺️ Location & Context")
        year = st.number_input("Year", min_value=2000, max_value=2050, value=2024)
        country = st.selectbox("Country", ['USA', 'India', 'Brazil', 'China', 'Australia', 'France'])
        region = st.selectbox("Region", ['North', 'South', 'East', 'West', 'Central'])
        crop_type = st.selectbox("Crop Type", ['Wheat', 'Rice', 'Corn', 'Soybeans', 'Cotton'])

    with col2:
        st.markdown("### 🌡️ Environment & Soil")
        avg_temp = st.number_input("Average Temperature (°C)", value=25.0)
        total_precip = st.number_input("Total Precipitation (mm)", value=1000.0)
        extreme_weather = st.selectbox("Extreme Weather Events", ['None', 'Drought', 'Flood', 'Heatwave', 'Storm'])
        soil_health = st.slider("Soil Health Index (0-100)", 0.0, 100.0, 75.0)

    with col3:
        st.markdown("### 🚜 Inputs & Economy")
        irrigation_access = st.slider("Irrigation Access (%)", 0.0, 100.0, 50.0)
        pesticide_use = st.number_input("Pesticide Use (kg/ha)", value=5.0)
        fertilizer_use = st.number_input("Fertilizer Use (kg/ha)", value=150.0)
        crop_yield = st.number_input("Expected Crop Yield (tons/ha)", value=5.0)
        adaptation = st.selectbox("Adaptation Strategy", ['Crop Rotation', 'Drip Irrigation', 'No-Till Farming', 'Multiple Cropping', 'None'])
        economic_impact = st.number_input("Economic Impact (Million USD)", value=2.5)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Generate Climate Impact Prediction", key="predict_manual"):
        input_data = {
            'Year': year,
            'Country': country,
            'Region': region,
            'Crop_Type': crop_type,
            'Average_Temperature': avg_temp,
            'Total_Precipitation': total_precip,
            'Crop_Yield': crop_yield,
            'Extreme_Weather_Events': extreme_weather,
            'Irrigation_Access': irrigation_access,
            'Pesticide_Use': pesticide_use,
            'Fertilizer_Use': fertilizer_use,
            'Soil_Health_Index': soil_health,
            'Adaptation_Strategies': adaptation,
            'Economic_Impact_Million_USD': economic_impact
        }
        
        df_input = pd.DataFrame([input_data])
        
        try:
            prediction = model.predict(df_input)[0]
            
            # Classification logic based on new scale
            if prediction < 10000:
                impact, cls = "Low Impact", "impact-low"
            elif prediction < 20000:
                impact, cls = "Medium Impact", "impact-medium"
            else:
                impact, cls = "High Impact", "impact-high"
                
            st.markdown(f"""
                <div class="prediction-card">
                    <div class="metric-label">Estimated CO₂ Emissions</div>
                    <div class="metric-value">{prediction:,.2f} <span style="font-size: 0.4em;">kg CO₂e</span></div>
                    <div class="{cls}" style="font-size: 1.5em;">{impact}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.success("Analysis complete! The prediction reflects the combined impact of climate factors and agricultural inputs.")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

with tab2:
    st.markdown("### ☁️ Bulk Prediction from File")
    st.write("Upload your dataset matching the new schema to analyze multiple records at once.")
    
    with st.expander("📌 View Required CSV Format"):
        st.write("The CSV should contain the following columns:")
        st.code("Year,Country,Region,Crop_Type,Average_Temperature,Total_Precipitation,Crop_Yield,Extreme_Weather_Events,Irrigation_Access,Pesticide_Use,Fertilizer_Use,Soil_Health_Index,Adaptation_Strategies,Economic_Impact_Million_USD")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            bulk_df = pd.read_csv(uploaded_file)
            
            # Use columns expected by the model
            required_cols = [
                'Year', 'Country', 'Region', 'Crop_Type', 'Average_Temperature', 
                'Total_Precipitation', 'Crop_Yield', 'Extreme_Weather_Events', 
                'Irrigation_Access', 'Pesticide_Use', 'Fertilizer_Use', 
                'Soil_Health_Index', 'Adaptation_Strategies', 'Economic_Impact_Million_USD'
            ]
            
            # Case insensitive check and renaming
            bulk_df.columns = [c.strip() for c in bulk_df.columns]
            
            if all(col in bulk_df.columns for col in required_cols):
                with st.spinner("Analyzing dataset..."):
                    predictions = model.predict(bulk_df[required_cols])
                    bulk_df['Predicted_CO2_Emissions'] = predictions
                    
                    # Impact classification
                    bulk_df['Impact_Level'] = pd.cut(bulk_df['Predicted_CO2_Emissions'], 
                                                     bins=[0, 10000, 20000, np.inf], 
                                                     labels=['Low', 'Medium', 'High'])
                
                st.markdown("### ✅ Prediction Results")
                st.dataframe(bulk_df.head(10), use_container_width=True)
                
                # Visualizations
                v_col1, v_col2 = st.columns(2)
                
                with v_col1:
                    st.write("#### CO2 Emission Distribution")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.histplot(bulk_df['Predicted_CO2_Emissions'], kde=True, color='#2ecc71', ax=ax)
                    st.pyplot(fig)
                
                with v_col2:
                    st.write("#### Impact Level Distribution")
                    impact_counts = bulk_df['Impact_Level'].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.pie(impact_counts, labels=impact_counts.index, autopct='%1.1f%%', 
                           colors=['#2ecc71', '#f1c40f', '#e74c3c'], startangle=140)
                    st.pyplot(fig)
                
                # Download link
                csv = bulk_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Processed Results", data=csv, 
                                   file_name="bulk_emissions_results.csv", mime="text/csv")
            else:
                missing = set(required_cols) - set(bulk_df.columns)
                st.error(f"❌ Missing columns: {', '.join(missing)}")
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #888;">
        <p>Built with ❤️ for a Greener Planet | <a href="https://github.com/Adhish31/co2.git" style="color: #2e7d32;">GitHub Repository</a></p>
    </div>
    """, unsafe_allow_html=True)

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
    return joblib.load(MODEL_PATH)

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
        <p style="color: #4caf50; font-size: 1.2em;">Advanced Machine Learning for Sustainable Agriculture</p>
    </div>
    """, unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model file not found! Please run `python ml_pipeline.py` to train the model first.")
    st.stop()

# Tabs for different prediction modes
tab1, tab2 = st.tabs(["📝 Manual Input", "📊 Bulk Dataset Prediction"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 🚜 Farming Activities")
        farm_size = st.number_input("Farm Size (acres)", min_value=1.0, value=50.0, step=1.0)
        crop_type = st.selectbox("Crop Type", ["Rice", "Wheat", "Maize", "Soybeans", "Cotton"])
        soil_type = st.selectbox("Soil Type", ["Clay", "Sandy", "Loamy", "Silty", "Peaty"])
        irrigation_type = st.selectbox("Irrigation Type", ["Drip", "Sprinkler", "Flood"])
        livestock_count = st.number_input("Livestock Count", min_value=0, value=10)

    with col2:
        st.markdown("### ⚡ Resource & Energy")
        fertilizer = st.number_input("Fertilizer Usage (kg)", min_value=0.0, value=200.0)
        pesticide = st.number_input("Pesticide Usage (liters)", min_value=0.0, value=10.0)
        fuel = st.number_input("Fuel Consumption (liters)", min_value=0.0, value=100.0)
        electricity = st.number_input("Electricity Consumption (kWh)", min_value=0.0, value=500.0)
        transport = st.number_input("Transportation Distance (km)", min_value=0.0, value=50.0)

    if st.button("Generate Prediction", key="predict_manual"):
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
        
        df_input = pd.DataFrame([input_data])
        prediction = model.predict(df_input)[0]
        
        # Classification
        if prediction < 50000:
            impact, cls = "Low Impact", "impact-low"
        elif prediction < 200000:
            impact, cls = "Medium Impact", "impact-medium"
        else:
            impact, cls = "High Impact", "impact-high"
            
        st.markdown(f"""
            <div class="prediction-card">
                <div class="metric-label">Estimated Carbon Footprint</div>
                <div class="metric-value">{prediction:,.2f} <span style="font-size: 0.4em;">kg CO₂e</span></div>
                <div class="{cls}" style="font-size: 1.5em;">{impact}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.success("Analysis complete! Consider optimizing fertilizer use to lower this score.")

with tab2:
    st.markdown("### ☁️ Bulk Prediction from File")
    st.write("Upload a CSV file containing your agricultural records. Ensure the column names match our required format.")
    
    with st.expander("📌 View Required CSV Format"):
        st.code("""Farm_Size_acres, Crop_Type, Soil_Type, Fertilizer_Usage_kg, Pesticide_Usage_liters, Fuel_Consumption_liters, Irrigation_Type, Electricity_Consumption_kWh, Livestock_Count, Transportation_Distance_km""")
        st.write("*Note: Crop_Type options: Rice, Wheat, Maize, Soybeans, Cotton*")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            bulk_df = pd.read_csv(uploaded_file)
            
            # Basic validation
            required_cols = ['Farm_Size_acres', 'Crop_Type', 'Soil_Type', 'Fertilizer_Usage_kg', 
                            'Pesticide_Usage_liters', 'Fuel_Consumption_liters', 'Irrigation_Type', 
                            'Electricity_Consumption_kWh', 'Livestock_Count', 'Transportation_Distance_km']
            
            if all(col in bulk_df.columns for col in required_cols):
                with st.spinner("Processing Large Dataset..."):
                    predictions = model.predict(bulk_df[required_cols])
                    bulk_df['Predicted_CO2e'] = predictions
                    
                    # Impact classification
                    bulk_df['Impact_Level'] = pd.cut(bulk_df['Predicted_CO2e'], 
                                                     bins=[0, 50000, 200000, np.inf], 
                                                     labels=['Low', 'Medium', 'High'])
                
                st.markdown("### ✅ Prediction Results")
                st.dataframe(bulk_df.head(10), use_container_width=True)
                
                # Visualizations for bulk data
                v_col1, v_col2 = st.columns(2)
                
                with v_col1:
                    st.write("#### Emission Distribution")
                    fig, ax = plt.subplots()
                    sns.histplot(bulk_df['Predicted_CO2e'], kde=True, color='#2ecc71', ax=ax)
                    st.pyplot(fig)
                
                with v_col2:
                    st.write("#### Impact Level Summary")
                    impact_counts = bulk_df['Impact_Level'].value_counts()
                    fig, ax = plt.subplots()
                    ax.pie(impact_counts, labels=impact_counts.index, autopct='%1.1f%%', 
                           colors=['#2ecc71', '#f1c40f', '#e74c3c'])
                    st.pyplot(fig)
                
                # Download link
                csv = bulk_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Full Results CSV", data=csv, 
                                   file_name="predicted_emissions.csv", mime="text/csv")
            else:
                st.error("❌ Column names in CSV do not match the required format. Please check the expander above.")
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #888;">
        <p>Built with ❤️ for a Greener Planet | <a href="https://github.com/Adhish31/co2.git" style="color: #2e7d32;">GitHub Repository</a></p>
    </div>
    """, unsafe_allow_html=True)

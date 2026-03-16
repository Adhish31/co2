import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

# Configuration
n_samples = 1500

# Feature Generation
data = {
    'Farm_Size_acres': np.random.uniform(5, 500, n_samples),
    'Crop_Type': np.random.choice(['Rice', 'Wheat', 'Maize', 'Soybeans', 'Cotton'], n_samples),
    'Soil_Type': np.random.choice(['Clay', 'Sandy', 'Loamy', 'Silty', 'Peaty'], n_samples),
    'Fertilizer_Usage_kg': np.random.uniform(50, 500, n_samples),
    'Pesticide_Usage_liters': np.random.uniform(1, 20, n_samples),
    'Fuel_Consumption_liters': np.random.uniform(30, 2000, n_samples),
    'Irrigation_Type': np.random.choice(['Drip', 'Sprinkler', 'Flood'], n_samples),
    'Electricity_Consumption_kWh': np.random.uniform(100, 5000, n_samples),
    'Livestock_Count': np.random.randint(0, 100, n_samples),
    'Transportation_Distance_km': np.random.uniform(10, 300, n_samples)
}

df = pd.DataFrame(data)

# Constants for emission calculation
EMISSION_FACTORS = {
    'Fertilizer_kg': 5.5, # kg CO2e per kg
    'Pesticide_liter': 12.0,
    'Fuel_liter': 2.7,
    'Electricity_kWh': 0.6,
    'Livestock_head': 1500, # Yearly avg
    'Transport_km': 0.3
}

CROP_MULTIPLIERS = {'Rice': 2.2, 'Wheat': 1.0, 'Maize': 1.3, 'Soybeans': 0.9, 'Cotton': 1.4}
IRRIGATION_MULTIPLIERS = {'Flood': 1.5, 'Sprinkler': 1.1, 'Drip': 0.8}
SOIL_MULTIPLIERS = {'Peaty': 1.4, 'Clay': 1.2, 'Loamy': 1.0, 'Silty': 1.1, 'Sandy': 0.9}

def calculate_emissions(row):
    # Base emissions
    base = (row['Fertilizer_Usage_kg'] * EMISSION_FACTORS['Fertilizer_kg'] +
            row['Pesticide_Usage_liters'] * EMISSION_FACTORS['Pesticide_liter'] +
            row['Fuel_Consumption_liters'] * EMISSION_FACTORS['Fuel_liter'] +
            row['Electricity_Consumption_kWh'] * EMISSION_FACTORS['Electricity_kWh'] +
            row['Livestock_Count'] * EMISSION_FACTORS['Livestock_head'] +
            row['Transportation_Distance_km'] * EMISSION_FACTORS['Transport_km'])
    
    # Apply multipliers
    total = base * CROP_MULTIPLIERS[row['Crop_Type']]
    total *= IRRIGATION_MULTIPLIERS[row['Irrigation_Type']]
    total *= SOIL_MULTIPLIERS[row['Soil_Type']]
    
    # Scale by farm size (baseline is 100 acres)
    total *= (row['Farm_Size_acres'] / 100)
    
    # Add noise
    noise = np.random.normal(0, total * 0.05)
    return max(0, total + noise)

df['Carbon_Emission_Score_kg'] = df.apply(calculate_emissions, axis=1)

# Save
output_path = 'agricultural_emissions.csv'
df.to_csv(output_path, index=False)
print(f"Dataset saved to {output_path}")
print(df.head())

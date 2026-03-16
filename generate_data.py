import pandas as pd
import numpy as np
import random

def generate_agricultural_climate_data(num_samples=2000):
    np.random.seed(42)
    
    countries = ['USA', 'India', 'Brazil', 'China', 'Australia', 'France']
    regions = ['North', 'South', 'East', 'West', 'Central']
    crops = ['Wheat', 'Rice', 'Corn', 'Soybeans', 'Cotton']
    weather_events = ['None', 'Drought', 'Flood', 'Heatwave', 'Storm']
    strategies = ['Crop Rotation', 'Drip Irrigation', 'No-Till Farming', 'Multiple Cropping', 'None']
    
    data = []
    
    for _ in range(num_samples):
        year = random.randint(2000, 2024)
        country = random.choice(countries)
        region = random.choice(regions)
        crop = random.choice(crops)
        
        # Environmental factors
        avg_temp = np.random.normal(25, 5) # degrees Celsius
        total_precip = np.random.normal(1000, 300) # mm
        
        # Operational factors
        pesticide_use = np.random.gamma(2, 5) # kg/ha
        fertilizer_use = np.random.gamma(5, 40) # kg/ha
        soil_health = np.random.uniform(30, 90) # index 0-100
        irrigation_access = np.random.uniform(0, 100) # percentage
        
        extreme_weather = random.choice(weather_events)
        adaptation = random.choice(strategies)
        
        # Calculate Crop Yield based on inputs (simplified logic)
        # Higher temp/low precip/extreme weather reduces yield
        # Better soil/fertilizer/irrigation improves yield
        yield_base = 5.0
        yield_mod = (soil_health/50) + (fertilizer_use/200) + (irrigation_access/100)
        if extreme_weather != 'None':
            yield_mod *= 0.6
        crop_yield = max(0.5, yield_base * yield_mod + np.random.normal(0, 0.5))
        
        # Calculate CO2 Emissions (The Target)
        # Fertilizer and Pesticide are big contributors
        # Irrigation/Transportation (implied) contributes
        # Crop type matters
        emissions = (fertilizer_use * 15.5) + (pesticide_use * 25.0) + (avg_temp * 100)
        if crop == 'Rice': emissions *= 1.5 # Rice has higher methane
        if crop == 'Cotton': emissions *= 1.2
        
        emissions += np.random.normal(0, 500)
        emissions = max(1000, emissions)
        
        # Economic Impact
        economic_impact = (crop_yield * 500) - (emissions * 0.1)
        if extreme_weather != 'None':
            economic_impact -= 200 # damage costs
            
        economic_impact = max(10, economic_impact / 100) # scale to Millions
        
        data.append({
            'Year': year,
            'Country': country,
            'Region': region,
            'Crop_Type': crop,
            'Average_Temperature': round(avg_temp, 2),
            'Total_Precipitation': round(total_precip, 2),
            'CO2_Emissions': round(emissions, 2),
            'Crop_Yield': round(crop_yield, 2),
            'Extreme_Weather_Events': extreme_weather,
            'Irrigation_Access': round(irrigation_access, 2),
            'Pesticide_Use': round(pesticide_use, 2),
            'Fertilizer_Use': round(fertilizer_use, 2),
            'Soil_Health_Index': round(soil_health, 2),
            'Adaptation_Strategies': adaptation,
            'Economic_Impact_Million_USD': round(economic_impact, 2)
        })
        
    df = pd.DataFrame(data)
    df.to_csv('agricultural_emissions.csv', index=False)
    print(f"Generated {num_samples} samples and saved to 'agricultural_emissions.csv'")
    return df

if __name__ == "__main__":
    generate_agricultural_climate_data()

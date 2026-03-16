import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load Dataset
df = pd.read_csv('agricultural_emissions.csv')

# 3. Data Preprocessing & Model Training
X = df.drop('CO2_Emissions', axis=1)
y = df['CO2_Emissions']

numeric_features = [
    'Year', 'Average_Temperature', 'Total_Precipitation', 
    'Crop_Yield', 'Irrigation_Access', 'Pesticide_Use', 
    'Fertilizer_Use', 'Soil_Health_Index', 'Economic_Impact_Million_USD'
]
categorical_features = ['Country', 'Region', 'Crop_Type', 'Extreme_Weather_Events', 'Adaptation_Strategies']

# Ensure categorical columns are strings
for col in categorical_features:
    X[col] = X[col].astype(str)

# Preprocessing Pipeline
# Using handle_unknown='ignore' to be safe during deployment
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Models
models = {
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression()
}

best_pipe = None
best_r2 = -float('inf')

print("--- Training Models ---")
for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"{name} - R2: {r2:.4f}, MAE: {mae:.2f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_pipe = pipe

# 4. Save best model
joblib.dump(best_pipe, 'best_carbon_model.pkl')
print(f"\nBest Model saved as 'best_carbon_model.pkl'")

# 5. Feature Importance Visualization
try:
    if hasattr(best_pipe.named_steps['regressor'], 'feature_importances_'):
        importances = best_pipe.named_steps['regressor'].feature_importances_
        ohe = best_pipe.named_steps['preprocessor'].named_transformers_['cat']
        ohe_features = ohe.get_feature_names_out(categorical_features)
        all_features = numeric_features + list(ohe_features)
        
        feat_df = pd.DataFrame({'Feature': all_features, 'Importance': importances}).sort_values('Importance', ascending=False)
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feat_df.head(15))
        plt.title('Top 15 Feature Importances')
        plt.savefig('feature_importance.png')
        print("Feature importance plot saved.")
except Exception as e:
    print(f"Could not generate feature importance: {e}")

# Helper Prediction Function
def predict_impact(input_dict):
    df_input = pd.DataFrame([input_dict])
    # Ensure types match training
    for col in categorical_features:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str)
            
    prediction = best_pipe.predict(df_input)[0]
    
    if prediction < 10000: impact = "Low"
    elif prediction < 20000: impact = "Medium"
    else: impact = "High"
    
    return prediction, impact

if __name__ == "__main__":
    # Test with first row
    test_sample = X.iloc[0].to_dict()
    val, imp = predict_impact(test_sample)
    print(f"\nTest Prediction: {val:.2f} kg, Impact: {imp}")

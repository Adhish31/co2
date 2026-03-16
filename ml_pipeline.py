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
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load Dataset
df = pd.read_csv('agricultural_emissions.csv')

# 2. EDA (Exploratory Data Analysis)
print("--- Summary Statistics ---")
print(df.describe())

plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig('correlation_heatmap.png')

plt.figure(figsize=(10, 6))
sns.histplot(df['Carbon_Emission_Score_kg'], kde=True)
plt.title("Distribution of Carbon Emissions")
plt.savefig('emission_distribution.png')

# 3. Data Preprocessing & Model Training Phase
X = df.drop('Carbon_Emission_Score_kg', axis=1)
y = df['Carbon_Emission_Score_kg']

numeric_features = [
    'Farm_Size_acres', 'Fertilizer_Usage_kg', 'Pesticide_Usage_liters', 
    'Fuel_Consumption_liters', 'Electricity_Consumption_kWh', 
    'Livestock_Count', 'Transportation_Distance_km'
]
categorical_features = ['Crop_Type', 'Soil_Type', 'Irrigation_Type']

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'SVR': SVR(kernel='rbf'),
    'Neural Network (MLP)': MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42)
}

results = []

def evaluate_model(name, model, X_t, y_t, X_v, y_v):
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    pipe.fit(X_t, y_t)
    y_pred = pipe.predict(X_v)
    
    mae = mean_absolute_error(y_v, y_pred)
    mse = mean_squared_error(y_v, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_v, y_pred)
    
    return pipe, {'Model': name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}, y_pred

best_model_name = ""
best_pipe = None
best_r2 = -float('inf')

for name, model in models.items():
    pipe, metrics, y_pred = evaluate_model(name, model, X_train, y_train, X_test, y_test)
    results.append(metrics)
    print(f"--- {name} Results ---")
    print(metrics)
    
    if metrics['R2'] > best_r2:
        best_r2 = metrics['R2']
        best_model_name = name
        best_pipe = pipe

# 5. Visualization of Results
results_df = pd.DataFrame(results)
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R2', data=results_df)
plt.title('Model R2 Comparison')
plt.savefig('model_comparison.png')

# 6. Feature Importance (for best model if applicable)
if hasattr(best_pipe.named_steps['regressor'], 'feature_importances_'):
    importances = best_pipe.named_steps['regressor'].feature_importances_
    # Construct feature names from OneHotEncoder
    ohe_features = best_pipe.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_features = numeric_features + list(ohe_features)
    
    feat_df = pd.DataFrame({'Feature': all_features, 'Importance': importances}).sort_values('Importance', ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_df.head(10))
    plt.title('Top 10 Feature Importances')
    plt.savefig('feature_importance.png')

# 8. Save the Best Model
joblib.dump(best_pipe, 'best_carbon_model.pkl')
print(f"\nBest Model: {best_model_name} with R2: {best_r2:.4f}")
print("Model saved as 'best_carbon_model.pkl'")

# 10. Helper Prediction Function for deployment usage
def predict_impact(input_data):
    # input_data is a dict
    df_input = pd.DataFrame([input_data])
    prediction = best_pipe.predict(df_input)[0]
    
    # Classification logic
    if prediction < 50000:
        impact = "Low"
    elif prediction < 200000:
        impact = "Medium"
    else:
        impact = "High"
    
    return prediction, impact

# Test prediction
test_input = {
    'Farm_Size_acres': 5,
    'Crop_Type': 'Rice',
    'Soil_Type': 'Clay',
    'Fertilizer_Usage_kg': 100,
    'Pesticide_Usage_liters': 8,
    'Fuel_Consumption_liters': 30,
    'Irrigation_Type': 'Flood',
    'Electricity_Consumption_kWh': 200,
    'Livestock_Count': 10,
    'Transportation_Distance_km': 50
}
pred, imp = predict_impact(test_input)
print(f"\nExample Prediction: {pred:.2f} kg CO2e, Impact: {imp}")

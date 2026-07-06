import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# === Load dataset ===
df = pd.read_csv("merged_lora_weather_final.csv")

# === Top 10 weather variables (from your prior analysis) ===
top_10_features = [
    'Humidity', 'RainRate', 'Temperature', 'RadarReflectivity',
    'Pressure', 'WindSpeed', 'RainAccumulated', 'KineticEnergy',
    'MORVisibility', 'ParticlesDetected'
]
target = 'RSSI'

# === Clean and prepare data ===
df = df[top_10_features + [target]].apply(pd.to_numeric, errors='coerce').dropna()
X = df[top_10_features]
y = df[target]

# === Split into training and testing sets ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# === Train the Linear Regression model ===
model = LinearRegression()
model.fit(X_train, y_train)

# === Predict and evaluate ===
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# === Output results ===
print(f"\n📈 Linear Regression Results (predicting RSSI):")
print(f"R² score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

# === Save model coefficients ===
coefficients = pd.Series(model.coef_, index=top_10_features).sort_values(ascending=False)
output_df = pd.DataFrame({
    "Feature": coefficients.index,
    "Coefficient": coefficients.values,
    "Abs Coefficient": coefficients.abs().values
})
output_df.sort_values(by="Abs Coefficient", ascending=False, inplace=True)
output_df.to_csv("rssi_linear_regression_coefficients.csv", index=False)

print("\n✅ Coefficients saved to 'rssi_linear_regression_coefficients.csv'")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# === Load and prepare dataset ===
df = pd.read_csv("merged_lora_weather_final.csv")

# Top 10 weather features
features = [
    'Humidity', 'RainRate', 'Temperature', 'RadarReflectivity',
    'Pressure', 'WindSpeed', 'RainAccumulated', 'KineticEnergy',
    'MORVisibility', 'ParticlesDetected'
]
target = 'RSSI'

# Clean and subset
df = df[features + [target]].apply(pd.to_numeric, errors='coerce').dropna()
X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Polynomial features ===
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# === Fit regression model ===
model = LinearRegression()
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)

# === Evaluate model ===
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"\n📈 Polynomial Regression Results (degree=2):")
print(f"R² score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

# === Plot predicted vs actual ===
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual RSSI")
plt.ylabel("Predicted RSSI")
plt.title("Polynomial Regression: Actual vs Predicted RSSI")
plt.grid(True)
plt.tight_layout()
plt.savefig("poly_regression_rssi_plot.png")
plt.show()

print("✅ Plot saved as 'poly_regression_rssi_plot.png'")

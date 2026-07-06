import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# === Load dataset ===
df = pd.read_csv("merged_lora_weather_final.csv")

# === Define top 10 features and target ===
features = [
    'Humidity', 'RainRate', 'Temperature', 'RadarReflectivity',
    'Pressure', 'WindSpeed', 'RainAccumulated', 'KineticEnergy',
    'MORVisibility', 'ParticlesDetected'
]
target = 'RSSI'

# === Clean and prepare data ===
df = df[features + [target]].apply(pd.to_numeric, errors='coerce').dropna()
X = df[features]
y = df[target]

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Random Forest Regressor ===
rf = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# === Evaluate model ===
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\n🌳 Random Forest Regression Results:")
print(f"R² score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

# === Plot actual vs predicted ===
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual RSSI")
plt.ylabel("Predicted RSSI")
plt.title("Random Forest Regression: Actual vs Predicted RSSI")
plt.grid(True)
plt.tight_layout()
plt.savefig("rf_regression_rssi_plot.png")
plt.show()

# === Feature Importance Plot ===
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
sorted_features = [features[i] for i in sorted_idx]

plt.figure(figsize=(10, 6))
plt.bar(range(len(features)), importances[sorted_idx], tick_label=sorted_features)
plt.xticks(rotation=45, ha='right')
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig("rf_feature_importance.png")
plt.show()

print("✅ Plots saved as 'rf_regression_rssi_plot.png' and 'rf_feature_importance.png'")

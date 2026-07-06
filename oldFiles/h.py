import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# === Feature scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Try different k values ===
k_values = [3, 5, 7, 9]
results = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    results.append((k, r2, mse))

    # Plotting predictions
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.title(f"k-NN Regression (k={k}) — RSSI Prediction")
    plt.xlabel("Actual RSSI")
    plt.ylabel("Predicted RSSI")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"knn_rssi_k{k}.png")
    plt.close()

# === Print Results ===
print("\n📊 k-NN Regression Results:")
for k, r2, mse in results:
    print(f"k = {k:<2} -> R²: {r2:+.4f}, MSE: {mse:.4f}")

print("\n✅ Plots saved as: knn_rssi_k3.png, knn_rssi_k5.png, ...")

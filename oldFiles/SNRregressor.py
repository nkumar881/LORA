import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# === Load merged dataset ===
df = pd.read_csv("merged_lora_weather_final.csv")

# === Define target and features ===
target = "SNR (LSNR)"
weather_features = [
    'Humidity', 'KineticEnergy', 'MORVisibility', 'ParticlesDetected', 'Pressure',
    'RadarReflectivity', 'RainAbsolute', 'RainAccumulated', 'RainIntensity',
    'SnowDepthIntensity', 'Temperature', 'WindDirection', 'WindSpeed', 'WeatherCode4680'
]

# === Drop missing values ===
df = df[[target] + weather_features].dropna()

X = df[weather_features]
y = df[target]

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Standardize for Lasso, Ridge, KNN ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}
feature_importances = {}

# === Linear Regression ===
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
results["LinearRegression"] = {
    "R2": r2_score(y_test, y_pred),
    "RMSE": mean_squared_error(y_test, y_pred) ** 0.5
}
feature_importances["LinearRegression"] = dict(zip(X.columns, lr.coef_))

# === Polynomial Regression (deg=2) ===
poly = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly.fit(X_train, y_train)
y_pred = poly.predict(X_test)
results["PolynomialRegression (deg=2)"] = {
    "R2": r2_score(y_test, y_pred),
    "RMSE": mean_squared_error(y_test, y_pred) ** 0.5
}
feature_importances["PolynomialRegression (deg=2)"] = "Feature importance not directly interpretable due to polynomial expansion."

# === Random Forest ===
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
results["RandomForest"] = {
    "R2": r2_score(y_test, y_pred),
    "RMSE": mean_squared_error(y_test, y_pred) ** 0.5
}
feature_importances["RandomForest"] = dict(zip(X.columns, rf.feature_importances_))

# === Lasso Regression ===
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
y_pred = lasso.predict(X_test_scaled)
results["Lasso"] = {
    "R2": r2_score(y_test, y_pred),
    "RMSE": mean_squared_error(y_test, y_pred) ** 0.5
}
feature_importances["Lasso"] = dict(zip(X.columns, lasso.coef_))

# === Ridge Regression ===
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred = ridge.predict(X_test_scaled)
results["Ridge"] = {
    "R2": r2_score(y_test, y_pred),
    "RMSE": mean_squared_error(y_test, y_pred) ** 0.5
}
feature_importances["Ridge"] = dict(zip(X.columns, ridge.coef_))

# === K-Nearest Neighbors ===
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
results["KNN"] = {
    "R2": r2_score(y_test, y_pred),
    "RMSE": mean_squared_error(y_test, y_pred) ** 0.5
}
feature_importances["KNN"] = "KNN is a non-parametric model and does not provide feature importances."

# === Output Results ===
print("Regression Results on SNR (LSNR):\n")
for model, metrics in results.items():
    print(f"{model}:")
    print(f"  R²: {metrics['R2']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print()

# === Plot feature importances (where available) ===
for model, importance in feature_importances.items():
    if isinstance(importance, dict):
        sorted_items = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        labels, values = zip(*sorted_items)

        plt.figure(figsize=(10, 5))
        plt.bar(labels, values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Importance / Coefficient")
        plt.title(f"Feature Importance: {model}")
        plt.tight_layout()
        plt.savefig(f"{model}_feature_importance.png")
        plt.close()
        print(f"Feature importance plot saved: {model}_feature_importance.png")
    else:
        print(f"{model}: {importance}")

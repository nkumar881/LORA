import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# === Load dataset ===
df = pd.read_csv("merged_lora_weather_final.csv")

# === Select features ===
target = "Throughput (bps)"
weather_features = [
    'Humidity', 'KineticEnergy', 'MORVisibility', 'ParticlesDetected', 'Pressure',
    'RadarReflectivity', 'RainAbsolute', 'RainAccumulated', 'RainIntensity',
    'SnowDepthIntensity', 'Temperature', 'WindDirection', 'WindSpeed'
]
df_clean = df[[target] + weather_features].dropna()
X = df_clean[weather_features]
y = df_clean[target]

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Scale for models that need it ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

# === 1. Linear Regression ===
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
results["LinearRegression"] = {
    "R2": r2_score(y_test, y_pred),
    "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
    "coeffs": dict(zip(X.columns, lr.coef_))
}

# === 2. Polynomial Regression (degree=2) ===
poly = make_pipeline(PolynomialFeatures(2), LinearRegression())
poly.fit(X_train, y_train)
y_pred = poly.predict(X_test)
results["PolynomialRegression (deg=2)"] = {
    "R2": r2_score(y_test, y_pred),
    "RMSE": mean_squared_error(y_test, y_pred) ** 0.5
}

# === 3. Random Forest ===
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
results["RandomForest"] = {
    "R2": r2_score(y_test, y_pred),
    "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
    "importances": dict(zip(X.columns, rf.feature_importances_))
}

# === 4. Lasso Regression ===
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
y_pred = lasso.predict(X_test_scaled)
results["Lasso"] = {
    "R2": r2_score(y_test, y_pred),
    "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
    "coeffs": dict(zip(X.columns, lasso.coef_))
}

# === 5. Ridge Regression ===
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred = ridge.predict(X_test_scaled)
results["Ridge"] = {
    "R2": r2_score(y_test, y_pred),
    "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
    "coeffs": dict(zip(X.columns, ridge.coef_))
}

# === 6. KNN Regressor ===
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
results["KNN"] = {
    "R2": r2_score(y_test, y_pred),
    "RMSE": mean_squared_error(y_test, y_pred) ** 0.5
}

# === Print Results ===
print("Regression Results on Throughput (bps):\n")
for model, metrics in results.items():
    print(f"{model}:")
    print(f"  R²: {metrics['R2']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}\n")

# === Plot feature importance ===
for model, metrics in results.items():
    if "coeffs" in metrics:
        plt.figure()
        plt.barh(*zip(*metrics["coeffs"].items()))
        plt.title(f"{model} Feature Coefficients")
        plt.xlabel("Coefficient")
        plt.tight_layout()
        plt.savefig(f"throughput_feature_importance_{model}.png")

    elif "importances" in metrics:
        plt.figure()
        plt.barh(*zip(*metrics["importances"].items()))
        plt.title(f"{model} Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(f"throughput_feature_importance_{model}.png")

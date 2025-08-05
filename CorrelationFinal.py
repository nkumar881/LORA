
import pandas as pd
from itertools import combinations

# Load the merged dataset
df = pd.read_csv("merged_lora_weather_final.csv")

# Define weather features
weather_vars = [
    'Humidity', 'KineticEnergy', 'MORVisibility', 'ParticlesDetected', 'Pressure',
    'RadarReflectivity', 'RainAbsolute', 'RainAccumulated', 'RainIntensity', 'RainRate',
    'SnowDepthIntensity', 'Temperature', 'WindDirection', 'WindSpeed',
    'AverageParticleSpeed', 'VolumeEquivalentDiameter', 'WeatherCode4680'
]

# Define LoRa metrics to analyze
lora_targets = ['RSSI', 'SNR (LSNR)', 'Throughput (bps)', 'Latency (s)']

# Convert all relevant columns to numeric
for col in weather_vars + lora_targets:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Iterate through each target metric
for target_metric in lora_targets:
    print(f"\n===== Analyzing correlation for: {target_metric} =====")

    # Step 1: Correlation of individual weather parameters with target metric
    correlations = []
    for col in weather_vars:
        corr = df[[col, target_metric]].corr().iloc[0, 1]
        correlations.append((col, corr))

    # Step 2: Filter based on |correlation| >= 0.05
    filtered = [(col, corr) for col, corr in correlations if abs(corr) >= 0.05]
    filtered_features = [col for col, _ in filtered]

    # Step 3: Display individual correlations
    print(f"\nCorrelation of weather features with {target_metric}:\n")
    for col, corr in sorted(filtered, key=lambda x: abs(x[1]), reverse=True):
        print(f"{col:25} -> {corr:+.4f}")

    # Step 4: Pairwise average and correlation with target metric
    print(f"\nPairwise averaged correlations with {target_metric} (R² >= 0.007):\n")
    pairwise_valid_features = set()
    combo_correlations_2 = []
    for col1, col2 in combinations(filtered_features, 2):
        combo_avg = df[[col1, col2]].mean(axis=1)
        r = combo_avg.corr(df[target_metric])
        r_squared = r**2 if pd.notnull(r) else float('nan')
        if r_squared >= 0.007:
            combo_correlations_2.append(((col1, col2), r, r_squared))
            pairwise_valid_features.update([col1, col2])

    combo_correlations_2.sort(key=lambda x: x[2], reverse=True)
    for (col1, col2), r, r_squared in combo_correlations_2:
        print(f"{col1:15} + {col2:15} -> r: {r:+.4f}, R²: {r_squared:.4f}")

    # Step 5: 3-way combinations from valid pairwise features
    print(f"\n3-way averaged correlations with {target_metric} (from filtered pairwise set):\n")
    combo_correlations_3 = []
    valid_features_list = list(pairwise_valid_features)
    for col1, col2, col3 in combinations(valid_features_list, 3):
        combo_avg = df[[col1, col2, col3]].mean(axis=1)
        r = combo_avg.corr(df[target_metric])
        r_squared = r**2 if pd.notnull(r) else float('nan')
        combo_correlations_3.append(((col1, col2, col3), r, r_squared))

    combo_correlations_3.sort(key=lambda x: x[2], reverse=True)
    for (col1, col2, col3), r, r_squared in combo_correlations_3:
        print(f"{col1:12} + {col2:12} + {col3:12} -> r: {r:+.4f}, R²: {r_squared:.4f}")

    # Save top 3-way combinations with R² >= 0.007
    output_rows = []
    for (col1, col2, col3), r, r_squared in combo_correlations_3:
        if r_squared >= 0.007:
            output_rows.append({
                'Variable 1': col1,
                'Variable 2': col2,
                'Variable 3': col3,
                'Correlation (r)': r,
                'R²': r_squared
            })

    if output_rows:
        output_df = pd.DataFrame(output_rows)
        output_filename = f"top_{target_metric.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '')}_trios.csv"
        output_df.to_csv(output_filename, index=False)
        print(f"\nSaved {len(output_rows)} 3-feature combinations to '{output_filename}'")
    else:
        print("\nNo 3-feature combinations had R² ≥ 0.007")

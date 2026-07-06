import pandas as pd
from itertools import combinations

# === Load the merged dataset ===
df = pd.read_csv("merged_lora_weather_final.csv")

# === Define weather and target (RSSI) ===
weather_vars = [
    'Humidity', 'KineticEnergy', 'MORVisibility', 'ParticlesDetected', 'Pressure',
    'RadarReflectivity', 'RainAbsolute', 'RainAccumulated', 'RainIntensity', 'RainRate',
    'SnowDepthIntensity', 'Temperature', 'WindDirection', 'WindSpeed',
    'AverageParticleSpeed', 'VolumeEquivalentDiameter', 'WeatherCode4680'
]

target_metric = 'RSSI'

# === Convert columns to numeric ===
df[target_metric] = pd.to_numeric(df[target_metric], errors='coerce')
for col in weather_vars:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# === Step 1: Correlation of individual weather parameters with RSSI ===
correlations = []
for col in weather_vars:
    corr = df[[col, target_metric]].corr().iloc[0, 1]
    correlations.append((col, corr))

# === Step 2: Filter based on |correlation| >= 0.05 ===
#filtered = [(col, corr) for col, corr in correlations if abs(corr) >= 0.05]
filtered = correlations  # Show all regardless of strength
filtered_features = [col for col, _ in filtered]

# === Step 2: No filtering — include all correlations ===
#filtered = correlations  # Show all regardless of strength


# === Step 3: Display individual correlations ===
print(f"\nCorrelation of weather features with {target_metric}:\n")
for col, corr in sorted(filtered, key=lambda x: abs(x[1]), reverse=True):
    print(f"{col:25} -> {corr:+.4f}")

# === Step 4: Pairwise average and correlation with RSSI ===
print(f"\nPairwise averaged correlations with {target_metric} (R² >= 0.009):\n")

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

# === Step 5: 3-feature combinations from valid pairwise features ===
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


# === Step 6: Save 3-way combinations with R² >= 0.005 === changed to 0.007
output_rows = []
for (col1, col2, col3), r, r_squared in combo_correlations_3:
    if r_squared >= 0.01:
        output_rows.append({
            'Variable 1': col1,
            'Variable 2': col2,
            'Variable 3': col3,
            'Correlation (r)': r,
            'R²': r_squared
        })

# Save to CSV if any combinations qualify
if output_rows:
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv("top_rssi_trios.csv", index=False)
    print(f"\n✅ Saved {len(output_rows)} 3-feature combinations (R² ≥ 0.005) to 'top_rssi_trios.csv'")
else:
    print("\n⚠️ No 3-feature combinations had R² ≥ 0.005")
from itertools import combinations

# === Step 7: 4-way combinations from top trio variables ===
print(f"\n🔍 Generating 4-variable combinations from top trios...\n")

# Get all unique variables from top trios
top_trio_vars = set()
for row in output_rows:
    top_trio_vars.update([row['Variable 1'], row['Variable 2'], row['Variable 3']])

top_trio_vars = sorted(top_trio_vars)
print(f"Found {len(top_trio_vars)} unique variables in top trios.")

quad_results = []
for col1, col2, col3, col4 in combinations(top_trio_vars, 4):
    quad_avg = df[[col1, col2, col3, col4]].mean(axis=1)
    r = quad_avg.corr(df[target_metric])
    r_squared = r**2 if pd.notnull(r) else float('nan')
    if r_squared >= 0.01:
        quad_results.append({
            'Variable 1': col1,
            'Variable 2': col2,
            'Variable 3': col3,
            'Variable 4': col4,
            'Correlation (r)': r,
            'R²': r_squared
        })

# Sort and display
quad_results.sort(key=lambda x: x['R²'], reverse=True)

print(f"\n4-way averaged correlations with {target_metric} (R² ≥ 0.005):\n")
for row in quad_results:
    print(f"{row['Variable 1']:12} + {row['Variable 2']:12} + {row['Variable 3']:12} + {row['Variable 4']:12} -> r: {row['Correlation (r)']:+.4f}, R²: {row['R²']:.4f}")

# Save to CSV
if quad_results:
    pd.DataFrame(quad_results).to_csv("top_rssi_quads.csv", index=False)
    print(f"\n✅ Saved {len(quad_results)} 4-feature combinations to 'top_rssi_quads.csv'")
else:
    print("\n⚠️ No 4-variable combinations had R² ≥ 0.005")
# === Step 8: 5-way combinations from top trio variables ===
print(f"\n🔍 Generating 5-variable combinations from top trios...\n")

quint_results = []
for cols in combinations(top_trio_vars, 5):
    quint_avg = df[list(cols)].mean(axis=1)
    r = quint_avg.corr(df[target_metric])
    r_squared = r**2 if pd.notnull(r) else float('nan')
    if r_squared >= 0.013:
        quint_results.append({
            'Variable 1': cols[0],
            'Variable 2': cols[1],
            'Variable 3': cols[2],
            'Variable 4': cols[3],
            'Variable 5': cols[4],
            'Correlation (r)': r,
            'R²': r_squared
        })

# Sort and display
quint_results.sort(key=lambda x: x['R²'], reverse=True)

print(f"\n5-way averaged correlations with {target_metric} (R² ≥ 0.005):\n")
for row in quint_results:
    print(f"{row['Variable 1']:12} + {row['Variable 2']:12} + {row['Variable 3']:12} + {row['Variable 4']:12} + {row['Variable 5']:12} -> r: {row['Correlation (r)']:+.4f}, R²: {row['R²']:.4f}")

# Save to CSV
if quint_results:
    pd.DataFrame(quint_results).to_csv("top_rssi_quints.csv", index=False)
    print(f"\n✅ Saved {len(quint_results)} 5-feature combinations to 'top_rssi_quints.csv'")
else:
    print("\n⚠️ No 5-variable combinations had R² ≥ 0.005")
# === Step 9: 6-way combinations from top quint variables ===
print(f"\n🔍 Generating 6-variable combinations from top 5-variable combos...\n")

# Extract all unique variables used in the top 5-combo results
top_quint_vars = set()
for row in quint_results:
    top_quint_vars.update([
        row['Variable 1'], row['Variable 2'], row['Variable 3'],
        row['Variable 4'], row['Variable 5']
    ])
top_quint_vars = sorted(top_quint_vars)
print(f"Found {len(top_quint_vars)} unique variables in top 5-feature combinations.")

# Compute 6-way averages and correlations
sixth_results = []
for cols in combinations(top_quint_vars, 6):
    sixth_avg = df[list(cols)].mean(axis=1)
    r = sixth_avg.corr(df[target_metric])
    r_squared = r**2 if pd.notnull(r) else float('nan')
    if r_squared >= 0.01:
        sixth_results.append({
            'Variable 1': cols[0],
            'Variable 2': cols[1],
            'Variable 3': cols[2],
            'Variable 4': cols[3],
            'Variable 5': cols[4],
            'Variable 6': cols[5],
            'Correlation (r)': r,
            'R²': r_squared
        })

# Sort and display
sixth_results.sort(key=lambda x: x['R²'], reverse=True)

print(f"\n6-way averaged correlations with {target_metric} (R² ≥ 0.005):\n")
for row in sixth_results:
    print(f"{row['Variable 1']:12} + {row['Variable 2']:12} + {row['Variable 3']:12} + {row['Variable 4']:12} + {row['Variable 5']:12} + {row['Variable 6']:12} -> r: {row['Correlation (r)']:+.4f}, R²: {row['R²']:.4f}")

# Save to CSV
if sixth_results:
    pd.DataFrame(sixth_results).to_csv("top_rssi_sixths.csv", index=False)
    print(f"\n✅ Saved {len(sixth_results)} 6-feature combinations to 'top_rssi_sixths.csv'")
else:
    print("\n⚠️ No 6-variable combinations had R² ≥ 0.005")




# === Step 10: 7-way combinations from top sixth variables ===
print(f"\n🔍 Generating 7-variable combinations from top 6-variable combos...\n")

# Extract all unique variables used in the top 6-combo results
top_sixth_vars = set()
for row in sixth_results:
    top_sixth_vars.update([
        row['Variable 1'], row['Variable 2'], row['Variable 3'],
        row['Variable 4'], row['Variable 5'], row['Variable 6']
    ])
top_sixth_vars = sorted(top_sixth_vars)
print(f"Found {len(top_sixth_vars)} unique variables in top 6-feature combinations.")

# Compute 7-way averages and correlations
seventh_results = []
for cols in combinations(top_sixth_vars, 7):
    seventh_avg = df[list(cols)].mean(axis=1)
    r = seventh_avg.corr(df[target_metric])
    r_squared = r**2 if pd.notnull(r) else float('nan')
    if r_squared >= 0.01:
        seventh_results.append({
            'Variable 1': cols[0],
            'Variable 2': cols[1],
            'Variable 3': cols[2],
            'Variable 4': cols[3],
            'Variable 5': cols[4],
            'Variable 6': cols[5],
            'Variable 7': cols[6],
            'Correlation (r)': r,
            'R²': r_squared
        })

# Sort and display
seventh_results.sort(key=lambda x: x['R²'], reverse=True)

print(f"\n7-way averaged correlations with {target_metric} (R² ≥ 0.007):\n")
for row in seventh_results:
    print(f"{row['Variable 1']:12} + {row['Variable 2']:12} + {row['Variable 3']:12} + "
          f"{row['Variable 4']:12} + {row['Variable 5']:12} + {row['Variable 6']:12} + "
          f"{row['Variable 7']:12} -> r: {row['Correlation (r)']:+.4f}, R²: {row['R²']:.4f}")

# Save to CSV
if seventh_results:
    pd.DataFrame(seventh_results).to_csv("top_rssi_sevenths.csv", index=False)
    print(f"\n✅ Saved {len(seventh_results)} 7-feature combinations to 'top_rssi_sevenths.csv'")
else:
    print("\n⚠️ No 7-variable combinations had R² ≥ 0.007")
# === Step 11: 8-way combinations from top 7-variable combos ===
print(f"\n🔍 Generating 8-variable combinations from top 7-variable combos...\n")

# Extract all unique variables used in the top 7-combo results
top_seventh_vars = set()
for row in seventh_results:
    top_seventh_vars.update([
        row['Variable 1'], row['Variable 2'], row['Variable 3'],
        row['Variable 4'], row['Variable 5'], row['Variable 6'],
        row['Variable 7']
    ])
top_seventh_vars = sorted(top_seventh_vars)
print(f"Found {len(top_seventh_vars)} unique variables in top 7-feature combinations.")

# Compute 8-way averages and correlations
eighth_results = []
for cols in combinations(top_seventh_vars, 8):
    eighth_avg = df[list(cols)].mean(axis=1)
    r = eighth_avg.corr(df[target_metric])
    r_squared = r**2 if pd.notnull(r) else float('nan')
    if r_squared >= 0.01:
        eighth_results.append({
            'Variable 1': cols[0],
            'Variable 2': cols[1],
            'Variable 3': cols[2],
            'Variable 4': cols[3],
            'Variable 5': cols[4],
            'Variable 6': cols[5],
            'Variable 7': cols[6],
            'Variable 8': cols[7],
            'Correlation (r)': r,
            'R²': r_squared
        })

# Sort and display
eighth_results.sort(key=lambda x: x['R²'], reverse=True)

print(f"\n8-way averaged correlations with {target_metric} (R² ≥ 0.01):\n")
for row in eighth_results:
    print(f"{row['Variable 1']:12} + {row['Variable 2']:12} + {row['Variable 3']:12} + "
          f"{row['Variable 4']:12} + {row['Variable 5']:12} + {row['Variable 6']:12} + "
          f"{row['Variable 7']:12} + {row['Variable 8']:12} -> r: {row['Correlation (r)']:+.4f}, R²: {row['R²']:.4f}")

# Save to CSV
if eighth_results:
    pd.DataFrame(eighth_results).to_csv("top_rssi_eighths.csv", index=False)
    print(f"\n✅ Saved {len(eighth_results)} 8-feature combinations to 'top_rssi_eighths.csv'")
else:
    print("\n⚠️ No 8-variable combinations had R² ≥ 0.01")


# === Step 12: 9-way combinations from top 8-variable combos ===
print(f"\n🔍 Generating 9-variable combinations from top 8-variable combos...\n")

# Extract all unique variables used in the top 8-combo results
top_eighth_vars = set()
for row in eighth_results:
    top_eighth_vars.update([
        row['Variable 1'], row['Variable 2'], row['Variable 3'],
        row['Variable 4'], row['Variable 5'], row['Variable 6'],
        row['Variable 7'], row['Variable 8']
    ])
top_eighth_vars = sorted(top_eighth_vars)
print(f"Found {len(top_eighth_vars)} unique variables in top 8-feature combinations.")

# Compute 9-way averages and correlations
ninth_results = []
for cols in combinations(top_eighth_vars, 9):
    ninth_avg = df[list(cols)].mean(axis=1)
    r = ninth_avg.corr(df[target_metric])
    r_squared = r**2 if pd.notnull(r) else float('nan')
    if r_squared >= 0.01:
        ninth_results.append({
            'Variable 1': cols[0],
            'Variable 2': cols[1],
            'Variable 3': cols[2],
            'Variable 4': cols[3],
            'Variable 5': cols[4],
            'Variable 6': cols[5],
            'Variable 7': cols[6],
            'Variable 8': cols[7],
            'Variable 9': cols[8],
            'Correlation (r)': r,
            'R²': r_squared
        })

# Sort and display
ninth_results.sort(key=lambda x: x['R²'], reverse=True)

print(f"\n9-way averaged correlations with {target_metric} (R² ≥ 0.01):\n")
for row in ninth_results:
    print(f"{row['Variable 1']:12} + {row['Variable 2']:12} + {row['Variable 3']:12} + "
          f"{row['Variable 4']:12} + {row['Variable 5']:12} + {row['Variable 6']:12} + "
          f"{row['Variable 7']:12} + {row['Variable 8']:12} + {row['Variable 9']:12} -> r: {row['Correlation (r)']:+.4f}, R²: {row['R²']:.4f}")

# Save to CSV
if ninth_results:
    pd.DataFrame(ninth_results).to_csv("top_rssi_ninths.csv", index=False)
    print(f"\n✅ Saved {len(ninth_results)} 9-feature combinations to 'top_rssi_ninths.csv'")
else:
    print("\n⚠️ No 9-variable combinations had R² ≥ 0.01")



# === Step 13: 10-way combinations from top 9-variable combos ===
print(f"\n🔍 Generating 10-variable combinations from top 9-variable combos...\n")

# Extract all unique variables used in the top 9-combo results
top_ninth_vars = set()
for row in ninth_results:
    top_ninth_vars.update([
        row['Variable 1'], row['Variable 2'], row['Variable 3'],
        row['Variable 4'], row['Variable 5'], row['Variable 6'],
        row['Variable 7'], row['Variable 8'], row['Variable 9']
    ])
top_ninth_vars = sorted(top_ninth_vars)
print(f"Found {len(top_ninth_vars)} unique variables in top 9-feature combinations.")

# Compute 10-way averages and correlations
tenth_results = []
for cols in combinations(top_ninth_vars, 10):
    tenth_avg = df[list(cols)].mean(axis=1)
    r = tenth_avg.corr(df[target_metric])
    r_squared = r**2 if pd.notnull(r) else float('nan')
    if r_squared >= 0.01:
        tenth_results.append({
            'Variable 1': cols[0],
            'Variable 2': cols[1],
            'Variable 3': cols[2],
            'Variable 4': cols[3],
            'Variable 5': cols[4],
            'Variable 6': cols[5],
            'Variable 7': cols[6],
            'Variable 8': cols[7],
            'Variable 9': cols[8],
            'Variable 10': cols[9],
            'Correlation (r)': r,
            'R²': r_squared
        })

# Sort and display
tenth_results.sort(key=lambda x: x['R²'], reverse=True)

print(f"\n10-way averaged correlations with {target_metric} (R² ≥ 0.01):\n")
for row in tenth_results:
    print(f"{row['Variable 1']:12} + {row['Variable 2']:12} + {row['Variable 3']:12} + "
          f"{row['Variable 4']:12} + {row['Variable 5']:12} + {row['Variable 6']:12} + "
          f"{row['Variable 7']:12} + {row['Variable 8']:12} + {row['Variable 9']:12} + "
          f"{row['Variable 10']:12} -> r: {row['Correlation (r)']:+.4f}, R²: {row['R²']:.4f}")

# Save to CSV
if tenth_results:
    pd.DataFrame(tenth_results).to_csv("top_rssi_tenths.csv", index=False)
    print(f"\n✅ Saved {len(tenth_results)} 10-feature combinations to 'top_rssi_tenths.csv'")
else:
    print("\n⚠️ No 10-variable combinations had R² ≥ 0.01")

import pandas as pd
import matplotlib.pyplot as plt

# Load the weather CSV
file_path = "WilsonHall_snow(in).csv"  # Update if needed
df = pd.read_csv(file_path)

# Create a pseudo-timestamp (best effort, assuming DateTime is in mm:ss.s format within a known session)
# We'll just treat it as elapsed seconds for plotting
df['ElapsedSeconds'] = df['DateTime'].str.extract(r'(\d+):(\d+\.\d+)').astype(float).apply(
    lambda x: x[0]*60 + x[1], axis=1)

# Set ElapsedSeconds as the index
df.set_index('ElapsedSeconds', inplace=True)

# Convert numeric columns that might be strings
columns_to_plot = [
    'Temperature', 'Humidity', 'SnowDepthIntensity',
    'RainIntensity', 'WindSpeed', 'MORVisibility'
]

for col in columns_to_plot:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Plotting
fig, axes = plt.subplots(len(columns_to_plot), 1, figsize=(14, 14), sharex=True)

for i, col in enumerate(columns_to_plot):
    df[col].plot(ax=axes[i], title=f'{col} over Time', linewidth=1.5)
    axes[i].set_ylabel(col)

axes[-1].set_xlabel('Elapsed Time (seconds)')
plt.tight_layout()
plt.savefig("weather_metrics_plot.png")  # Saves figure for presentations
plt.show()

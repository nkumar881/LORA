import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "lora_log_snow(in).csv"  # Update this path as needed
df = pd.read_csv(file_path)

# Extract full timestamp from 'Decoded Data'
df['FullTimestamp'] = df['Decoded Data'].str.extract(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)')
df['FullTimestamp'] = pd.to_datetime(df['FullTimestamp'])

# Set the full timestamp as the index
df.set_index('FullTimestamp', inplace=True)

# Create subplots for each metric
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Plot RSSI
df['RSSI'].plot(ax=axes[0], title='RSSI over Time', color='blue')
axes[0].set_ylabel('RSSI (dBm)')

# Plot SNR
df['SNR (LSNR)'].plot(ax=axes[1], title='SNR over Time', color='green')
axes[1].set_ylabel('SNR (dB)')

# Plot Throughput
df['Throughput (bps)'].plot(ax=axes[2], title='Throughput over Time', color='purple')
axes[2].set_ylabel('Throughput (bps)')

# Plot Latency
df['Latency (s)'].plot(ax=axes[3], title='Latency over Time', color='red')
axes[3].set_ylabel('Latency (s)')
axes[3].set_xlabel('Time')

# Layout adjustment
plt.tight_layout()
plt.show()

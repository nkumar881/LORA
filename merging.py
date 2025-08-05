import pandas as pd

# === Load the datasets ===
lora_df = pd.read_csv("combined_lora_log_final.csv")
weather_df = pd.read_csv("combined_wilson_log_final.csv")

# === Parse timestamps ===
lora_df['FullTimestamp'] = pd.to_datetime(
    lora_df['Decoded Data'].astype(str).str.split(';').str[1], errors='coerce'
)
weather_df['DateTime'] = pd.to_datetime(weather_df['DateTime'], errors='coerce')

# === Identify column types ===
numeric_cols = weather_df.select_dtypes(include='number').columns.tolist()
non_numeric_cols = [col for col in weather_df.columns if col not in numeric_cols and col not in ['DateTime']]

# === Merge logic (±5 seconds) ===
merged_rows = []

for _, row in lora_df.iterrows():
    timestamp = row.get('FullTimestamp')
    if pd.isnull(timestamp):
        continue

    nearby = weather_df[
        (weather_df['DateTime'] >= timestamp - pd.Timedelta(seconds=5)) &
        (weather_df['DateTime'] <= timestamp + pd.Timedelta(seconds=5))
    ]

    if not nearby.empty:
        merged_row = row.to_dict()

        # Average numeric columns
        for col in numeric_cols:
            merged_row[col] = nearby[col].mean()

        # Mode of non-numeric (categorical) columns
        for col in non_numeric_cols:
            mode_val = nearby[col].mode()
            merged_row[col] = mode_val[0] if not mode_val.empty else None

        merged_rows.append(merged_row)

# === Save result ===
merged_df = pd.DataFrame(merged_rows)
merged_df.to_csv("merged_lora_weather_final.csv", index=False)
print("Merged dataset saved as 'merged_lora_weather_final.csv'")

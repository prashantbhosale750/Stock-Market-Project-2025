import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Load original data
df = pd.read_csv('data/processed/merged_finbert_sentiment.csv', parse_dates=['Date'])

# Sort and reset index
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Columns to scale
features_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume']
scaler = MinMaxScaler()
df_scaled = df.copy()

# Apply scaling
df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Create next_day_close target
df_scaled['next_day_close'] = df_scaled['Close'].shift(-1)

# Drop the last row (NaN target)
df_scaled = df_scaled[:-1]

# Create output directory if not exists
os.makedirs('data/preprocessed', exist_ok=True)

# Save to CSV
df_scaled.to_csv('data/preprocessed/lstm_input.csv', index=False)
print("✅ Saved: data/preprocessed/lstm_input.csv")

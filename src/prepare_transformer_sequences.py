# src/prepare_transformer_sequences.py

import pandas as pd
import numpy as np
import os

def prepare_transformer_data(input_file, output_dir="data/processed", lookback=20):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"📂 Loading data from {input_file}")
    df = pd.read_csv(input_file)

    # Drop non-numeric columns (Date, Headline, etc.)
    df_numeric = df.select_dtypes(include=[np.number])
    print(f"✅ Using numeric features: {list(df_numeric.columns)}")

    data_array = df_numeric.values.astype(np.float32)

    # Create sequences
    X, y = [], []
    for i in range(len(data_array) - lookback):
        X.append(data_array[i:i+lookback])
        y.append(data_array[i+lookback][df_numeric.columns.get_loc("Close")])  # target = next-day Close

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    # Save sequences
    np.save(os.path.join(output_dir, "transformer_X.npy"), X)
    np.save(os.path.join(output_dir, "transformer_y.npy"), y)

    print(f"✅ Sequences saved: {X.shape[0]} samples")
    print(f"→ X shape: {X.shape}, y shape: {y.shape}")
    print(f"📦 Files saved to: {output_dir}/transformer_X.npy and transformer_y.npy")

if __name__ == "__main__":
    input_path = "data/preprocessed/lstm_input.csv"  # or wherever your normalized CSV is
    prepare_transformer_data(input_path)

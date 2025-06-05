# to implement LSTM model we use : merger_finbert_sentiment.csv

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def load_and_prepare_data(filepath, scaler_path='models/minmax_scaler.save', lookback=60):
    # Load dataset
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Select numeric features
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    # Save the scaler for future use
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

    # Create sequences of lookback days
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])  # past lookback days
        y.append(scaled_data[i, features.index('Close')])  # target: next-day Close

    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output: predicted close price
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    # Path to the processed dataset with FinBERT sentiment (optional)
    data_path = "data/processed/merged_finbert_sentiment.csv"

    # Load and prepare data
    X, y, scaler = load_and_prepare_data(data_path)

    # Split into train/test
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Build and train model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_regressor.h5")
    print("\u2705 Model saved to models/lstm_regressor.h5")
    print("\u2705 Scaler saved to models/minmax_scaler.save")

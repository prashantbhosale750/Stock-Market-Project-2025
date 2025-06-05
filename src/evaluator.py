import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import joblib
import os

# Paths
data_path = "data/processed/merged_finbert_sentiment.csv"
model_path = "models/lstm_regressor.h5"
scaler_path = "models/minmax_scaler.save"
metrics_path = "models/metrics.txt"

# Parameters
window_size = 20
features = ['Open', 'High', 'Low', 'Close', 'Volume']

# Load data and scaler
df = pd.read_csv(data_path)
scaler = joblib.load(scaler_path)
model = load_model(model_path)

# Normalize features
scaled_features = scaler.transform(df[features])

# Create sequences for evaluation
X = []
y = []
for i in range(window_size, len(scaled_features)):
    X.append(scaled_features[i - window_size:i])
    y.append(scaled_features[i, features.index('Close')])

X = np.array(X)
y = np.array(y)

# Predict
y_pred_scaled = model.predict(X).flatten()

# Reconstruct for inverse_transform
def reconstruct(arr, col_index, shape):
    dummy = np.zeros(shape)
    dummy[:, col_index] = arr
    return dummy

# Inverse scale predictions and actual values
close_index = features.index('Close')
y_pred_recon = reconstruct(y_pred_scaled, close_index, (len(y_pred_scaled), len(features)))
y_true_recon = reconstruct(y, close_index, (len(y), len(features)))

y_pred_actual = scaler.inverse_transform(y_pred_recon)[:, close_index]
y_true_actual = scaler.inverse_transform(y_true_recon)[:, close_index]

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_true_actual, y_pred_actual))
mae = mean_absolute_error(y_true_actual, y_pred_actual)
r2 = r2_score(y_true_actual, y_pred_actual)

# Save metrics
os.makedirs("models", exist_ok=True)
with open(metrics_path, "w") as f:
    f.write(f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2 Score: {r2:.4f}\n")

print(f"✅ Evaluation metrics saved to {metrics_path}")

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(y_true_actual, label="Actual Close", alpha=0.7)
plt.plot(y_pred_actual, label="Predicted Close", alpha=0.7)
plt.title("LSTM Prediction vs Actual Closing Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

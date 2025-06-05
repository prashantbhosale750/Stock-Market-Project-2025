import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the final encoded dataset
df = pd.read_csv("data/processed/merged_finbert_encoded.csv")

# Features and Target
features = ['Open', 'High', 'Low', 'Volume', 'sentiment_label']
target = 'Close'
X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics
print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ MAE : {mae:.4f}")
print(f"✅ R²  : {r2:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/random_forest_regressor.pkl")

# Save metrics
os.makedirs("metrics", exist_ok=True)
with open("metrics/random_forest_metrics.txt", "w") as f:
    f.write(f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2: {r2:.4f}\n")

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual", linewidth=2)
plt.plot(y_pred, label="Predicted", linewidth=2)
plt.title("Random Forest: Actual vs Predicted Close Prices")
plt.xlabel("Index")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/random_forest_plot.png")
plt.show()

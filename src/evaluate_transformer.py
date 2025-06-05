from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load model without compiling
model = load_model('models/transformer_regressor.h5', compile=False)

# Load test data
X_test = np.load('data/processed/transformer_X.npy')
y_test = np.load('data/processed/transformer_y.npy')

# Make predictions
y_pred = model.predict(X_test).flatten()
y_true = y_test.flatten()

# Compute metrics
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ MAE : {mae:.4f}")
print(f"✅ R²  : {r2:.4f}")

# Save to file
with open("models/transformer_metrics.txt", "w") as f:
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"MAE : {mae:.4f}\n")
    f.write(f"R2  : {r2:.4f}\n")

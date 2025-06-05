# test_models.py (final fixed version)

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from joblib import load as load_joblib
import os

# === 📂 Load Test Data ===
test_data = pd.read_csv("data/processed/test_data.csv")

# 🔍 Drop non-numeric or irrelevant columns (like 'Date', 'news_title', etc.)
drop_cols = [col for col in test_data.columns if not np.issubdtype(test_data[col].dtype, np.number)]
print(f"Dropping non-numeric columns from test set: {drop_cols}")
test_data.drop(columns=drop_cols, inplace=True)

# Separate features and target
feature_cols = [col for col in test_data.columns if col != 'Close']
X_test = test_data[feature_cols].values
y_test = test_data['Close'].values

# === 🌲 Test Random Forest Model ===
print("\n🔍 Testing Random Forest Model")
try:
    rf_model = load_joblib("models/random_forest_regressor.pkl")
    y_pred_rf = rf_model.predict(X_test)

    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))  # updated here
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_r2 = r2_score(y_test, y_pred_rf)

    print(f"✅ RF RMSE: {rf_rmse:.4f}")
    print(f"✅ RF MAE : {rf_mae:.4f}")
    print(f"✅ RF R²  : {rf_r2:.4f}")
    print(f"✅ RF Accuracy Estimate ≈ {(1 - (rf_rmse / np.mean(y_test))) * 100:.2f}%")
except Exception as e:
    print(f"❌ Random Forest testing failed: {e}")
 
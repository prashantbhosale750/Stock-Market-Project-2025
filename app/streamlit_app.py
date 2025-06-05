# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load as load_joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

# === Load Data & Model ===
data_path = "data/processed/test_data.csv"
model_path = "models/random_forest_regressor.pkl"

df = pd.read_csv(data_path)
model = load_joblib(model_path)

# Drop non-numeric columns (keep for display)
display_df = df.copy()
feature_cols = [col for col in df.columns if col not in ['Date', 'Headline', 'Close', 'sentiment_positive', 'sentiment_neutral', 'sentiment_negative']]
df_model = df.drop(columns=['Date', 'Headline', 'Close'])

# === Streamlit UI ===
st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title("📈 Stock Closing Price Predictor")
st.markdown("Enter a **specific date** below to view the predicted closing price.")

# Date input
date_input = st.date_input("📅 Select a Date", min_value=datetime(2016,1,1), max_value=datetime(2022,12,31))

# Search & Predict
if st.button("🔍 Predict Price"):
    date_str = date_input.strftime("%Y-%m-%d")
    if date_str not in display_df['Date'].values:
        st.error("❌ No data available for the selected date. Try another date.")
    else:
        row = display_df[display_df['Date'] == date_str]
        features = row[feature_cols].values
        prediction = model.predict(features)[0]

        st.success(f"✅ Predicted Closing Price for {date_str}: ₹{prediction:.2f}")

        # Evaluation metrics
        X_test = df_model.values
        y_test = df['Close'].values
        y_pred = model.predict(X_test)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.markdown("---")
        st.markdown("### 📊 Model Evaluation Metrics")
        st.write(f"**RMSE**: {rmse:.4f}")
        st.write(f"**MAE**: {mae:.4f}")
        st.write(f"**R² Score**: {r2:.4f}")

        # Plot closing prices with prediction marker
        st.markdown("---")
        st.markdown("### 📉 Historical Prices with Prediction")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(display_df['Date'], display_df['Close'], label="Actual Closing Price")
        ax.axvline(x=date_str, color='red', linestyle='--', label=f"Predicted: {date_str}")
        ax.scatter(date_str, prediction, color='red', zorder=5)
        ax.set_title("Stock Closing Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (₹)")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Optional: Feature importance
        st.markdown("---")
        st.markdown("### 🔍 Feature Importance")
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        sorted_features = [feature_cols[i] for i in sorted_idx]

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.barh(sorted_features[::-1], importances[sorted_idx][::-1])
        ax2.set_title("Feature Importance")
        ax2.set_xlabel("Importance")
        st.pyplot(fig2)

st.markdown("---")
st.caption("📌 Note: Only works for dates available in test dataset.")

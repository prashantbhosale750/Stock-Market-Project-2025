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
feature_cols = ['Open', 'High', 'Low', 'Volume', 'sentiment_neutral']
df_model = df[feature_cols]

# === Streamlit UI ===
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    .reportview-container .markdown-text-container { font-family: 'Segoe UI', sans-serif; }
    .stButton>button { background-color: #0073e6; color: white; border-radius: 6px; }
    .stSidebar { background-color: #e3ecf7; }
    .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.image("https://img.icons8.com/clouds/100/stock-share.png", width=100)
st.sidebar.title("📊 Options")

nav = st.sidebar.radio("Navigate", ["📈 Predict Price", "📉 Evaluation Metrics", "📋 Feature Importance"])

# === Shared UI Elements ===
date_input = st.sidebar.date_input("📅 Select a Date", min_value=datetime(2016,1,1), max_value=datetime(2022,12,31))
use_custom_sentiment = st.sidebar.checkbox("🎭 Adjust Sentiment")
custom_sentiment = st.sidebar.slider("Sentiment Neutral Score", 0.0, 1.0, 0.5, step=0.01)

# === Page 1: Predict Price ===
if nav == "📈 Predict Price":
    st.title("📈 Stock Closing Price Predictor with Sentiment Adjustment")
    st.markdown("Use this app to predict closing price based on historical and sentiment data.")

    date_str = date_input.strftime("%Y-%m-%d")
    if date_str not in display_df['Date'].values:
        st.warning("❌ No data available for the selected date.")
    else:
        row = display_df[display_df['Date'] == date_str]
        feature_values = row[feature_cols].copy()

        if use_custom_sentiment:
            feature_values['sentiment_neutral'] = custom_sentiment
            st.info(f"Custom Sentiment Score applied: {custom_sentiment:.2f}")

        prediction = model.predict(feature_values.values)[0]
        st.success(f"✅ Predicted Closing Price on {date_str}: ₹{prediction:.2f}")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(display_df['Date'], display_df['Close'], label="Actual Closing Price")
        ax.axvline(x=date_str, color='red', linestyle='--', label=f"Prediction Date")
        ax.scatter(date_str, prediction, color='red', zorder=5)
        ax.set_title("Stock Closing Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (₹)")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

# === Page 2: Evaluation Metrics ===
elif nav == "📉 Evaluation Metrics":
    st.title("📉 Model Evaluation Metrics")
    X_test = df_model.values
    y_test = df['Close'].values
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.metric(label="RMSE", value=f"{rmse:.2f}")
    st.metric(label="MAE", value=f"{mae:.2f}")
    st.metric(label="R² Score", value=f"{r2:.2f}")

# === Page 3: Feature Importance ===
elif nav == "📋 Feature Importance":
    st.title("📋 Feature Importance")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [feature_cols[i] for i in sorted_idx]

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.barh(sorted_features[::-1], importances[sorted_idx][::-1], color="#0073e6")
    ax2.set_title("Random Forest Feature Importance")
    ax2.set_xlabel("Importance Score")
    st.pyplot(fig2)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit | For educational purposes")



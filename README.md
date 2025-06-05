# 📊 Smart Stock Market Prediction & Sentiment Analysis : ALCOA corp. 

This project analyzes and predicts stock prices using historical price data and financial news sentiment.

## 🔍 Features
- Fetches historical stock price data using Yahoo Finance
- Collects recent financial news using NewsAPI
- Merges and prepares data for sentiment and time-series analysis
- Future modules: LSTM forecasting, sentiment scoring, Streamlit dashboard

## 🗂️ Project steps

# PHASE 1: Data Ingestion & Merging
Step	Task	Tools	Output
1.1	Load raw stock & news data	pandas	DataFrames
1.2	Convert Date columns to datetime	pandas	—
1.3	Shift news by +1 day to reflect next-day impact	pandas	—
1.4	Filter news to match only valid trading days	pandas	—
1.5	Group news headlines by date	pandas	Text blob per date
1.6	Merge stock & news data on Date	pandas	merged_stock_news.csv
📁 Code File: src/merger.py
📂 Data Output: data/processed/merged_stock_news.csv

# PHASE 2: Sentiment Analysis (NLP)
Step	Task	Tools	Output
2.1	Apply sentiment analysis on grouped headlines	VADER or FinBERT	Sentiment score per day
2.2	Add sentiment column to merged data	pandas	merged_sentiment.csv
2.3	(Optional) Clean headlines, summarize long texts	nltk / transformers	—
📁 Code File: src/sentiment_analyzer.py
What is VADER?
VADER (Valence Aware Dictionary and sEntiment Reasoner) is a pre-trained sentiment analysis model designed specifically for social media and short texts like news headlines.It outputs a compound score and classifies sentiment based on that score.
![alt text](output.png)
✅ Here's the pie chart showing the distribution of sentiment labels predicted by FinBERT on your headlines:
🟧 Negative: 56.2% 
🟠 Neutral: 31.3% 
🔴 Positive: 12.5% 
This reflects a cautious market tone in the financial news related to your stock (Alcoa) — which is realistic in business reporting. 

# 





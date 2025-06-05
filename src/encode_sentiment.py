import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/processed/merged_finbert_encoded.csv")

# Create sentiment_label column from finbert_sentiment
if 'sentiment_label' not in df.columns and 'finbert_sentiment' in df.columns:
    le = LabelEncoder()
    df['sentiment_label'] = le.fit_transform(df['finbert_sentiment'])
    print("✅ Sentiment label encoded successfully.")
else:
    print("⚠️ Either sentiment_label already exists or finbert_sentiment column is missing.")

# Save updated file
df.to_csv("data/processed/merged_finbert_encoded.csv", index=False)
print("📁 File saved: data/processed/merged_finbert_encoded.csv")


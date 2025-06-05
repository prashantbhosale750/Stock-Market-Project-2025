import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os


# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

def apply_sentiment_analysis(input_path, output_path=None):
    # Load the merged stock-news dataset
    df = pd.read_csv(input_path)

    # Initialize VADER
    sia = SentimentIntensityAnalyzer()

    # Apply sentiment score to each headline
    def get_score(text):
        if pd.isna(text):
            return 0
        return sia.polarity_scores(text)['compound']

    df['sentiment_score'] = df['Headline'].apply(get_score)

    # Convert sentiment score into label
    def get_label(score):
        if score >= 0.05:
            return 'positive'
        elif score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    df['sentiment_label'] = df['sentiment_score'].apply(get_label)

    # Save the output
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Sentiment-scored data saved to: {output_path}")

    return df

if __name__ == "__main__":
    input_file = "data/processed/merged_stock_news.csv"
    output_file = "data/processed/merged_sentiment.csv"
    
    result_df = apply_sentiment_analysis(input_file, output_file)
    print("✅ Sample Output with Sentiment Columns:")
    print(result_df[['Date', 'Close', 'Headline', 'sentiment_score', 'sentiment_label']].head())

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import os

def load_model():
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def predict_sentiment(texts, tokenizer, model, batch_size=32):
    sentiments = []
    model.eval()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probs, dim=1).tolist()

            for pred in predictions:
                label = ['negative', 'neutral', 'positive'][pred]
                sentiments.append(label)

    return sentiments


def apply_finbert_sentiment(input_file, output_file):
    df = pd.read_csv(input_file)

    print("🚀 Loading FinBERT model...")
    tokenizer, model = load_model()

    print("🔍 Predicting sentiment for headlines...")
    df = df.dropna(subset=['Headline'])
    df['finbert_sentiment'] = predict_sentiment(df['Headline'].tolist(), tokenizer, model)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✅ Saved with FinBERT sentiment to: {output_file}")

if __name__ == "__main__":
    input_path = "data/processed/merged_stock_news.csv"
    output_path = "data/processed/merged_finbert_sentiment.csv"
    apply_finbert_sentiment(input_path, output_path)

import pandas as pd
import os

def merge_stock_and_news(stock_path, news_path, save_path=None):
    # Step 1.1: Load raw stock & news data
    stock_df = pd.read_csv(stock_path)
    news_df = pd.read_csv(news_path)

    # Step 1.2: Convert 'Date' columns to datetime objects
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    news_df['Date'] = pd.to_datetime(news_df['Date'], errors='coerce')

    # Step 1.3: Shift news dates by +1 day to simulate next-day market impact
    news_df['Date'] = news_df['Date'] + pd.Timedelta(days=1)

    # Step 1.4: Filter news data to only include dates present in stock data
    valid_dates = stock_df['Date'].unique()
    news_df = news_df[news_df['Date'].isin(valid_dates)]

    # Step 1.5: Group news headlines by Date
    news_grouped = news_df.groupby('Date')['Headline'].apply(lambda x: ' || '.join(x)).reset_index()

    # Step 1.6: Merge the grouped news with the stock data on 'Date'
    merged_df = pd.merge(stock_df, news_grouped, on='Date', how='left')

    # Step 1.7: Save the merged DataFrame
    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        merged_df.to_csv(save_path, index=False)

    return merged_df

if __name__ == "__main__":
    stock_file = r"D:\AA Intellipaat Course\Stock Market Project 2025\data\row\AA Stock data.csv"
    news_file = r"D:\AA Intellipaat Course\Stock Market Project 2025\data\row\AA news headlines.csv"
    output_file = r"D:\AA Intellipaat Course\Stock Market Project 2025\data\processed\merged_stock_news.csv"


    merged_data = merge_stock_and_news(stock_file, news_file, save_path=output_file)

    print("✅ Merged Stock + News Data Sample:")
    print(merged_data.head())


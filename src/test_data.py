import pandas as pd
from sklearn.model_selection import train_test_split

# Load the encoded dataset
df = pd.read_csv("data/processed/merged_finbert_encoded.csv")

# Shuffle and split the dataset: 80% train, 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the test set
test_df.to_csv("data/processed/test_data.csv", index=False)

print(f"✅ Test dataset generated with {len(test_df)} records.")

import pandas as pd
from sklearn.model_selection import train_test_split
from models.data.telomere_dataset import TelomereDataset

df = pd.read_csv('data/processed/telomere_data_large.csv')
train_df, test_df = train_test_split(df, test_size=0.7, random_state=42, shuffle=True)

train_df.to_csv('data/processed/telomere_data_large_train.csv', index=False)
test_df.to_csv('data/processed/telomere_data_large_test.csv', index=False)

print(f"Train set: {len(train_df)} rows")
print(f"Test set: {len(test_df)} rows")

train_dataset = TelomereDataset(data_file='data/processed/telomere_data_large_train.csv')
test_dataset = TelomereDataset(data_file='data/processed/telomere_data_large_test.csv')
# load_data.py
from src.data_processing import DataProcessor

processor = DataProcessor()
df, summary = processor.process(
    '../data/raw/raw_data.csv',
    '../data/processed/processed_data.csv'
)

print("Cluster Summary:")
print(summary)
# load_data.py
from src.data_processing import DataProcessor

# Initialize processor
processor = DataProcessor()

# Process the data
df, summary = processor.process(
    input_path='data/raw/raw_data.csv',                # ← Relative to current directory
    output_path='data/processed/processed_data.csv'
)

print("Processing completed!")
print("\nCluster Summary:")
print(summary)
# tests/test_data_processing.py
import pytest
import pandas as pd
from src.data_processing import DataProcessor

def test_rfm_creation():
    df = pd.DataFrame({
        'AccountId': [1, 1, 2],
        'TransactionStartTime': pd.to_datetime(['2025-06-01', '2025-06-05', '2025-06-10']),
        'Amount': [100, 200, 50],
        'Value': [100, 200, 50],
        'TransactionId': [1, 2, 3]
    })
    processor = DataProcessor(snapshot_date=pd.to_datetime('2025-06-11'))
    rfm = processor.create_rfm_features(df)
    assert len(rfm) == 2
    assert rfm.loc[rfm['AccountId'] == 1, 'Recency'].values[0] == 6
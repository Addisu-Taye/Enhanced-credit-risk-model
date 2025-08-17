import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.data_processing import create_feature_pipeline, create_rfm_features

@pytest.fixture
def sample_transaction_data():
    return pd.DataFrame({
        'TransactionId': ['T1', 'T2', 'T3'],
        'CustomerId': ['C1', 'C1', 'C2'],
        'Amount': [100, 150, -50],
        'Value': [100, 150, 50],
        'ProductCategory': ['Electronics', 'Clothing', 'Food'],
        'ChannelId': ['Web', 'Mobile', 'Web'],
        'TransactionStartTime': [
            datetime(2023, 1, 1),
            datetime(2023, 1, 5),
            datetime(2023, 1, 10)
        ]
    })

def test_create_feature_pipeline(sample_transaction_data):
    pipeline = create_feature_pipeline()
    transformed = pipeline.fit_transform(sample_transaction_data)
    
    # Test numerical features were scaled
    assert transformed.shape[0] == 3
    assert not np.isnan(transformed).any()
    
def test_create_rfm_features(sample_transaction_data):
    risk_labels = create_rfm_features(sample_transaction_data)
    
    # Test we get one label per customer
    assert len(risk_labels) == 2
    assert risk_labels.index.name == 'CustomerId'
    assert risk_labels.isin([0, 1]).all()

def test_rfm_cluster_identification(sample_transaction_data):
    risk_labels = create_rfm_features(sample_transaction_data)
    # Verify high-risk customers are correctly identified
    # C2 has negative amount (credit) and only 1 transaction
    assert risk_labels.loc['C2'] == 1
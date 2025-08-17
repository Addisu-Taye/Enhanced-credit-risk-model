# data_processing.py
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def create_rfm_features(df):
    # Calculate RFM metrics
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    })
    
    # Log transform to handle skewness
    rfm['Monetary'] = rfm['Monetary'].apply(lambda x: np.log1p(x) if x > 0 else 0)
    
    # Scale features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    # Cluster customers
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(rfm_scaled)
    
    # Identify high-risk cluster (cluster with lowest monetary and frequency)
    cluster_stats = rfm.groupby(clusters).mean()
    high_risk_cluster = cluster_stats['Monetary'].idxmin()
    
    # Create target variable
    rfm['is_high_risk'] = (clusters == high_risk_cluster).astype(int)
    
    return rfm['is_high_risk']
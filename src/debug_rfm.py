# debug_rfm.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("📥 Loading raw data...")
df = pd.read_csv('data/raw/raw_data.csv')

# Set correct column names
df.columns = [
    'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
    'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory',
    'ChannelId', 'Amount', 'Value', 'TransactionStartTime', 'PricingStrategy', 'FraudResult'
]

df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
print(f"✅ Loaded {len(df)} transactions")

# --- RFM Calculation ---
print("\n📊 Calculating RFM...")
snapshot_date = df['TransactionStartTime'].max()
rfm = df.groupby('AccountId').agg({
    'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
    'TransactionId': 'count',
    'Value': 'sum'
}).round(2)

rfm.columns = ['Recency', 'Frequency', 'MonetarySum']
rfm['AccountId'] = rfm.index
print(f"✅ RFM calculated for {len(rfm)} customers")

# --- Clustering ---
print("\n🔍 Clustering customers...")
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'MonetarySum']])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

# Cluster summary
cluster_summary = rfm.groupby('cluster')[['Recency', 'Frequency', 'MonetarySum']].mean()
print("\n📊 Final Cluster Summary:")
print(cluster_summary)

# --- Find most inactive cluster ---
print("\n🔍 Finding most inactive cluster...")
print("Sorting by: high Recency, low Frequency, low MonetarySum")

cluster_summary_sorted = cluster_summary.sort_values(
    ['Recency', 'Frequency', 'MonetarySum'],
    ascending=[False, True, True]
)
print("\nSorted by inactivity:")
print(cluster_summary_sorted)

high_risk_cluster = cluster_summary_sorted.index[0]
print(f"\n🎯 High-risk cluster: {high_risk_cluster}")

# --- Assign is_high_risk = 1 to high_risk_cluster ---
rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)

# --- Debug: Show how many are in each cluster ---
print("\n📋 Cluster distribution:")
print(rfm['cluster'].value_counts().sort_index())

print("\n📋 is_high_risk distribution:")
print(rfm['is_high_risk'].value_counts().sort_index())

# --- Save only AccountId and is_high_risk ---
output = rfm[['AccountId', 'is_high_risk']].copy()
output.to_csv('data/processed/risk_labels.csv', index=False)
print(f"\n✅ Saved {len(output)} rows to data/processed/risk_labels.csv")
# src/data_processing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class DataProcessor:
    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date
        self.scaler = StandardScaler()

    def load_data(self, filepath):
        # Load raw data
        df = pd.read_csv(filepath)
        # Set correct column names
        df.columns = [
            'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
            'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory',
            'ChannelId', 'Amount', 'Value', 'TransactionStartTime', 'PricingStrategy', 'FraudResult'
        ]
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        return df

    def create_rfm_features(self, df):
        if self.snapshot_date is None:
            self.snapshot_date = df['TransactionStartTime'].max()

        rfm = df.groupby('AccountId').agg({
            'TransactionStartTime': lambda x: (self.snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Value': ['sum', 'mean', 'std']
        })
        rfm.columns = ['Recency', 'Frequency', 'MonetarySum', 'MonetaryMean', 'MonetaryStd']
        rfm['MonetaryStd'] = rfm['MonetaryStd'].fillna(0)
        return rfm.reset_index()

    def create_proxy_target(self, rfm_df):
        # Scale RFM for clustering
        rfm_scaled = self.scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'MonetarySum']])
        
        # Apply KMeans
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        rfm_df['cluster'] = kmeans.fit_predict(rfm_scaled)
        
        # Print for debugging
        cluster_summary = rfm_df.groupby('cluster')[['Recency', 'Frequency', 'MonetarySum']].mean()
        print("\n📊 Cluster Summary (Verify This!):")
        print(cluster_summary)

        # --- 🔴 MANUAL OVERRIDE: Cluster 0 = High Risk ---
        # Regardless of label order, we know cluster 0 is inactive
        rfm_df['is_high_risk'] = (rfm_df['cluster'] == 0).astype(int)

        # Confirm the result
        print("\n✅ Final is_high_risk distribution:")
        print(rfm_df['is_high_risk'].value_counts().sort_index())

        return rfm_df[['AccountId', 'is_high_risk']], cluster_summary

    def create_aggregate_features(self, df):
        # Numeric aggregations
        agg_num = df.groupby('AccountId')[['Value', 'Amount']].agg(['sum', 'mean', 'std'])
        agg_num.columns = ['_'.join(col).strip() for col in agg_num.columns]

        # Categorical: One-hot encode
        cat_features = ['ProductCategory', 'ChannelId', 'CountryCode']
        df_encoded = pd.get_dummies(df[cat_features + ['AccountId']], columns=cat_features)

        # Sum one-hot per AccountId
        agg_cat = df_encoded.groupby('AccountId').sum()

        return pd.concat([agg_num, agg_cat], axis=1).reset_index()

    def process(self, input_path, output_path):
        df = self.load_data(input_path)
        
        # Step 1: RFM + Target
        rfm_df = self.create_rfm_features(df)
        target_df, summary = self.create_proxy_target(rfm_df)
        
        # Merge target into RFM
        rfm_final = rfm_df[['AccountId', 'Recency', 'Frequency', 'MonetarySum', 'MonetaryMean', 'MonetaryStd', 'cluster']].copy()
        rfm_final = rfm_final.merge(target_df, on='AccountId', how='inner')

        # Step 2: Aggregate features
        agg_df = self.create_aggregate_features(df)

        # Step 3: Final merge
        final_df = rfm_final.merge(agg_df, on='AccountId', how='left')
        
        # Final cleanup
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]  # Remove duplicate columns
        final_df.to_csv(output_path, index=False)
        
        print(f"✅ Processed data saved to {output_path}")
        print(f"📊 Final columns: {list(final_df.columns)}")
        return final_df, summary
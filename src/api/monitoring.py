import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score
import mlflow

class DriftDetector:
    def __init__(self, reference_data, model_name="credit_risk_model"):
        self.reference_metrics = self._calculate_reference_metrics(reference_data)
        self.model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
        
    def _calculate_reference_metrics(self, data):
        return {
            'feature_means': data.mean().to_dict(),
            'feature_stds': data.std().to_dict(),
            'class_balance': data['is_high_risk'].mean()
        }
    
    def check_drift(self, new_data):
        results = {}
        
        # Data drift
        current_means = new_data.mean().to_dict()
        for feature, ref_mean in self.reference_metrics['feature_means'].items():
            current_mean = current_means.get(feature)
            if current_mean is not None:
                z_score = (current_mean - ref_mean) / self.reference_metrics['feature_stds'][feature]
                results[f'{feature}_drift'] = abs(z_score) > 3  # 3 sigma rule
        
        # Concept drift
        if 'is_high_risk' in new_data:
            y_true = new_data['is_high_risk']
            y_pred = self.model.predict(new_data.drop('is_high_risk', axis=1))
            current_auc = roc_auc_score(y_true, y_pred)
            results['performance_drift'] = current_auc < 0.7  # Threshold
        
        return results

def log_drift_metrics(drift_results):
    with mlflow.start_run(run_name="drift_monitoring"):
        mlflow.log_metrics({
            'data_drift': int(any(v for k,v in drift_results.items() if 'drift' in k)),
            'performance_drift': int(drift_results.get('performance_drift', False))
        })
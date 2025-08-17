# train.py
import mlflow
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def train_and_evaluate(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize models
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Train and evaluate models
    best_score = 0
    best_model = None
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Log metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            mlflow.log_metrics({
                'accuracy': report['accuracy'],
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1': report['weighted avg']['f1-score'],
                'roc_auc': roc_auc_score(y_test, y_proba)
            })
            
            # Log model
            mlflow.sklearn.log_model(model, name)
            
            # Track best model
            current_score = roc_auc_score(y_test, y_proba)
            if current_score > best_score:
                best_score = current_score
                best_model = name
    
    # Register best model
    if best_model:
        mlflow.register_model(
            f"runs:/{mlflow.last_active_run().info.run_id}/{best_model}",
            "credit_risk_model")
    
    return best_model
# src/train.py
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import os
import json
from mlflow.tracking import MlflowClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Configuration ===
TRACKING_URI = os.path.abspath("mlruns")  # Use absolute path for Windows
EXPERIMENT_NAME = "credit-risk-model"
MODEL_NAME = "CreditRiskModel"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
MODELS_DIR = "models"

# Set MLflow tracking URI (Windows-safe)
mlflow.set_tracking_uri(f"file:///{TRACKING_URI.replace(os.sep, '/')}")
mlflow.set_experiment(EXPERIMENT_NAME)


def evaluate(y_true, y_pred, y_prob):
    """Evaluate model performance."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }


def remove_duplicate_columns(df):
    """Remove duplicate columns by keeping the first occurrence."""
    return df.loc[:, ~df.columns.duplicated()]


def clean_column_names(df):
    """Remove suffixes like _x, _y from merge artifacts."""
    df.columns = df.columns.str.replace(r'\.(x|y)$', '', regex=True)
    df.columns = df.columns.str.replace(r'_x$', '', regex=True)
    df.columns = df.columns.str.replace(r'_y$', '', regex=True)
    return df


def train():
    logger.info("🔍 Loading processed data...")
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"❌ Processed data not found at {PROCESSED_DATA_PATH}")

    df = pd.read_csv(PROCESSED_DATA_PATH)

    # --- Clean column names and duplicates ---
    df = clean_column_names(df)
    df = remove_duplicate_columns(df)

    # --- Validate target ---
    if 'is_high_risk' not in df.columns:
        raise ValueError("❌ Target column 'is_high_risk' not found. Re-run data processing.")

    # --- Prepare features ---
    y = df['is_high_risk']
    drop_cols = ['is_high_risk', 'AccountId', 'cluster']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Keep only numeric columns and fill missing values
    X = X.select_dtypes(include=[np.number]).fillna(0)
    X = X.replace([np.inf, -np.inf], 0)  # Replace inf values
    X = X.loc[:, X.var() != 0]  # Remove zero-variance features

    logger.info(f"📊 Data shape: {X.shape}, Target distribution:\n{y.value_counts().sort_index()}")

    # --- Train-Test Split ---
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        logger.warning("Stratify failed, falling back to random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # --- Scale Features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Models to Train ---
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    }

    best_model = None
    best_auc = 0
    best_name = ""

    # --- Train and Evaluate ---
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            logger.info(f"🚀 Training {name}...")

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

            metrics = evaluate(y_test, y_pred, y_prob)

            # Log to MLflow
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model,
                "model",
                input_example=X_test[:2]  # Auto-infer signature
            )

            logger.info(f"✅ {name} - ROC AUC: {metrics['roc_auc']:.4f}")

            if metrics['roc_auc'] > best_auc:
                best_auc = metrics['roc_auc']
                best_model = model
                best_name = name

    logger.info(f"🏆 Best model: {best_name} with AUC = {best_auc:.4f}")

    # --- Save Artifacts ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    # Save feature names for API alignment
    feature_columns = X.columns.tolist()
    features_path = os.path.join(MODELS_DIR, "feature_columns.json")
    with open(features_path, 'w') as f:
        json.dump(feature_columns, f)

    # --- Log Best Model to MLflow ---
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            best_model,
            "best_model",
            input_example=X_test[:2]
        )
        mlflow.log_artifact(scaler_path, "scaler")
        mlflow.log_artifact(features_path, "features")

        # ✅ Only log floats as metrics
        mlflow.log_metric("best_model_auc", best_auc)
        mlflow.log_metric("n_features", len(feature_columns))
        mlflow.log_metric("n_train_samples", len(X_train))
        mlflow.log_metric("n_test_samples", len(X_test))

        # ✅ Log strings as parameters
        mlflow.log_param("best_model_name", best_name)
        mlflow.log_param("class_balance", "balanced")

    # --- Register Model in MLflow ---
    client = MlflowClient()
    model_uri = f"runs:/{run.info.run_id}/best_model"

    try:
        client.create_registered_model(MODEL_NAME)
        logger.info(f"✅ Created new registered model: {MODEL_NAME}")
    except Exception as e:
        logger.warning(f"⚠️ Model {MODEL_NAME} may already exist. Error: {e}")

    try:
        version = client.create_model_version(
            name=MODEL_NAME,
            source=model_uri,
            run_id=run.info.run_id
        )
        logger.info(f"✅ Model version {version.version} registered: {MODEL_NAME}")
    except Exception as e:
        logger.error(f"❌ Failed to create model version: {e}")

    logger.info("🎉 Training completed. Model and artifacts saved.")
    return best_model, scaler


if __name__ == "__main__":
    train()
# src/api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="Credit Risk Prediction API")

# Set MLflow tracking URI
mlflow.set_tracking_uri("../mlruns")  # Adjust path if needed

# Load model from MLflow registry
model = mlflow.sklearn.load_model("models:/CreditRiskModel/Production")

# Load scaler and feature columns
scaler = joblib.load("../models/scaler.joblib")
feature_columns = joblib.load("../models/feature_columns.json")


class CustomerRequest(BaseModel):
    Recency: float
    Frequency: float
    MonetarySum: float
    MonetaryMean: float
    MonetaryStd: float
    Value_sum: float
    Value_mean: float
    Value_std: float
    Amount_sum: float
    Amount_mean: float
    Amount_std: float
    Hour: int
    Day: int
    Month: int
    Year: int
    DayOfWeek: int
    ProductCategory_airtime: int = 0
    ProductCategory_data_bundles: int = 0
    ProductCategory_financial_services: int = 0
    ProductCategory_movies: int = 0
    ProductCategory_other: int = 0
    ProductCategory_ticket: int = 0
    ProductCategory_transport: int = 0
    ProductCategory_tv: int = 0
    ProductCategory_utility_bill: int = 0
    ChannelId_ChannelId_1: int = 0
    ChannelId_ChannelId_2: int = 0
    ChannelId_ChannelId_3: int = 0
    ChannelId_ChannelId_5: int = 0
    CountryCode_256: int = 1


class PredictionResponse(BaseModel):
    risk_probability: float
    credit_score: int  # FICO-like score (e.g., 300–850)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: CustomerRequest):
    # Convert input to DataFrame
    df = pd.DataFrame([request.dict()])

    # Reorder and align with training features
    df = df[feature_columns]

    # Scale
    scaled_data = scaler.transform(df)

    # Predict
    risk_prob = model.predict_proba(scaled_data)[0, 1]
    credit_score = int((1 - risk_prob) * 550 + 300)  # Map to 300–850 range

    return {"risk_probability": float(risk_prob), "credit_score": credit_score}
# src/api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from pydantic import BaseModel

app = FastAPI(title="Credit Risk API", description="BNPL Risk Scoring Engine", version="1.0.0")

# --- Resolve paths ---
ROOT_DIR = Path(__file__).parent.parent.parent
MLRUNS_PATH = ROOT_DIR / "mlruns"
MODELS_DIR = ROOT_DIR / "models"

# --- Set MLflow tracking URI (Windows-safe) ---
TRACKING_URI = f"file:///{MLRUNS_PATH.resolve().as_posix()}"
print(f"🎯 MLflow Tracking URI: {TRACKING_URI}")
mlflow.set_tracking_uri(TRACKING_URI)

# --- Load model ---
try:
    model = mlflow.sklearn.load_model("models:/CreditRiskModel/Production")
    print("✅ Loaded model from MLflow registry (Production)")
except Exception as e:
    print(f"⚠️ Production model not found: {e}")
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)
        runs = client.search_runs(
            experiment_ids=["148739871483647610"],
            filter_string="metrics.best_model_auc > 0.5",
            order_by=["start_time DESC"],
            max_results=1
        )
        if not runs:
            raise Exception("No runs found in experiment")
        run_id = runs[0].info.run_id
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/best_model")
        print(f"✅ Fallback: Loaded model from run {run_id}")
    except Exception as fallback_error:
        raise RuntimeError(f"Failed to load model: {fallback_error}")

# --- Load scaler and feature columns ---
try:
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    print("✅ Loaded scaler from models/scaler.joblib")
except Exception as e:
    raise RuntimeError(f"Failed to load scaler: {e}")

try:
    with open(MODELS_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)
    print("✅ Loaded feature_columns.json")
except Exception as e:
    raise RuntimeError(f"Failed to load feature columns: {e}")


# --- Request Model ---
class CustomerRequest(BaseModel):
    Recency: float
    Frequency: float
    MonetarySum: float
    ProductCategory: str = "airtime"
    Channel: str = "web"
    CountryCode: int = 256
    Hour: int = 12


# --- Prediction Endpoint ---
@app.post("/predict")
def predict(request: CustomerRequest):
    try:
        # Build input DataFrame
        df = pd.DataFrame([{
            "Recency": request.Recency,
            "Frequency": request.Frequency,
            "MonetarySum": request.MonetarySum,
            "Value_sum": request.MonetarySum,
            "Amount_sum": request.MonetarySum,
            "Hour": request.Hour,
            "ProductCategory_airtime": 1 if request.ProductCategory == "airtime" else 0,
            "ProductCategory_data_bundles": 1 if request.ProductCategory == "data_bundles" else 0,
            "ChannelId_ChannelId_1": 1 if request.Channel == "web" else 0,
            "ChannelId_ChannelId_3": 1 if request.Channel == "pay_later" else 0,
            "CountryCode_256": 1 if request.CountryCode == 256 else 0
        }])

        # Align with training features
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]
        df = df.astype('float64')  # Fix MLflow integer warning

        # Scale and predict
        scaled = scaler.transform(df)
        prob = model.predict_proba(scaled)[0, 1]
        score = int((1 - prob) * 550 + 300)
        level = "High Risk" if prob > 0.7 else "Medium Risk" if prob > 0.4 else "Low Risk"

        return {
            "risk_probability": float(prob),
            "credit_score": score,
            "risk_level": level
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Serve User-Friendly Frontend at `/` ---
@app.get("/", response_class=HTMLResponse)
def get_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <title>💳 Credit Risk Calculator</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-align: center;
                margin: 0;
                padding: 20px;
            }
            .container {
                max-width: 600px;
                margin: 40px auto;
                background: rgba(255, 255, 255, 0.1);
                padding: 40px;
                border-radius: 16px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }
            h1 { margin: 0; font-size: 2.2rem; font-weight: 600; }
            p { opacity: 0.9; margin: 10px 0; }
            .form-group {
                text-align: left;
                margin: 15px 0;
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 500;
            }
            input, select {
                width: 100%;
                padding: 12px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
            }
            button {
                margin-top: 25px;
                padding: 14px 30px;
                background: #00c853;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 18px;
                font-weight: bold;
                cursor: pointer;
                transition: 0.3s;
                width: 100%;
            }
            button:hover:not(:disabled) { background: #00e676; }
            button:disabled { background: #9e9e9e; cursor: not-allowed; }

            .result {
                margin-top: 25px;
                padding: 20px;
                background: rgba(0, 0, 0, 0.3);
                border-radius: 12px;
                text-align: left;
            }
            .high-risk { color: #f44336; font-weight: bold; }
            .medium-risk { color: #ff9800; font-weight: bold; }
            .low-risk { color: #00c853; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>💳 Hibret Bank Credit Risk Score</h1>
            <p>Assess BNPL customer risk in seconds</p>

            <form id="riskForm">
                <div class="form-group">
                    <label>Recency (days since last transaction)</label>
                    <input type="number" id="Recency" placeholder="e.g. 30" required />
                </div>

                <div class="form-group">
                    <label>Frequency (number of transactions)</label>
                    <input type="number" id="Frequency" placeholder="e.g. 5" required />
                </div>

                <div class="form-group">
                    <label>Total Transaction Value (UGX)</label>
                    <input type="number" id="MonetarySum" placeholder="e.g. 10000" required />
                </div>

                <div class="form-group">
                    <label>Product Category</label>
                    <select id="ProductCategory">
                        <option value="airtime">Airtime</option>
                        <option value="data_bundles">Data Bundles</option>
                        <option value="financial_services">Financial Services</option>
                        <option value="utility_bill">Utility Bill</option>
                        <option value="movies">Movies</option>
                        <option value="tv">TV</option>
                        <option value="transport">Transport</option>
                        <option value="ticket">Ticket</option>
                        <option value="other">Other</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Channel</label>
                    <select id="Channel">
                        <option value="web">Web</option>
                        <option value="pay_later">Pay Later</option>
                    </select>
                </div>

                <button type="submit">Calculate Credit Score</button>
            </form>

            <div id="result" class="result" style="display: none;">
                <p><strong>Risk Probability:</strong> <span id="riskProb"></span>%</p>
                <p><strong>Credit Score:</strong> <span id="creditScore"></span></p>
                <p><strong>Risk Level:</strong> <span id="riskLevel" class=""></span></p>
            </div>
        </div>

        <script>
            const form = document.getElementById('riskForm');
            const result = document.getElementById('result');
            const riskProb = document.getElementById('riskProb');
            const creditScore = document.getElementById('creditScore');
            const riskLevel = document.getElementById('riskLevel');

            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                const data = {
                    Recency: parseFloat(document.getElementById('Recency').value),
                    Frequency: parseFloat(document.getElementById('Frequency').value),
                    MonetarySum: parseFloat(document.getElementById('MonetarySum').value),
                    ProductCategory: document.getElementById('ProductCategory').value,
                    Channel: document.getElementById('Channel').value
                };

                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });

                    if (!res.ok) {
                        const error = await res.json();
                        alert("❌ " + JSON.stringify(error, null, 2));
                        return;
                    }

                    const json = await res.json();
                    riskProb.textContent = (json.risk_probability * 100).toFixed(2);
                    creditScore.textContent = json.credit_score;
                    riskLevel.textContent = json.risk_level;

                    if (json.risk_level.includes("High")) riskLevel.className = "high-risk";
                    else if (json.risk_level.includes("Medium")) riskLevel.className = "medium-risk";
                    else riskLevel.className = "low-risk";

                    result.style.display = 'block';
                } catch (error) {
                    alert("Request failed: " + error.message);
                }
            });
        </script>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "ok"}
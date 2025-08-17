app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="API for predicting customer credit risk scores",
    version="1.0.0",
    contact={
        "name": "Bati Bank Analytics Team",
        "email": "analytics@batibank.com"
    },
    license_info={
        "name": "Proprietary",
    },
)

@app.post("/predict",
    response_model=RiskPrediction,
    summary="Predict credit risk",
    description="Predicts the credit risk score and probability for a customer",
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "customer_id": "C123",
                        "risk_score": 1,
                        "risk_probability": 0.85
                    }
                }
            }
        },
        422: {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "Amount"],
                                "msg": "field required",
                                "type": "value_error.missing"
                            }
                        ]
                    }
                }
            }
        }
    })
async def predict_risk(customer_data: CustomerData):
    ...
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.pydantic_models import CustomerData

client = TestClient(app)

def test_predict_endpoint():
    test_data = {
        "CustomerId": "C123",
        "Amount": 100.0,
        "Value": 100.0,
        "ProductCategory": "Electronics",
        "ChannelId": "Web"
    }
    
    response = client.post("/predict", json=test_data)
    
    assert response.status_code == 200
    assert "risk_score" in response.json()
    assert "risk_probability" in response.json()
    assert 0 <= response.json()["risk_probability"] <= 1
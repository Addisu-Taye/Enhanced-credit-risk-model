# src/api/pydantic_models.py
from pydantic import BaseModel
from typing import List

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
    ProductCategory_Digital: int = 0
    ProductCategory_Fashion: int = 0
    ProductCategory_Home_&_Living: int = 0
    ProductCategory_Other: int = 0
    ProductCategory_Sports: int = 0
    ChannelId_Android: int = 0
    ChannelId_Checkout: int = 0
    ChannelId_IOS: int = 0
    ChannelId_PayLater: int = 0
    ChannelId_Web: int = 0
    CountryCode: int

class PredictionResponse(BaseModel):
    risk_probability: float
    credit_score: int
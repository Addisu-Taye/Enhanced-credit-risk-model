#!/usr/bin/env python3
"""
Credit Risk Model Prediction Script

This script handles:
- Loading a trained model from MLflow
- Preprocessing input data
- Making predictions
- Formatting output with proper error handling
"""

import os
import logging
import pandas as pd
import mlflow
import mlflow.sklearn
from typing import Dict, Union
from src.data_processing import create_feature_pipeline
from src.logging_config import logger

# Initialize logging
logger = logging.getLogger(__name__)

class CreditRiskPredictor:
    def __init__(self, model_path: str = None):
        """
        Initialize the predictor by loading model and preprocessing pipeline
        
        Args:
            model_path: Path to MLflow model (default: load from registry)
        """
        try:
            # Load model from MLflow
            if model_path is None:
                logger.info("Loading production model from MLflow registry")
                self.model = mlflow.pyfunc.load_model(
                    model_uri="models:/credit_risk_model/Production"
                )
            else:
                logger.info(f"Loading model from local path: {model_path}")
                self.model = mlflow.pyfunc.load_model(model_uri=model_path)

            # Initialize preprocessing pipeline
            self.preprocessor = create_feature_pipeline()
            
            logger.info("Predictor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {str(e)}")
            raise

    def preprocess_input(self, input_data: Dict) -> pd.DataFrame:
        """
        Preprocess input data to match model requirements
        
        Args:
            input_data: Dictionary of input features
            
        Returns:
            Processed DataFrame ready for prediction
        """
        try:
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Validate required columns
            required_columns = [
                'CustomerId', 'Amount', 'Value', 
                'ProductCategory', 'ChannelId'
            ]
            missing_cols = [col for col in required_columns if col not in input_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Apply preprocessing
            processed_data = self.preprocessor.transform(input_df)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise

    def predict(self, input_data: Dict) -> Dict[str, Union[int, float]]:
        """
        Make prediction on input data
        
        Args:
            input_data: Dictionary containing customer features
            
        Returns:
            Dictionary with risk_score and risk_probability
        """
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            
            # Make prediction
            risk_score = self.model.predict(processed_data)[0]
            risk_prob = self.model.predict_proba(processed_data)[0][1]
            
            # Format output
            return {
                "customer_id": input_data.get("CustomerId"),
                "risk_score": int(risk_score),
                "risk_probability": float(risk_prob),
                "model_version": self.model.metadata.run_id
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for customer {input_data.get('CustomerId')}: {str(e)}")
            raise

def main():
    """Example usage of the predictor"""
    try:
        # Initialize predictor
        predictor = CreditRiskPredictor()
        
        # Sample input data
        test_data = {
            "CustomerId": "CUST_12345",
            "Amount": 150.0,
            "Value": 150.0,
            "ProductCategory": "Electronics",
            "ChannelId": "Mobile"
        }
        
        # Make prediction
        prediction = predictor.predict(test_data)
        print("Prediction Result:", prediction)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    # Configure MLflow tracking URI if not set in environment
    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow.set_tracking_uri("http://localhost:5000")
    
    main()
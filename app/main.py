from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
from typing import List

app = FastAPI(title="Credit Card Fraud Detection API")

# Load models and preprocessing objects
import os

# Get the absolute path to the models directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, 'models')

# Load the models using absolute paths
xgb_model = joblib.load(os.path.join(models_dir, 'xgb_model.joblib'))
lgb_model = joblib.load(os.path.join(models_dir, 'lgb_model.joblib'))
nn_model = tf.keras.models.load_model(os.path.join(models_dir, 'nn_model.keras'))
scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
feature_names = joblib.load(os.path.join(models_dir, 'feature_names.joblib'))

class Transaction(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_confidence: dict

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    try:
        # Validate input
        if len(transaction.features) != len(feature_names):
            raise HTTPException(status_code=400, detail="Invalid number of features")
        
        # Prepare input
        X = np.array(transaction.features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        # Get predictions from all models
        xgb_prob = xgb_model.predict_proba(X_scaled)[0, 1]
        lgb_prob = lgb_model.predict_proba(X_scaled)[0, 1]
        nn_prob = nn_model.predict(X_scaled)[0, 0]
        
        # Ensemble prediction
        proba = np.mean([xgb_prob, lgb_prob, nn_prob])
        prediction = 1 if proba > 0.5 else 0
        
        # Model confidences
        confidences = {
            "xgboost": float(xgb_prob),
            "lightgbm": float(lgb_prob),
            "neural_network": float(nn_prob),
            "ensemble": float(proba)
        }
        
        return PredictionResponse(
            prediction=prediction,
            probability=float(proba),
            model_confidence=confidences
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

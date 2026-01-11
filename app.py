from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
import os

# -----------------------------
# Load model & scaler
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "fraud_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

# -----------------------------
# Initialize FastAPI app
# -----------------------------
app = FastAPI(title="Fraud Detection API", version="1.0")

# -----------------------------
# Request schema
# -----------------------------
class PredictionInput(BaseModel):
    features: List[float]

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def root():
    return {"status": "Fraud Detection API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(data: PredictionInput, threshold: float = 0.5):
    features = np.array(data.features).reshape(1, -1)

    if features.shape[1] != scaler.n_features_in_:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {scaler.n_features_in_} features, got {features.shape[1]}"
        )

    features_scaled = scaler.transform(features)

    # 🔥 Probability prediction
    fraud_probability = model.predict_proba(features_scaled)[0][1]

    prediction = int(fraud_probability >= threshold)

    return {
        "fraud_probability": round(float(fraud_probability), 4),
        "threshold_used": threshold,
        "fraud_prediction": prediction
    }

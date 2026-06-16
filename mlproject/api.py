"""API de prediction FastAPI — Loan Approval Classifier.

Seance 12 - TP FastAPI
    Expose un endpoint POST /predict qui charge le modele depuis
    models/model.joblib et retourne la prediction + probabilite.

Lancement :
    uvicorn mlproject.api:app --reload
    uvicorn mlproject.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
MODEL_NAME = os.getenv("MODEL_NAME", "loan-classifier")
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")

ml: dict = {}


# ---------------------------------------------------------------------------
# S12-1 : Schema des features — colonnes du dataset loan approval
# ---------------------------------------------------------------------------

class Features(BaseModel):
    no_of_dependents: int = Field(..., ge=0, le=20, example=2)
    education: str = Field(..., example="Graduate")
    self_employed: str = Field(..., example="No")
    income_annum: float = Field(..., gt=0, example=5000000)
    loan_amount: float = Field(..., gt=0, example=12000000)
    loan_term: int = Field(..., ge=1, le=30, example=10)
    cibil_score: int = Field(..., ge=300, le=900, example=750)
    residential_assets_value: float = Field(..., example=8000000)
    commercial_assets_value: float = Field(..., example=2000000)
    luxury_assets_value: float = Field(..., example=5000000)
    bank_asset_value: float = Field(..., example=3000000)


# ---------------------------------------------------------------------------
# S12-2 : Schema de la reponse
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = Rejected, 1 = Approved")
    probability: float = Field(..., description="Probabilite d'approbation")
    label: str = Field(..., description="Approved ou Rejected")


# ---------------------------------------------------------------------------
# S12-3 : Chargement du modele au demarrage
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        logger.warning("Modele introuvable : %s — lancez d'abord make train", MODEL_PATH)
    else:
        ml["model"] = joblib.load(MODEL_PATH)
        logger.info("Modele charge depuis %s", MODEL_PATH)
    yield
    ml.clear()


app = FastAPI(
    title="Loan Approval Classifier",
    description="API de prediction d'approbation de pret — Projet MLOps ESGI",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Verificateur de disponibilite du service."""
    return {"status": "ok", "model_loaded": "model" in ml}


# S12-4 : Endpoint de prediction
@app.post("/predict", response_model=PredictionResponse)
def predict(features: Features):
    """Predire l'approbation d'un pret a partir des features du demandeur."""
    if "model" not in ml:
        raise HTTPException(status_code=503, detail="Modele non charge. Lancez make train.")

    df = pd.DataFrame([features.model_dump()])
    proba = float(ml["model"].predict_proba(df)[0, 1])
    prediction = int(proba >= 0.5)

    return PredictionResponse(
        prediction=prediction,
        probability=round(proba, 4),
        label="Approved" if prediction == 1 else "Rejected",
    )


# S12-5 bonus : informations sur le modele deploye
@app.get("/info")
def info():
    """Informations sur le modele actuellement charge."""
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_path": str(MODEL_PATH),
        "model_loaded": "model" in ml,
    }

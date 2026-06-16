"""API de prediction FastAPI.

Seance 12 - TP FastAPI
    Completez les TODO (S12-1 a S12-5 bonus).

Lancement :
    uvicorn mlproject.api:app --reload
    uvicorn mlproject.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))

# Stockage global du modele charge au demarrage
ml: dict = {}


# TODO (S12-1) : definir la classe Features avec les colonnes du dataset :
#   no_of_dependents: int
#   education: str
#   self_employed: str
#   income_annum: float
#   loan_amount: float
#   loan_term: int
#   cibil_score: int
#   residential_assets_value: float
#   commercial_assets_value: float
#   luxury_assets_value: float
#   bank_asset_value: float
class Features(BaseModel):
    """Schema des features d'entree — a completer (S12-1)."""

    pass  # TODO (S12-1) : remplacer par les vraies colonnes


# TODO (S12-2) : definir PredictionResponse avec prediction: int et probability: float
class PredictionResponse(BaseModel):
    """Schema de la reponse de prediction — a completer (S12-2)."""

    pass  # TODO (S12-2)


# TODO (S12-3) : implementer le lifespan pour charger le modele au demarrage
#   import joblib ; ml["model"] = joblib.load(MODEL_PATH)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO (S12-3) : charger le modele ici
    yield
    ml.clear()


app = FastAPI(
    title="Loan Approval Classifier",
    description="API de prediction d'approbation de pret — Projet MLOps ESGI",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Verificateur de disponibilite du service."""
    return {"status": "ok", "model_loaded": "model" in ml}


# TODO (S12-4) : implementer POST /predict
#   @app.post("/predict", response_model=PredictionResponse)
#   def predict(features: Features):
#       if "model" not in ml:
#           raise HTTPException(status_code=503, detail="Modele non charge")
#       import pandas as pd
#       df = pd.DataFrame([features.model_dump()])
#       proba = ml["model"].predict_proba(df)[0, 1]
#       return PredictionResponse(prediction=int(proba >= 0.5), probability=float(proba))


# TODO (S12-5 bonus) : endpoint GET /info exposant MODEL_NAME et MODEL_VERSION
#   depuis les variables d'environnement

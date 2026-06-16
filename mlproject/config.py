"""Configuration centrale du projet — SEUL fichier a adapter pour votre dataset.

Seance 0 — TP Projet personnel
    Completez les TODO (S0-1 a S0-4) pour connecter votre dataset.
    data.py, features.py et les scripts d'entrainement lisent toutes les
    colonnes via ces constantes : ne les modifiez pas ailleurs.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

# S0-1 : chemin vers votre fichier CSV
DATA_PATH = DATA_DIR / "loan_approval_dataset.csv"

# ---------------------------------------------------------------------------
# Dataset — Loan Approval Prediction
# Probleme : predire si un pret sera approuve (1) ou refuse (0)
# Source    : https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset
# ---------------------------------------------------------------------------

# S0-2 : colonne cible (valeurs 0 / 1 apres encodage dans data.py)
TARGET = "loan_status"

# S0-3 : features numeriques
NUMERIC_FEATURES: list[str] = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

# S0-4 : features categorielles
CATEGORICAL_FEATURES: list[str] = [
    "education",
    "self_employed",
]

# ---------------------------------------------------------------------------
# Reproductibilite
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_EXPERIMENT: str = os.getenv("MLFLOW_EXPERIMENT", "loan-approval")
MODEL_NAME: str = os.getenv("MODEL_NAME", "loan-classifier")

MLFLOW_EXPERIMENT_DESCRIPTION: str = (
    "Pipeline MLOps — classification binaire de l'approbation de prets bancaires. "
    "Projet fil rouge ESGI/IABD."
)

def _parse_tags(raw: str) -> dict[str, str]:
    """Convertir 'k1=v1,k2=v2' en {'k1': 'v1', 'k2': 'v2'}."""
    if not raw:
        return {}
    return dict(pair.split("=", 1) for pair in raw.split(",") if "=" in pair)

MLFLOW_EXPERIMENT_TAGS: dict[str, str] = _parse_tags(
    os.getenv("MLFLOW_EXPERIMENT_TAGS", "project=loan-approval,team=esgi")
)

# ---------------------------------------------------------------------------
# Portes qualite (evaluation / validation)
# ---------------------------------------------------------------------------
EVAL_ROC_AUC_MIN: float = float(os.getenv("EVAL_ROC_AUC_MIN", "0.75"))
EVAL_F1_MIN: float = float(os.getenv("EVAL_F1_MIN", "0.70"))

# ---------------------------------------------------------------------------
# API FastAPI
# ---------------------------------------------------------------------------
API_URL: str = os.getenv("API_URL", "http://localhost:8000")

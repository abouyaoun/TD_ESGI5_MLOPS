"""Centralisation de la configuration MLflow (S5-8, S5-9).

Seance 5 - TP MLflow Tracking
    Ce module evite de dupliquer la configuration du tracking dans chaque
    script d'entrainement. Appelez `setup_experiment()` en debut de run et
    `log_dataset()` a l'interieur d'un run actif pour tracer la lignee des
    donnees.
"""
from __future__ import annotations

import logging

import mlflow
import mlflow.data
import pandas as pd

from mlproject.config import (
    DATA_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_EXPERIMENT_DESCRIPTION,
    MLFLOW_EXPERIMENT_TAGS,
    MLFLOW_TRACKING_URI,
    TARGET,
)

logger = logging.getLogger(__name__)


def setup_experiment() -> None:
    """Configurer le tracking MLflow et les metadonnees de l'experience.

    - Pointe vers MLFLOW_TRACKING_URI.
    - Selectionne (ou cree) l'experience MLFLOW_EXPERIMENT.
    - Applique la description et les tags via MlflowClient.

    Operation idempotente : peut etre appelee plusieurs fois sans erreur.
    """
    # S5-8
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT)

    client = mlflow.MlflowClient()
    client.set_experiment_tag(
        experiment.experiment_id,
        "mlflow.note.content",
        MLFLOW_EXPERIMENT_DESCRIPTION,
    )
    for key, value in MLFLOW_EXPERIMENT_TAGS.items():
        client.set_experiment_tag(experiment.experiment_id, key, value)

    logger.info(
        "MLflow experience '%s' configuree (id=%s)",
        MLFLOW_EXPERIMENT,
        experiment.experiment_id,
    )


def log_dataset(df: pd.DataFrame, context: str, name: str = "dataset") -> None:
    """Logger un dataset dans le run MLflow courant (tracabilite donnees → modele).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a referencer (features + cible).
    context : str
        Role du dataset, par exemple ``"training"`` ou ``"evaluation"``.
    name : str, optional
        Nom logique du dataset, par defaut ``"dataset"``.
    """
    # S5-9
    dataset = mlflow.data.from_pandas(
        df,
        source=str(DATA_PATH),
        targets=TARGET,
        name=name,
    )
    mlflow.log_input(dataset, context=context)

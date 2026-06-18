"""Entrainement d'un modele baseline Logistic Regression avec suivi MLflow.

Seance 5 - TP MLflow Tracking
    Le script entrainement un pipeline sklearn (preprocessing + LogReg),
    evalue les performances et trace l'experience dans MLflow.

Lancement :
    python -m mlproject.train
    python -m mlproject.train --c 0.1 --max-iter 500
"""
from __future__ import annotations

import argparse
import logging

import joblib
import matplotlib.pyplot as plt

# S5-1 : imports MLflow
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from mlproject.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    MODEL_NAME,
    RANDOM_STATE,
    TARGET,
)
from mlproject.data import load_data, split
from mlproject.features import build_preprocessor
from mlproject.tracking import log_dataset, setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train(c: float = 1.0, max_iter: int = 1000) -> dict[str, float]:
    """Entrainer un pipeline LogisticRegression et tracker l'experience dans MLflow.

    Parameters
    ----------
    c : float
        Inverse de la regularisation L2.
    max_iter : int
        Nombre maximum d'iterations du solveur.

    Returns
    -------
    dict
        Dictionnaire des metriques {f1, roc_auc}.
    """
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    # S5-2 : configurer le tracking URI et selectionner l'experience
    setup_experiment()

    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("clf", LogisticRegression(C=c, max_iter=max_iter, random_state=RANDOM_STATE)),
    ])

    # S5-3 : encapsuler entrainement + evaluation dans un run MLflow
    with mlflow.start_run(run_name=f"logreg-c{c}"):

        # Tracer la lignee du jeu d'entrainement (S5-9)
        train_df = x_train.copy()
        train_df[TARGET] = y_train.values
        log_dataset(train_df, context="training", name="train")

        model.fit(x_train, y_train)

        proba = model.predict_proba(x_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        metrics = {
            "f1": float(f1_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        }
        logger.info("f1=%.3f  roc_auc=%.3f", metrics["f1"], metrics["roc_auc"])

        # S5-4 : logger les hyperparametres
        mlflow.log_params({"c": c, "max_iter": max_iter, "model": "logreg"})

        # S5-5 : logger les metriques
        mlflow.log_metrics(metrics)

        # S5-6 : logger le modele
        mlflow.sklearn.log_model(model, artifact_path="model")

        # S5-7 bonus : matrice de confusion comme artefact
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax)
        ax.set_title(f"Matrice de confusion — LogReg (C={c})")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

    # Sauvegarder le modele localement pour l'API
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    logger.info("Modele sauvegarde dans %s/model.joblib", MODEL_DIR)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c", type=float, default=1.0, help="Regularisation L2 (defaut: 1.0)")
    parser.add_argument("--max-iter", type=int, default=1000, help="Iterations solveur (defaut: 1000)")
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter)


if __name__ == "__main__":
    main()

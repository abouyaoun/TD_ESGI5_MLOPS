"""Optimisation bayesienne des hyperparametres avec Optuna.

Seance 6 - TP Optuna + Model Registry
    Ce module optimise trois familles de modeles (Random Forest, XGBoost,
    LightGBM) via Optuna (TPE sampler) et enregistre le meilleur dans le
    Model Registry MLflow.
    Completez les TODO (S6-1 a S6-7 bonus).

Lancement :
    python -m mlproject.train_optuna
    python -m mlproject.train_optuna --n-trials 50 --cv 3
    python -m mlproject.train_optuna --no-mlflow
"""
from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

# TODO (S6-1) : importer optuna, RandomForestClassifier (sklearn.ensemble),
#               XGBClassifier (xgboost), LGBMClassifier (lightgbm),
#               cross_val_score (sklearn.model_selection) et RANDOM_STATE
#               (mlproject.config)

from mlproject.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    MODEL_NAME,
)
from mlproject.data import load_data, split
from mlproject.evaluation import log_shap_summary
from mlproject.features import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)


@dataclass
class ModelSpec:
    """Specification d'un modele a optimiser par Optuna."""

    name: str
    suggest_params: Callable  # (trial) -> dict de parametres sklearn prefixes "clf__"
    build_estimator: Callable  # (params) -> estimateur sklearn


def build_model_specs() -> list[ModelSpec]:
    """Definir les trois familles de modeles et leurs espaces de recherche.

    Returns
    -------
    list of ModelSpec
        Random Forest, XGBoost et LightGBM.
    """
    # TODO (S6-2) : retourner une liste de trois ModelSpec.
    #   Exemple pour Random Forest :
    #     ModelSpec(
    #         name="random_forest",
    #         suggest_params=lambda trial: {
    #             "clf__n_estimators": trial.suggest_int("n_estimators", 50, 300),
    #             "clf__max_depth": trial.suggest_int("max_depth", 3, 20, step=1),
    #             "clf__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    #         },
    #         build_estimator=lambda p: RandomForestClassifier(
    #             random_state=RANDOM_STATE, **{k.replace("clf__", ""): v for k, v in p.items()}
    #         ),
    #     )
    #   Idem pour XGBClassifier et LGBMClassifier.
    raise NotImplementedError


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("clf", estimator)])


def objective(trial, spec: ModelSpec, x_train, y_train, cv: int) -> float:
    """Fonction objectif Optuna : score CV ROC AUC.

    Parameters
    ----------
    trial : optuna.Trial
    spec : ModelSpec
    x_train, y_train : donnees d'entrainement
    cv : int
        Nombre de plis de validation croisee.

    Returns
    -------
    float
        Score moyen de validation croisee (ROC AUC).
    """
    # TODO (S6-3) :
    #   - params = spec.suggest_params(trial)
    #   - estimator = spec.build_estimator(params)
    #   - pipeline = build_pipeline(estimator)
    #   - retourner cross_val_score(pipeline, x_train, y_train,
    #       cv=cv, scoring="roc_auc", n_jobs=-1).mean()
    raise NotImplementedError


def run_study(spec: ModelSpec, x_train, y_train, n_trials: int, cv: int):
    """Lancer une etude Optuna pour un modele donne.

    Returns
    -------
    tuple
        (best_params, best_pipeline, cv_score)
    """
    # TODO (S6-4) :
    #   - creer un optuna.create_study(direction="maximize",
    #       sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    #   - lancer study.optimize(lambda trial: objective(trial, spec, x_train, y_train, cv),
    #       n_trials=n_trials, show_progress_bar=True)
    #   - construire le meilleur pipeline avec spec.build_estimator(study.best_params)
    #     et l'entrainer sur x_train / y_train
    #   - retourner (study.best_params, best_pipeline, study.best_value)
    raise NotImplementedError


def log_optuna_run(
    spec_name: str,
    best_pipeline: Pipeline,
    best_params: dict,
    cv_score: float,
    x_test,
    y_test,
    register_as: str | None = None,
) -> None:
    """Logger un run Optuna dans MLflow (run imbrique).

    Parameters
    ----------
    register_as : str, optional
        Si fourni, enregistre dans le Model Registry sous ce nom.
    """
    proba = best_pipeline.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    with mlflow.start_run(run_name=spec_name, nested=True):
        mlflow.set_tag("model_family", spec_name)

        # TODO (S6-5) : logger les hyperparametres (best_params),
        #   les metriques cv_roc_auc, f1, roc_auc et la matrice de confusion
        #   (voir train_models.py comme modele)
        raise NotImplementedError

        # TODO (S6-6) : appeler log_shap_summary si applicable
        # TODO (S6-7 bonus) : enregistrer le modele dans le registry


def train_all(n_trials: int = 30, cv: int = 5, use_mlflow: bool = True) -> None:
    """Entrainer et comparer les trois modeles, sauvegarder le meilleur."""
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    if use_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

    results = []
    for spec in build_model_specs():
        best_params, best_pipeline, cv_score = run_study(spec, x_train, y_train, n_trials, cv)
        proba = best_pipeline.predict_proba(x_test)[:, 1]
        roc = float(roc_auc_score(y_test, proba))
        results.append((spec.name, best_pipeline, best_params, cv_score, roc))

    results.sort(key=lambda r: r[4], reverse=True)
    best_name, best_pipeline, best_params, best_cv, best_roc = results[0]
    logger.info("Meilleur modele : %s (roc_auc=%.3f)", best_name, best_roc)

    if use_mlflow:
        with mlflow.start_run(run_name="optuna-compare"):
            for name, pipeline, params, cv_score, roc in results:
                register_as = MODEL_NAME if name == best_name else None
                log_optuna_run(name, pipeline, params, cv_score, x_test, y_test, register_as)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_DIR / "model.joblib")
    logger.info("Modele sauvegarde dans %s/model.joblib", MODEL_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--no-mlflow", dest="use_mlflow", action="store_false")
    args = parser.parse_args()
    train_all(n_trials=args.n_trials, cv=args.cv, use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()

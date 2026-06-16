"""Optimisation bayesienne des hyperparametres avec Optuna.

Seance 6 - TP Optuna + Model Registry
    Ce module optimise trois familles de modeles (Random Forest, XGBoost,
    LightGBM) via Optuna (TPE sampler) et enregistre le meilleur dans le
    Model Registry MLflow.

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
import optuna
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from mlproject.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    MODEL_NAME,
    RANDOM_STATE,
)
from mlproject.data import load_data, split
from mlproject.evaluation import log_shap_summary
from mlproject.features import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)


# ---------------------------------------------------------------------------
# S6-2 : Specifications des modeles et espaces de recherche
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Specification d'un modele a optimiser par Optuna."""

    name: str
    suggest_params: Callable  # (trial) -> dict prefixe "clf__"
    build_estimator: Callable  # (params) -> estimateur sklearn


def build_model_specs() -> list[ModelSpec]:
    """Definir les trois familles de modeles et leurs espaces de recherche."""
    return [
        ModelSpec(
            name="random_forest",
            suggest_params=lambda trial: {
                "clf__n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "clf__max_depth": trial.suggest_int("max_depth", 3, 20),
                "clf__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "clf__max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            },
            build_estimator=lambda p: RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                **{k.replace("clf__", ""): v for k, v in p.items()},
            ),
        ),
        ModelSpec(
            name="xgboost",
            suggest_params=lambda trial: {
                "clf__n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "clf__max_depth": trial.suggest_int("max_depth", 3, 8),
                "clf__learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "clf__subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "clf__colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            },
            build_estimator=lambda p: XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
                verbosity=0,
                **{k.replace("clf__", ""): v for k, v in p.items()},
            ),
        ),
        ModelSpec(
            name="lightgbm",
            suggest_params=lambda trial: {
                "clf__n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "clf__num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "clf__learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "clf__min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "clf__subsample": trial.suggest_float("subsample", 0.6, 1.0),
            },
            build_estimator=lambda p: LGBMClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
                **{k.replace("clf__", ""): v for k, v in p.items()},
            ),
        ),
    ]


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("clf", estimator)])


# ---------------------------------------------------------------------------
# S6-3 : Fonction objectif
# ---------------------------------------------------------------------------

def objective(trial, spec: ModelSpec, x_train, y_train, cv: int) -> float:
    """Fonction objectif Optuna : ROC AUC moyen en validation croisee."""
    params = spec.suggest_params(trial)
    estimator = spec.build_estimator(params)
    pipeline = build_pipeline(estimator)
    scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


# ---------------------------------------------------------------------------
# S6-4 : Etude Optuna
# ---------------------------------------------------------------------------

def run_study(
    spec: ModelSpec,
    x_train,
    y_train,
    n_trials: int,
    cv: int,
) -> tuple[dict, Pipeline, float]:
    """Lancer une etude Optuna et retourner le meilleur pipeline entraine.

    Returns
    -------
    tuple
        (best_params, best_pipeline_fitted, best_cv_score)
    """
    logger.info("Etude Optuna : %s (%d trials, cv=%d)", spec.name, n_trials, cv)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective(trial, spec, x_train, y_train, cv),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params
    logger.info(
        "%s — meilleur CV ROC AUC : %.4f | params : %s",
        spec.name, study.best_value, best_params,
    )

    # Re-entrainer sur tout x_train avec les meilleurs hyperparametres
    best_estimator = spec.build_estimator(
        {f"clf__{k}": v for k, v in best_params.items()}
    )
    best_pipeline = build_pipeline(best_estimator)
    best_pipeline.fit(x_train, y_train)

    return best_params, best_pipeline, study.best_value


# ---------------------------------------------------------------------------
# S6-5 / S6-6 / S6-7 : Logging MLflow
# ---------------------------------------------------------------------------

def log_optuna_run(
    spec_name: str,
    best_pipeline: Pipeline,
    best_params: dict,
    cv_score: float,
    x_test,
    y_test,
    register_as: str | None = None,
) -> None:
    """Logger un run Optuna dans MLflow (run imbrique sous le run parent)."""
    proba = best_pipeline.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    f1 = float(f1_score(y_test, preds))
    roc = float(roc_auc_score(y_test, proba))

    with mlflow.start_run(run_name=spec_name, nested=True):
        mlflow.set_tag("model_family", spec_name)

        # S6-5 : hyperparametres + metriques
        mlflow.log_params(best_params)
        mlflow.log_metrics({"cv_roc_auc": cv_score, "f1": f1, "roc_auc": roc})

        # Matrice de confusion
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        ax.set_title(f"Matrice de confusion : {spec_name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # Rapport de classification
        report_text = classification_report(y_test, preds)
        mlflow.log_text(report_text, "classification_report.txt")
        mlflow.log_dict(
            classification_report(y_test, preds, output_dict=True),
            "classification_report.json",
        )

        # S6-6 : SHAP summary plot
        log_shap_summary(best_pipeline, x_test, spec_name)

        # S6-7 bonus : enregistrement dans le Model Registry
        signature = infer_signature(x_test, best_pipeline.predict(x_test))
        model_info = mlflow.sklearn.log_model(
            best_pipeline,
            name="model",
            signature=signature,
            input_example=x_test.iloc[:5],
            registered_model_name=register_as,
        )

        if register_as and model_info.registered_model_version:
            _document_registry_version(
                name=register_as,
                version=int(model_info.registered_model_version),
                spec_name=spec_name,
                best_params=best_params,
                cv_score=cv_score,
                f1=f1,
                roc=roc,
            )
            logger.info(
                "Modele '%s' v%s enregistre dans le registry",
                register_as,
                model_info.registered_model_version,
            )


def _document_registry_version(
    name: str,
    version: int,
    spec_name: str,
    best_params: dict,
    cv_score: float,
    f1: float,
    roc: float,
) -> None:
    """Ajouter description et tags a une version du Model Registry."""
    client = mlflow.MlflowClient()
    description = (
        f"Modele : {spec_name}\n"
        f"Hyperparametres : {best_params}\n"
        f"CV ROC AUC : {cv_score:.4f} | F1 : {f1:.4f} | ROC AUC test : {roc:.4f}\n"
        "Optimise par Optuna (TPE sampler) — Seance 6."
    )
    client.update_model_version(name, str(version), description=description)
    for key, value in {
        "model_family": spec_name,
        "search_method": "optuna_tpe",
        "f1": f"{f1:.4f}",
        "roc_auc": f"{roc:.4f}",
        "cv_roc_auc": f"{cv_score:.4f}",
    }.items():
        client.set_model_version_tag(name, str(version), key, value)


# ---------------------------------------------------------------------------
# Orchestration principale
# ---------------------------------------------------------------------------

def train_all(n_trials: int = 30, cv: int = 5, use_mlflow: bool = True) -> None:
    """Entrainer et comparer les trois modeles, sauvegarder le meilleur."""
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    if use_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        logger.info("MLflow : %s (experience : %s)", MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT)

    results = []
    for spec in build_model_specs():
        best_params, best_pipeline, cv_score = run_study(spec, x_train, y_train, n_trials, cv)
        proba = best_pipeline.predict_proba(x_test)[:, 1]
        roc = float(roc_auc_score(y_test, proba))
        results.append((spec.name, best_pipeline, best_params, cv_score, roc))

    results.sort(key=lambda r: r[4], reverse=True)
    best_name, best_pipeline, best_params, best_cv, best_roc = results[0]
    logger.info("Meilleur modele : %s (roc_auc=%.4f)", best_name, best_roc)

    if use_mlflow:
        with mlflow.start_run(run_name="optuna-compare"):
            mlflow.set_tag("best_model", best_name)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("cv", cv)
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

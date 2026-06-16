"""Entrainement et optimisation de plusieurs modeles de classification (AutoML + SHAP).

Seance 7 - TP AutoML & SHAP
    Ce module compare trois familles de modeles (Random Forest, XGBoost,
    LightGBM), chacune optimisee par GridSearchCV, et persiste la meilleure
    dans ``models/model.joblib``.
    Chaque modele est suivi dans MLflow (un run par modele, imbrique sous un
    run parent ``compare-models``) et le meilleur est enregistre dans le
    Model Registry.

Lancement :
    python -m mlproject.train_models
    python -m mlproject.train_models --cv 3 --scoring roc_auc
    python -m mlproject.train_models --no-mlflow
"""
from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from typing import cast

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
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

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)


# ---------------------------------------------------------------------------
# S7-2 : Specifications des modeles et grilles d'hyperparametres
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Specification d'un modele a optimiser."""

    name: str
    estimator: ClassifierMixin
    param_grid: dict


def build_model_specs() -> list[ModelSpec]:
    """Construire la liste des trois modeles a optimiser par GridSearchCV."""
    return [
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_leaf": [1, 2],
            },
        ),
        ModelSpec(
            name="xgboost",
            estimator=XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
                verbosity=0,
            ),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.1, 0.01],
            },
        ),
        ModelSpec(
            name="lightgbm",
            estimator=LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__num_leaves": [31, 63],
                "clf__learning_rate": [0.1, 0.01],
            },
        ),
    ]


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("clf", estimator)])


# ---------------------------------------------------------------------------
# S7-3 : Optimisation par GridSearchCV
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Resultat d'optimisation d'un modele."""

    name: str
    best_estimator: Pipeline
    best_params: dict
    cv_score: float
    f1: float
    roc_auc: float
    preds: np.ndarray


def optimize_model(
    spec: ModelSpec,
    x_train,
    y_train,
    x_test,
    y_test,
    cv: int = 5,
    scoring: str = "roc_auc",
) -> FitResult:
    """Optimiser un modele par GridSearchCV et l'evaluer sur le jeu de test."""
    logger.info("Optimisation de %s (cv=%d, scoring=%s)", spec.name, cv, scoring)

    search = GridSearchCV(
        estimator=build_pipeline(spec.estimator),
        param_grid=spec.param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)

    best = search.best_estimator_
    proba = best.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    logger.info(
        "%s — meilleur CV %s : %.4f | params : %s",
        spec.name, scoring, search.best_score_, search.best_params_,
    )

    return FitResult(
        name=spec.name,
        best_estimator=best,
        best_params=search.best_params_,
        cv_score=float(search.best_score_),
        f1=float(f1_score(y_test, preds)),
        roc_auc=float(roc_auc_score(y_test, proba)),
        preds=preds,
    )


# ---------------------------------------------------------------------------
# S7-4 / S7-5 : Logging MLflow
# ---------------------------------------------------------------------------

def log_run_to_mlflow(
    result: FitResult,
    x_test,
    y_test,
    cv: int,
    scoring: str,
    register_as: str | None = None,
) -> None:
    """Logger un resultat GridSearchCV dans un run MLflow imbrique."""
    with mlflow.start_run(run_name=result.name, nested=True):
        mlflow.set_tag("model_family", result.name)
        mlflow.log_param("cv", cv)
        mlflow.log_param("scoring", scoring)

        # S7-4a : hyperparametres + metriques
        mlflow.log_params(result.best_params)
        mlflow.log_metrics({
            f"cv_{scoring}": result.cv_score,
            "f1": result.f1,
            "roc_auc": result.roc_auc,
        })

        # Matrice de confusion
        cm = confusion_matrix(y_test, result.preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        ax.set_title(f"Matrice de confusion : {result.name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # Rapport de classification
        report_dict = cast(dict, classification_report(y_test, result.preds, output_dict=True))
        mlflow.log_dict(report_dict, "classification_report.json")
        report_text = cast(str, classification_report(y_test, result.preds))
        mlflow.log_text(report_text, "classification_report.txt")

        # S7-4b : SHAP summary plot
        log_shap_summary(result.best_estimator, x_test, result.name)

        # Log + enregistrement dans le registry
        signature = infer_signature(x_test, result.best_estimator.predict(x_test))
        model_info = mlflow.sklearn.log_model(
            result.best_estimator,
            name="model",
            signature=signature,
            input_example=x_test.iloc[:5],
            registered_model_name=register_as,
        )

        # S7-5 bonus : documenter la version dans le registry
        if register_as and model_info.registered_model_version:
            describe_registered_version(
                name=register_as,
                version=int(model_info.registered_model_version),
                result=result,
                cv=cv,
                scoring=scoring,
            )
            logger.info(
                "Modele '%s' v%s enregistre dans le registry",
                register_as,
                model_info.registered_model_version,
            )


def describe_registered_version(
    name: str,
    version: int,
    result: FitResult,
    cv: int,
    scoring: str,
) -> None:
    """Ajouter description et tags a une version du Model Registry."""
    client = mlflow.MlflowClient()
    description = (
        f"Modele : {result.name}\n"
        f"Hyperparametres : {result.best_params}\n"
        f"CV {scoring} : {result.cv_score:.4f} | F1 : {result.f1:.4f} | ROC AUC : {result.roc_auc:.4f}\n"
        "Optimise par GridSearchCV — Seance 7."
    )
    client.update_model_version(name, str(version), description=description)
    for key, value in {
        "model_family": result.name,
        "search_method": "gridsearchcv",
        "cv": str(cv),
        "scoring": scoring,
        "f1": f"{result.f1:.4f}",
        "roc_auc": f"{result.roc_auc:.4f}",
    }.items():
        client.set_model_version_tag(name, str(version), key, value)


# ---------------------------------------------------------------------------
# Orchestration principale
# ---------------------------------------------------------------------------

def train_all(
    cv: int = 5,
    scoring: str = "roc_auc",
    use_mlflow: bool = True,
) -> list[FitResult]:
    """Entrainer et comparer les trois modeles, sauvegarder le meilleur."""
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    if use_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        logger.info("MLflow : %s (experience : %s)", MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT)

    results = [
        optimize_model(spec, x_train, y_train, x_test, y_test, cv=cv, scoring=scoring)
        for spec in build_model_specs()
    ]
    results.sort(key=lambda r: r.roc_auc, reverse=True)
    best = results[0]
    logger.info("Meilleur modele : %s (roc_auc=%.4f)", best.name, best.roc_auc)

    if use_mlflow:
        with mlflow.start_run(run_name="compare-models"):
            mlflow.log_param("cv", cv)
            mlflow.log_param("scoring", scoring)
            mlflow.set_tag("best_model", best.name)
            for result in results:
                register_as = MODEL_NAME if result is best else None
                log_run_to_mlflow(result, x_test, y_test, cv, scoring, register_as)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_estimator, MODEL_DIR / "model.joblib")
    logger.info("Modele sauvegarde dans %s/model.joblib", MODEL_DIR)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--scoring", type=str, default="roc_auc")
    parser.add_argument("--no-mlflow", dest="use_mlflow", action="store_false")
    args = parser.parse_args()
    train_all(cv=args.cv, scoring=args.scoring, use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()

"""Evaluation automatisee et validation qualite du modele.

Seance 11 - TP Tests Donnees & Modele
    `mlflow.models.evaluate` calcule en une passe un ensemble de metriques et
    d'artefacts (matrice de confusion, courbes ROC/precision-rappel).
    `mlflow.validate_evaluation_results` applique ensuite une porte qualite.
    Completez les TODO (S11-1 a S11-3).

Lancement :
    python -m mlproject.evaluate
    python -m mlproject.evaluate --model-uri models:/loan-classifier/1
    python -m mlproject.evaluate --no-validate
"""
from __future__ import annotations

import argparse
import logging

import mlflow
import mlflow.data
import mlflow.models
from mlflow.exceptions import MlflowException
from mlflow.models import MetricThreshold

from mlproject.config import (
    DATA_PATH,
    EVAL_F1_MIN,
    EVAL_ROC_AUC_MIN,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    TARGET,
)
from mlproject.data import load_data, split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def latest_model_uri() -> str:
    """Resoudre l'URI de la derniere version enregistree de MODEL_NAME."""
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(
            f"Aucune version enregistree pour '{MODEL_NAME}'. "
            "Lancez d'abord un entrainement (make train)."
        )
    latest = max(versions, key=lambda v: int(v.version))
    return f"models:/{MODEL_NAME}/{latest.version}"


def build_thresholds() -> dict[str, MetricThreshold]:
    """Construire les seuils de validation.

    Returns
    -------
    dict of str to MetricThreshold
        Seuils minimaux sur roc_auc et f1_score.
    """
    # TODO (S11-1) : retourner {
    #     "roc_auc": MetricThreshold(threshold=EVAL_ROC_AUC_MIN, greater_is_better=True),
    #     "f1_score": MetricThreshold(threshold=EVAL_F1_MIN, greater_is_better=True),
    # }
    raise NotImplementedError


def evaluate_model(model_uri: str | None = None, validate: bool = True):
    """Evaluer un modele du registry et appliquer la porte qualite.

    Parameters
    ----------
    model_uri : str, optional
        URI MLflow du modele. Par defaut, la derniere version de MODEL_NAME.
    validate : bool, optional
        Appliquer la porte qualite, par defaut True.
    """
    df = load_data()
    _, x_test, _, y_test = split(df)
    eval_df = x_test.copy()
    eval_df[TARGET] = y_test.values

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    model_uri = model_uri or latest_model_uri()
    logger.info("Evaluation de %s", model_uri)

    with mlflow.start_run(run_name="evaluate"):
        # TODO (S11-2) :
        #   a) dataset = mlflow.data.from_pandas(eval_df, source=str(DATA_PATH),
        #                                        targets=TARGET, name="eval")
        #      mlflow.log_input(dataset, context="evaluation")
        #   b) result = mlflow.models.evaluate(model_uri, data=eval_df,
        #          targets=TARGET, model_type="classifier", evaluators=["default"])
        #      logger.info("f1_score=%.3f roc_auc=%.3f",
        #                  result.metrics["f1_score"], result.metrics["roc_auc"])

        # TODO (S11-3) : si validate :
        #      mlflow.validate_evaluation_results(build_thresholds(), result)

        raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-uri", default=None)
    parser.add_argument("--no-validate", dest="validate", action="store_false")
    args = parser.parse_args()
    try:
        evaluate_model(model_uri=args.model_uri, validate=args.validate)
    except MlflowException as exc:
        logger.error("Validation echouee : %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

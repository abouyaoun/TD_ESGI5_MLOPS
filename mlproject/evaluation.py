"""Visualisation SHAP pour l'interpretabilite des modeles.

Fourni — ne pas modifier.
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import mlflow
import numpy as np

logger = logging.getLogger(__name__)


def log_shap_summary(pipeline, x_test, model_name: str, max_samples: int = 200) -> None:
    """Generer et logger un SHAP summary plot comme artefact MLflow.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline entraine avec les etapes ``preprocessor`` et ``clf``.
    x_test : pd.DataFrame
        Donnees de test (features brutes).
    model_name : str
        Nom utilise dans le titre du graphique.
    max_samples : int, optional
        Nombre maximum d'echantillons pour le calcul SHAP, par defaut 200.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap non installe — summary plot ignore.")
        return

    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["clf"]

        x_transformed = preprocessor.transform(x_test)
        if hasattr(x_transformed, "toarray"):
            x_transformed = x_transformed.toarray()

        # Noms de features apres transformation
        feature_names = preprocessor.get_feature_names_out()

        x_sample = x_transformed[: min(max_samples, len(x_transformed))]

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(x_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            x_sample,
            feature_names=feature_names,
            show=False,
            plot_type="bar",
        )
        ax.set_title(f"SHAP Summary — {model_name}")
        mlflow.log_figure(fig, "shap_summary.png")
        plt.close(fig)
    except Exception as exc:
        logger.warning("log_shap_summary echoue pour %s : %s", model_name, exc)

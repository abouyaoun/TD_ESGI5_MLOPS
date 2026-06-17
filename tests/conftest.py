"""Fixtures partagées entre tous les tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import mlproject.api as api_module
from mlproject.features import build_preprocessor


def _make_synthetic_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "no_of_dependents": rng.integers(0, 6, n),
            "education": rng.choice(["Graduate", "Not Graduate"], n),
            "self_employed": rng.choice(["Yes", "No"], n),
            "income_annum": rng.integers(200_000, 9_900_000, n),
            "loan_amount": rng.integers(300_000, 39_500_000, n),
            "loan_term": rng.integers(2, 20, n),
            "cibil_score": rng.integers(300, 900, n),
            "residential_assets_value": rng.integers(100_000, 29_000_000, n),
            "commercial_assets_value": rng.integers(0, 19_700_000, n),
            "luxury_assets_value": rng.integers(300_000, 39_200_000, n),
            "bank_asset_value": rng.integers(0, 14_900_000, n),
            "loan_status": rng.integers(0, 2, n),
        }
    )


@pytest.fixture(scope="session")
def trained_model():
    """Pipeline LogReg entraîné sur données synthétiques."""
    df = _make_synthetic_df()
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]
    model = Pipeline(
        [("pre", build_preprocessor()), ("clf", LogisticRegression(max_iter=1000))]
    )
    model.fit(X, y)
    return model


@pytest.fixture()
def loaded_model(trained_model):
    """Injecte le modèle dans l'état global de l'API le temps du test."""
    api_module.ml["model"] = trained_model
    yield trained_model
    api_module.ml.clear()

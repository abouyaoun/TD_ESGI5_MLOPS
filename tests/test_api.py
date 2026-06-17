"""Tests de l'API FastAPI."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mlproject.api import app

client = TestClient(app)

VALID_PAYLOAD = {
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 5_000_000,
    "loan_amount": 12_000_000,
    "loan_term": 10,
    "cibil_score": 750,
    "residential_assets_value": 8_000_000,
    "commercial_assets_value": 2_000_000,
    "luxury_assets_value": 5_000_000,
    "bank_asset_value": 3_000_000,
}


def test_health_returns_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_model_not_loaded_by_default():
    r = client.get("/health")
    assert r.json()["model_loaded"] is False


def test_predict_without_model_returns_503():
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 503


def test_predict_invalid_payload_returns_422():
    r = client.post("/predict", json={"bad_field": "oops"})
    assert r.status_code == 422


def test_predict_with_model_returns_200(loaded_model):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200


def test_predict_response_schema(loaded_model):
    r = client.post("/predict", json=VALID_PAYLOAD)
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in ("Approved", "Rejected")


def test_predict_label_matches_prediction(loaded_model):
    r = client.post("/predict", json=VALID_PAYLOAD)
    body = r.json()
    expected_label = "Approved" if body["prediction"] == 1 else "Rejected"
    assert body["label"] == expected_label


def test_info_endpoint():
    r = client.get("/info")
    assert r.status_code == 200
    assert "model_name" in r.json()

"""Client de test de l'API de prediction.

Seance 15 - TP Tests de l'API
    Teste les endpoints /health, /predict et /info de l'API FastAPI.
    Echantillonne quelques lignes du dataset pour construire les payloads.

Lancement :
    python scripts/predict_client.py
    python scripts/predict_client.py --url http://localhost:8000 --n 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import httpx

sys.path.insert(0, ".")
from mlproject.config import API_URL
from mlproject.data import load_data
from mlproject.config import TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_payloads(n: int = 3) -> list[dict]:
    """Construire des payloads de test a partir du dataset."""
    df = load_data().drop(columns=[TARGET])
    sample = df.sample(n=n, random_state=42)
    return sample.to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=API_URL, help="URL de base de l'API")
    parser.add_argument("--n", type=int, default=3, help="Nombre d'echantillons a tester")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    payloads = build_payloads(args.n)

    with httpx.Client(base_url=base_url, timeout=10.0) as client:

        # S15-1 : verifier la disponibilite de l'API
        logger.info("--- GET /health ---")
        resp = client.get("/health")
        logger.info("Status : %d | %s", resp.status_code, resp.json())

        # S15-2 : envoyer des predictions
        logger.info("--- POST /predict (%d echantillons) ---", len(payloads))
        for i, payload in enumerate(payloads):
            resp = client.post("/predict", json=payload)
            if resp.status_code == 200:
                result = resp.json()
                logger.info(
                    "Sample %d → %s (proba=%.3f)",
                    i + 1,
                    result["label"],
                    result["probability"],
                )
            else:
                logger.error("Sample %d → erreur %d : %s", i + 1, resp.status_code, resp.text)

        # Infos modele
        logger.info("--- GET /info ---")
        resp = client.get("/info")
        logger.info("Modele : %s", json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    main()

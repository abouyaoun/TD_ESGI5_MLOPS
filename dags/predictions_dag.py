from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

API_URL = os.getenv("API_URL", "http://api:8000")
N_PREDICTIONS = 10

default_args = {
    "owner": "ayman",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def task_send_predictions(**context):
    import httpx
    import pandas as pd

    df = pd.read_csv("/opt/airflow/data/loan_approval_dataset.csv")
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["loan_id", "loan_status"], errors="ignore")

    # TODO S17-6 : échantillonner les données
    sample = df.sample(n=N_PREDICTIONS)

    sent = 0
    # TODO S17-7 : envoyer les prévisions à l'API
    with httpx.Client(base_url=API_URL, timeout=10.0) as client:
        client.get("/health").raise_for_status()
        for _, row in sample.iterrows():
            payload = json.loads(row.to_json())
            client.post("/predict", json=payload).raise_for_status()
            sent += 1

    print(f"{sent} prévisions envoyées à {API_URL}/predict")
    return sent


with DAG(
    dag_id="daily_predictions",
    default_args=default_args,
    description="Envoi quotidien de prévisions à l'API loan-classifier",
    schedule="0 10 * * *",  # TODO S17-8 : tous les jours à 10h
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "esgi"],
) as dag:

    send = PythonOperator(
        task_id="send_predictions",
        python_callable=task_send_predictions,
    )

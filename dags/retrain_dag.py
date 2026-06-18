from __future__ import annotations

import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

QUALITY_THRESHOLD = 0.70

default_args = {
    "owner": "ayman",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# TODO S17-1 : préparation des données
def task_prepare_data(**context):
    import pandas as pd

    data_path = "/opt/airflow/data/loan_approval_dataset.csv"
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["loan_id"], errors="ignore")
    n_rows = len(df)
    print(f"Données chargées : {n_rows} lignes, {df.shape[1]} colonnes")
    assert n_rows > 0, "Dataset vide !"
    return n_rows


# TODO S17-2 : entraînement + XCom
def task_train(**context):
    sys.path.insert(0, "/opt/airflow")
    from mlproject.train import train

    metrics = train()
    context["ti"].xcom_push(key="f1", value=metrics["f1"])
    print(f"Entraînement terminé — f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")
    return metrics


# TODO S17-3 : contrôle qualité
def task_check_quality(**context):
    f1 = context["ti"].xcom_pull(task_ids="train", key="f1")
    if f1 < QUALITY_THRESHOLD:
        raise ValueError(f"f1={f1:.3f} < seuil {QUALITY_THRESHOLD}")
    print(f"Qualité OK : f1={f1:.3f}")


with DAG(
    dag_id="model_retraining",
    default_args=default_args,
    description="Ré-entraînement planifié du modèle loan-classifier",
    schedule="0 3 * * 1",  # TODO S17-4 : tous les lundis à 3h
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "esgi"],
) as dag:

    prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=task_prepare_data,
    )

    train_task = PythonOperator(
        task_id="train",
        python_callable=task_train,
    )

    check = PythonOperator(
        task_id="check_quality",
        python_callable=task_check_quality,
    )

    # TODO S17-5 : ordre d'exécution
    prepare >> train_task >> check

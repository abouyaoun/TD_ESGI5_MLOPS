"""Frontend Streamlit : tester l'API de classification Loan Approval.

Seance 14 bis - TP Streamlit
    Application qui appelle l'API FastAPI (TP S12).
    Lancement : `streamlit run frontend/app.py`

L'URL de l'API est lue depuis la variable d'environnement API_URL
(utile en docker compose, ou l'API est joignable via le nom de service `api`).
"""
from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Loan Approval Classifier", layout="wide")
st.title("Loan Approval — Demonstrateur de classification")

api_url = st.text_input("URL de l'API", value=API_URL)

predict_tab, history_tab = st.tabs(["Prediction", "Historique"])

with predict_tab:
    st.subheader("Tester l'endpoint /predict")

    with st.form("predict_form"):
        st.markdown("#### Informations personnelles")
        col1, col2 = st.columns(2)

        with col1:
            # S14bis-1 : champs numeriques
            no_of_dependents = st.number_input(
                "Nombre de dependants", min_value=0, max_value=20, value=2, step=1
            )
            income_annum = st.number_input(
                "Revenu annuel (₹)", min_value=0, value=5_000_000, step=100_000
            )
            loan_amount = st.number_input(
                "Montant du pret (₹)", min_value=0, value=12_000_000, step=500_000
            )
            loan_term = st.number_input(
                "Duree du pret (annees)", min_value=1, max_value=30, value=10, step=1
            )
            cibil_score = st.number_input(
                "Score CIBIL", min_value=300, max_value=900, value=750, step=10
            )

        with col2:
            residential_assets_value = st.number_input(
                "Valeur immobilier residentiel (₹)", min_value=0, value=8_000_000, step=500_000
            )
            commercial_assets_value = st.number_input(
                "Valeur immobilier commercial (₹)", min_value=0, value=2_000_000, step=500_000
            )
            luxury_assets_value = st.number_input(
                "Valeur actifs de luxe (₹)", min_value=0, value=5_000_000, step=500_000
            )
            bank_asset_value = st.number_input(
                "Valeur actifs bancaires (₹)", min_value=0, value=3_000_000, step=500_000
            )

        st.markdown("#### Profil")
        col3, col4 = st.columns(2)
        with col3:
            # S14bis-1 : champs categoriels
            education = st.selectbox("Niveau d'education", ["Graduate", "Not Graduate"])
        with col4:
            self_employed = st.selectbox("Independant", ["No", "Yes"])

        submitted = st.form_submit_button("Predire", use_container_width=True)

    if submitted:
        # S14bis-2 : payload avec les memes cles que le schema Features de l'API
        payload = {
            "no_of_dependents": no_of_dependents,
            "education": education,
            "self_employed": self_employed,
            "income_annum": float(income_annum),
            "loan_amount": float(loan_amount),
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": float(residential_assets_value),
            "commercial_assets_value": float(commercial_assets_value),
            "luxury_assets_value": float(luxury_assets_value),
            "bank_asset_value": float(bank_asset_value),
        }
        try:
            response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as exc:
            st.error(f"Appel a l'API impossible : {exc}")
        else:
            # S14bis-3 : affichage lisible du resultat
            st.divider()
            st.subheader("Resultat de la prediction")

            approved = result["prediction"] == 1
            label = result["label"]
            proba = result["probability"]

            if approved:
                st.success(f"Pret **{label}**")
            else:
                st.error(f"Pret **{label}**")

            col1, col2 = st.columns(2)
            col1.metric("Classe predite", label)
            col2.metric("Probabilite d'approbation", f"{proba:.1%}")
            st.progress(proba)

with history_tab:
    st.subheader("Historique des previsions")
    # S14bis-4 bonus : appel GET /predictions si l'endpoint existe
    if st.button("Rafraichir"):
        try:
            rows = httpx.get(f"{api_url}/predictions", timeout=10.0)
            rows.raise_for_status()
            st.dataframe(pd.DataFrame(rows.json()), use_container_width=True)
        except httpx.HTTPError:
            st.info("Aucun journal de previsions : ajoutez un endpoint /predictions a l'API (bonus).")
    else:
        st.info("Aucun journal de previsions : ajoutez un endpoint /predictions a l'API (bonus).")

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Loan Approval Classifier",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* Header auteur */
    .author-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 28px 40px;
        margin-bottom: 32px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .author-name {
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 2px;
        margin: 0;
    }
    .author-badge {
        background: #e94560;
        color: white;
        padding: 6px 18px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .app-subtitle {
        color: #a0aec0;
        font-size: 1rem;
        margin-top: 4px;
    }

    /* Cartes de stats */
    .metric-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #334155;
    }

    /* Résultat approved */
    .result-approved {
        background: linear-gradient(135deg, #065f46, #047857);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(4, 120, 87, 0.4);
    }
    .result-rejected {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(153, 27, 27, 0.4);
    }

    /* Section formulaire */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #334155;
    }

    /* Bouton principal */
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #e94560, #c62a47) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        padding: 14px !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
    }

    /* URL bar */
    .api-bar {
        background: #0f172a;
        border-radius: 8px;
        padding: 10px 16px;
        font-family: monospace;
        color: #38bdf8;
        font-size: 0.85rem;
        border: 1px solid #1e40af;
        margin-bottom: 24px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Header auteur ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="author-banner">
    <div>
        <p class="author-name">🎓 Ayman BOUAYOUN</p>
        <p class="app-subtitle">MLOps · ESGI 5 · Déploiement VPS · 2026</p>
    </div>
    <div>
        <span class="author-badge">MLOPS TP</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("## 🏦 Loan Approval Classifier")
st.markdown("Système de classification automatique des demandes de prêt bancaire, déployé via Docker sur Oracle Cloud.")

st.divider()

# ── Config API ─────────────────────────────────────────────────────────────────
with st.expander("⚙️ Configuration de l'API", expanded=False):
    api_url = st.text_input("URL de l'API", value=API_URL, label_visibility="collapsed")
    st.markdown(f'<div class="api-bar">🔗 {api_url}/docs</div>', unsafe_allow_html=True)

    col_health1, col_health2 = st.columns([1, 3])
    with col_health1:
        if st.button("Tester la connexion"):
            try:
                r = httpx.get(f"{api_url}/health", timeout=5.0)
                if r.status_code == 200:
                    st.success("API opérationnelle")
                else:
                    st.warning(f"Statut : {r.status_code}")
            except Exception:
                st.error("API inaccessible")
else:
    api_url = API_URL

predict_tab, history_tab, info_tab = st.tabs(["🔮 Prédiction", "📋 Historique", "ℹ️ À propos"])

# ── Onglet Prédiction ──────────────────────────────────────────────────────────
with predict_tab:
    with st.form("predict_form"):

        st.markdown('<p class="section-title">👤 Profil personnel</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            no_of_dependents = st.number_input("Nombre de dépendants", min_value=0, max_value=20, value=2, step=1)
        with col2:
            education = st.selectbox("Niveau d'éducation", ["Graduate", "Not Graduate"])
        with col3:
            self_employed = st.selectbox("Travailleur indépendant", ["No", "Yes"])

        st.markdown('<p class="section-title">💰 Situation financière</p>', unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3)
        with col4:
            income_annum = st.number_input("Revenu annuel (₹)", min_value=0, value=5_000_000, step=100_000, format="%d")
        with col5:
            loan_amount = st.number_input("Montant du prêt (₹)", min_value=0, value=12_000_000, step=500_000, format="%d")
        with col6:
            loan_term = st.number_input("Durée du prêt (années)", min_value=1, max_value=30, value=10, step=1)

        st.markdown('<p class="section-title">📊 Score & Actifs</p>', unsafe_allow_html=True)
        col7, col8 = st.columns(2)
        with col7:
            cibil_score = st.slider("Score CIBIL", min_value=300, max_value=900, value=750, step=10,
                                    help="Score de crédit : 300 (mauvais) → 900 (excellent)")
        with col8:
            st.markdown(f"""
            **Interprétation du score CIBIL**
            - 🔴 300–549 : Mauvais
            - 🟠 550–649 : Passable
            - 🟡 650–749 : Bon
            - 🟢 750–900 : Excellent
            """)

        col9, col10, col11, col12 = st.columns(4)
        with col9:
            residential_assets_value = st.number_input("Immobilier résidentiel (₹)", min_value=0, value=8_000_000, step=500_000, format="%d")
        with col10:
            commercial_assets_value = st.number_input("Immobilier commercial (₹)", min_value=0, value=2_000_000, step=500_000, format="%d")
        with col11:
            luxury_assets_value = st.number_input("Actifs de luxe (₹)", min_value=0, value=5_000_000, step=500_000, format="%d")
        with col12:
            bank_asset_value = st.number_input("Actifs bancaires (₹)", min_value=0, value=3_000_000, step=500_000, format="%d")

        st.markdown("")
        submitted = st.form_submit_button("🔮 Lancer la prédiction", use_container_width=True)

    if submitted:
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

        with st.spinner("Analyse en cours..."):
            try:
                response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
                response.raise_for_status()
                result = response.json()
            except httpx.HTTPError as exc:
                st.error(f"Erreur de connexion à l'API : {exc}")
                st.stop()

        approved = result["prediction"] == 1
        label = result["label"]
        proba = result["probability"]

        st.divider()
        st.markdown("### Résultat de l'analyse")

        if approved:
            st.markdown(f'<div class="result-approved">✅ Prêt {label}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-rejected">❌ Prêt {label}</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Décision", label)
        m2.metric("Probabilité d'approbation", f"{proba:.1%}")
        m3.metric("Score CIBIL", cibil_score)
        m4.metric("Ratio prêt/revenu", f"{loan_amount / income_annum:.1f}x" if income_annum > 0 else "N/A")

        st.markdown("**Probabilité d'approbation**")
        st.progress(proba)

        total_assets = residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value
        st.markdown("**Récapitulatif du dossier**")
        recap_col1, recap_col2 = st.columns(2)
        with recap_col1:
            st.markdown(f"""
            | Critère | Valeur |
            |---|---|
            | Revenu annuel | ₹{income_annum:,.0f} |
            | Montant demandé | ₹{loan_amount:,.0f} |
            | Durée | {loan_term} ans |
            | Score CIBIL | {cibil_score} |
            """)
        with recap_col2:
            st.markdown(f"""
            | Critère | Valeur |
            |---|---|
            | Total actifs | ₹{total_assets:,.0f} |
            | Éducation | {education} |
            | Indépendant | {self_employed} |
            | Dépendants | {no_of_dependents} |
            """)

# ── Onglet Historique ──────────────────────────────────────────────────────────
with history_tab:
    st.markdown("### 📋 Historique des prédictions")
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        refresh = st.button("🔄 Rafraîchir", use_container_width=True)

    if refresh:
        try:
            rows = httpx.get(f"{api_url}/predictions", timeout=10.0)
            rows.raise_for_status()
            data = rows.json()
            if data:
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True, height=400)
                st.markdown(f"**{len(df)} prédiction(s) enregistrée(s)**")
            else:
                st.info("Aucune prédiction enregistrée pour le moment.")
        except httpx.HTTPError:
            st.info("L'endpoint `/predictions` n'est pas disponible sur cette API.")
    else:
        st.info("Cliquez sur **Rafraîchir** pour charger l'historique.")

# ── Onglet À propos ────────────────────────────────────────────────────────────
with info_tab:
    st.markdown("### ℹ️ À propos de ce projet")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Projet MLOps — ESGI 5**

        Ce projet illustre un pipeline MLOps complet :
        - Entraînement d'un modèle de classification (Logistic Regression)
        - Exposition via une API REST FastAPI
        - Interface utilisateur Streamlit
        - Déploiement Docker sur Oracle Cloud VPS
        - CI/CD via GitHub Actions
        """)
    with col_b:
        st.markdown("""
        **Stack technique**

        | Composant | Technologie |
        |---|---|
        | Modèle | Scikit-learn |
        | API | FastAPI + Uvicorn |
        | Frontend | Streamlit |
        | Containerisation | Docker Compose |
        | Cloud | Oracle Cloud (Always Free) |
        | CI/CD | GitHub Actions |
        """)

    st.divider()
    st.markdown("""
    <div style="text-align:center; color:#64748b; font-size:0.85rem; margin-top:20px;">
        Réalisé par <strong>Ayman BOUAYOUN</strong> · ESGI 5 · MLOps 2026
    </div>
    """, unsafe_allow_html=True)

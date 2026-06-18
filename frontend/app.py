from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
AIRFLOW_INTERNAL = os.environ.get("AIRFLOW_INTERNAL", "http://airflow:8080")
VPS_IP = os.environ.get("VPS_IP", "")

PUBLIC_API_URL = f"http://{VPS_IP}:8000" if VPS_IP else API_URL
PUBLIC_AIRFLOW_URL = f"http://{VPS_IP}:8080" if VPS_IP else "http://localhost:8080"
PUBLIC_MLFLOW_URL = f"http://{VPS_IP}:5000" if VPS_IP else "http://localhost:5000"

st.set_page_config(
    page_title="Loan Approval Classifier",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .author-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 28px 40px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .author-name {
        font-size: 2.2rem;
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
    .app-subtitle { color: #a0aec0; font-size: 1rem; margin-top: 4px; }
    .result-approved {
        background: linear-gradient(135deg, #065f46, #047857);
        border-radius: 16px; padding: 30px; text-align: center;
        color: white; font-size: 1.8rem; font-weight: 700; margin: 20px 0;
        box-shadow: 0 4px 20px rgba(4, 120, 87, 0.4);
    }
    .result-rejected {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border-radius: 16px; padding: 30px; text-align: center;
        color: white; font-size: 1.8rem; font-weight: 700; margin: 20px 0;
        box-shadow: 0 4px 20px rgba(153, 27, 27, 0.4);
    }
    .section-title {
        font-size: 1rem; font-weight: 700; color: #94a3b8;
        text-transform: uppercase; letter-spacing: 1.5px;
        margin: 20px 0 10px 0; padding-bottom: 6px;
        border-bottom: 2px solid #334155;
    }
    .service-card {
        background: #1e293b; border-radius: 12px; padding: 16px;
        border: 1px solid #334155; margin-bottom: 10px;
    }
    .status-dot-green { color: #22c55e; font-size: 1.2rem; }
    .status-dot-red { color: #ef4444; font-size: 1.2rem; }
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #e94560, #c62a47) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-size: 1.1rem !important;
        font-weight: 700 !important; padding: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Navigation")
    st.divider()

    st.markdown("### 🔗 Services déployés")

    def check_service(url: str) -> bool:
        try:
            r = httpx.get(url, timeout=3.0)
            return r.status_code < 500
        except Exception:
            return False

    api_ok = check_service(f"{API_URL}/health")
    airflow_ok = check_service(f"{AIRFLOW_INTERNAL}/health")

    api_dot = "🟢" if api_ok else "🔴"
    airflow_dot = "🟢" if airflow_ok else "🔴"

    st.markdown(f"""
    {api_dot} **API FastAPI**
    [📖 Documentation]({PUBLIC_API_URL}/docs) · [❤️ Health]({PUBLIC_API_URL}/health)

    🟡 **MLflow UI**
    [📊 Expériences]({PUBLIC_MLFLOW_URL})

    {airflow_dot} **Airflow**
    [⚙️ Orchestration]({PUBLIC_AIRFLOW_URL})
    """)

    st.divider()
    st.markdown("### 📌 Infos modèle")
    st.markdown("""
    - **Algorithme** : Logistic Regression
    - **Cible** : Approbation prêt
    - **Seuil qualité** : F1 ≥ 0.70
    - **Version** : 1.0
    """)

    st.divider()
    st.caption("Ayman BOUAYOUN · ESGI 5 · MLOps 2026")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="author-banner">
    <div>
        <p class="author-name">🎓 Ayman BOUAYOUN</p>
        <p class="app-subtitle">MLOps · ESGI 5 · Déploiement VPS Oracle Cloud · 2026</p>
    </div>
    <span class="author-badge">MLOPS TP</span>
</div>
""", unsafe_allow_html=True)

predict_tab, history_tab, infra_tab, pipeline_tab, about_tab = st.tabs([
    "🔮 Prédiction", "📋 Historique", "🏗️ Infrastructure", "🔄 Pipeline MLOps", "ℹ️ À propos"
])

# ── Onglet Prédiction ──────────────────────────────────────────────────────────
with predict_tab:
    st.markdown("### 🔮 Simuler une demande de prêt")
    st.caption("Remplissez le formulaire pour obtenir une prédiction instantanée via l'API.")

    with st.form("predict_form"):
        st.markdown('<p class="section-title">👤 Profil personnel</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            no_of_dependents = st.number_input("Nombre de dépendants", min_value=0, max_value=20, value=2, step=1)
        with c2:
            education = st.selectbox("Niveau d'éducation", ["Graduate", "Not Graduate"])
        with c3:
            self_employed = st.selectbox("Travailleur indépendant", ["No", "Yes"])

        st.markdown('<p class="section-title">💰 Situation financière</p>', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            income_annum = st.number_input("Revenu annuel (₹)", min_value=0, value=5_000_000, step=100_000, format="%d")
        with c5:
            loan_amount = st.number_input("Montant du prêt (₹)", min_value=0, value=12_000_000, step=500_000, format="%d")
        with c6:
            loan_term = st.number_input("Durée du prêt (années)", min_value=1, max_value=30, value=10, step=1)

        st.markdown('<p class="section-title">📊 Score de crédit & Actifs</p>', unsafe_allow_html=True)
        c7, c8 = st.columns([2, 1])
        with c7:
            cibil_score = st.slider("Score CIBIL", min_value=300, max_value=900, value=750, step=10)
        with c8:
            if cibil_score < 550:
                st.error("🔴 Mauvais (300–549)")
            elif cibil_score < 650:
                st.warning("🟠 Passable (550–649)")
            elif cibil_score < 750:
                st.info("🟡 Bon (650–749)")
            else:
                st.success("🟢 Excellent (750–900)")

        c9, c10, c11, c12 = st.columns(4)
        with c9:
            residential_assets_value = st.number_input("Immobilier résidentiel (₹)", min_value=0, value=8_000_000, step=500_000, format="%d")
        with c10:
            commercial_assets_value = st.number_input("Immobilier commercial (₹)", min_value=0, value=2_000_000, step=500_000, format="%d")
        with c11:
            luxury_assets_value = st.number_input("Actifs de luxe (₹)", min_value=0, value=5_000_000, step=500_000, format="%d")
        with c12:
            bank_asset_value = st.number_input("Actifs bancaires (₹)", min_value=0, value=3_000_000, step=500_000, format="%d")

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
                response = httpx.post(f"{API_URL}/predict", json=payload, timeout=10.0)
                response.raise_for_status()
                result = response.json()
            except httpx.HTTPError as exc:
                st.error(f"Erreur API : {exc}")
                st.stop()

        approved = result["prediction"] == 1
        label = result["label"]
        proba = result["probability"]
        total_assets = residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value

        st.divider()
        if approved:
            st.markdown(f'<div class="result-approved">✅ Prêt {label}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-rejected">❌ Prêt {label}</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Décision", label)
        m2.metric("Probabilité d'approbation", f"{proba:.1%}")
        m3.metric("Score CIBIL", cibil_score)
        m4.metric("Ratio prêt/revenu", f"{loan_amount / income_annum:.1f}x" if income_annum > 0 else "N/A")

        col_chart, col_recap = st.columns([1, 1])

        with col_chart:
            st.markdown("**Probabilité (Approuvé vs Refusé)**")
            chart_df = pd.DataFrame({
                "Probabilité": [proba, 1 - proba]
            }, index=["Approuvé", "Refusé"])
            st.bar_chart(chart_df, color=["#22c55e"])

        with col_recap:
            st.markdown("**Récapitulatif du dossier**")
            st.dataframe(pd.DataFrame({
                "Critère": ["Revenu annuel", "Montant demandé", "Durée", "Score CIBIL", "Total actifs", "Éducation", "Indépendant"],
                "Valeur": [
                    f"₹{income_annum:,.0f}",
                    f"₹{loan_amount:,.0f}",
                    f"{loan_term} ans",
                    str(cibil_score),
                    f"₹{total_assets:,.0f}",
                    education,
                    self_employed,
                ]
            }), use_container_width=True, hide_index=True)

# ── Onglet Historique ──────────────────────────────────────────────────────────
with history_tab:
    st.markdown("### 📋 Historique des prédictions")
    st.caption("Toutes les prédictions effectuées via l'API ou le DAG Airflow.")

    col_btn, col_info, _ = st.columns([1, 2, 3])
    with col_btn:
        refresh = st.button("🔄 Rafraîchir", use_container_width=True)

    if refresh:
        try:
            rows = httpx.get(f"{API_URL}/predictions", timeout=10.0)
            rows.raise_for_status()
            data = rows.json()
            if data:
                df = pd.DataFrame(data)
                total = len(df)

                s1, s2, s3 = st.columns(3)
                s1.metric("Total prédictions", total)
                if "prediction" in df.columns:
                    approved_count = (df["prediction"] == 1).sum()
                    s2.metric("Approuvés", approved_count)
                    s3.metric("Refusés", total - approved_count)

                st.dataframe(df, use_container_width=True, height=400)
            else:
                st.info("Aucune prédiction enregistrée.")
        except httpx.HTTPError:
            st.info("L'endpoint `/predictions` n'est pas disponible sur cette API.")
    else:
        st.info("Cliquez sur **Rafraîchir** pour charger l'historique.")

# ── Onglet Infrastructure ──────────────────────────────────────────────────────
with infra_tab:
    st.markdown("### 🏗️ État de l'infrastructure")
    st.caption("Statut en temps réel des services déployés sur Oracle Cloud VPS.")

    if st.button("🔄 Actualiser les statuts"):
        st.rerun()

    services = [
        {
            "name": "API FastAPI",
            "icon": "⚡",
            "desc": "Endpoint de prédiction REST",
            "check_url": f"{API_URL}/health",
            "public_url": PUBLIC_API_URL,
            "port": 8000,
            "docs": f"{PUBLIC_API_URL}/docs",
        },
        {
            "name": "Airflow",
            "icon": "⚙️",
            "desc": "Orchestration des DAGs MLOps",
            "check_url": f"{AIRFLOW_INTERNAL}/health",
            "public_url": PUBLIC_AIRFLOW_URL,
            "port": 8080,
            "docs": None,
        },
        {
            "name": "MLflow",
            "icon": "📊",
            "desc": "Suivi des expériences ML",
            "check_url": f"{PUBLIC_MLFLOW_URL}/health",
            "public_url": PUBLIC_MLFLOW_URL,
            "port": 5000,
            "docs": None,
        },
        {
            "name": "Frontend Streamlit",
            "icon": "🖥️",
            "desc": "Interface utilisateur (cette page)",
            "check_url": None,
            "public_url": f"http://{VPS_IP}:8501" if VPS_IP else "http://localhost:8501",
            "port": 8501,
            "docs": None,
        },
    ]

    for svc in services:
        with st.container():
            col_status, col_info, col_link = st.columns([1, 4, 2])
            with col_status:
                if svc["check_url"] is None:
                    st.markdown("🟢 **UP**")
                else:
                    ok = check_service(svc["check_url"])
                    st.markdown("🟢 **UP**" if ok else "🔴 **DOWN**")
            with col_info:
                st.markdown(f"**{svc['icon']} {svc['name']}** — port `{svc['port']}`")
                st.caption(svc["desc"])
            with col_link:
                st.markdown(f"[🔗 Ouvrir]({svc['public_url']})")
                if svc.get("docs"):
                    st.markdown(f"[📖 Docs]({svc['docs']})")
            st.divider()

    st.markdown("### 🐳 Conteneurs Docker")
    st.code("""docker compose ps
docker compose logs -f api
docker compose logs -f airflow""", language="bash")

# ── Onglet Pipeline MLOps ──────────────────────────────────────────────────────
with pipeline_tab:
    st.markdown("### 🔄 Pipeline MLOps complet")

    st.markdown("""
    Ce projet implémente un pipeline MLOps de bout en bout :
    """)

    steps = [
        ("1️⃣", "Données", "Dataset Loan Approval (Kaggle) — 4 269 entrées, 12 features", "#3b82f6"),
        ("2️⃣", "Entraînement", "Logistic Regression via Scikit-learn, tracking MLflow (F1, ROC-AUC)", "#8b5cf6"),
        ("3️⃣", "Validation", "Contrôle qualité automatique — seuil F1 ≥ 0.70", "#f59e0b"),
        ("4️⃣", "API", "FastAPI + Uvicorn — endpoint /predict, /health, /predictions", "#10b981"),
        ("5️⃣", "Orchestration", "Airflow DAGs — ré-entraînement hebdomadaire + prévisions quotidiennes", "#ef4444"),
        ("6️⃣", "Déploiement", "Docker Compose sur Oracle Cloud VPS (Always Free)", "#06b6d4"),
        ("7️⃣", "CI/CD", "GitHub Actions — tests automatisés + déploiement continu", "#f97316"),
    ]

    for icon, title, desc, color in steps:
        col_icon, col_content = st.columns([1, 7])
        with col_icon:
            st.markdown(f"## {icon}")
        with col_content:
            st.markdown(f"**{title}**")
            st.caption(desc)
        st.divider()

    st.markdown("### 📁 Structure du projet")
    st.code("""TD_ESGI5_MLOPS/
├── mlproject/          # Package Python (train, api, features, data)
├── dags/               # DAGs Airflow
│   ├── retrain_dag.py      # Ré-entraînement hebdomadaire
│   └── predictions_dag.py  # Prévisions quotidiennes
├── frontend/           # Interface Streamlit
├── data/               # Dataset CSV
├── models/             # Modèle entraîné (joblib)
├── Dockerfile          # Image API
├── Dockerfile.frontend # Image Frontend
├── Dockerfile.airflow  # Image Airflow
├── docker-compose.yml  # Stack complète
└── .github/workflows/  # CI/CD GitHub Actions""", language="text")

# ── Onglet À propos ────────────────────────────────────────────────────────────
with about_tab:
    st.markdown("### ℹ️ À propos du projet")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Projet MLOps — ESGI 5**

        Réalisation d'un pipeline MLOps complet pour la prédiction d'approbation
        de prêts bancaires.

        **Objectifs pédagogiques :**
        - Tracking d'expériences avec MLflow
        - Exposition d'un modèle via FastAPI
        - Orchestration avec Apache Airflow
        - Containerisation Docker
        - Déploiement cloud sur VPS Oracle
        - CI/CD avec GitHub Actions
        """)
    with col_b:
        st.markdown("""
        **Stack technique**

        | Composant | Technologie |
        |---|---|
        | Modèle | Scikit-learn (LogReg) |
        | Tracking | MLflow |
        | API | FastAPI + Uvicorn |
        | Orchestration | Apache Airflow |
        | Frontend | Streamlit |
        | Containers | Docker Compose |
        | Cloud | Oracle Cloud Always Free |
        | CI/CD | GitHub Actions |
        | Python | 3.11 |
        """)

    st.divider()
    st.markdown("""
    <div style="text-align:center; padding: 20px; background: #1e293b; border-radius: 12px;">
        <p style="font-size:1.3rem; font-weight:700; color:white; margin:0;">Ayman BOUAYOUN</p>
        <p style="color:#94a3b8; margin:4px 0;">ESGI 5 · Filière IA & Big Data · Promotion 2026</p>
        <p style="color:#64748b; font-size:0.85rem;">Projet MLOps · Déploiement VPS Oracle Cloud</p>
    </div>
    """, unsafe_allow_html=True)

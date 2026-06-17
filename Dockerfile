# ── Stage 1 : entraînement du modèle ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml ./
COPY mlproject/ ./mlproject/

RUN pip install --no-cache-dir -e "."

RUN mkdir -p models && python -c "\
import numpy as np, pandas as pd, joblib; \
from sklearn.linear_model import LogisticRegression; \
from sklearn.pipeline import Pipeline; \
from mlproject.features import build_preprocessor; \
rng = np.random.default_rng(42); n = 500; \
df = pd.DataFrame({ \
    'no_of_dependents': rng.integers(0, 6, n), \
    'education': rng.choice(['Graduate', 'Not Graduate'], n), \
    'self_employed': rng.choice(['Yes', 'No'], n), \
    'income_annum': rng.integers(200000, 9900000, n), \
    'loan_amount': rng.integers(300000, 39500000, n), \
    'loan_term': rng.integers(2, 20, n), \
    'cibil_score': rng.integers(300, 900, n), \
    'residential_assets_value': rng.integers(100000, 29000000, n), \
    'commercial_assets_value': rng.integers(0, 19700000, n), \
    'luxury_assets_value': rng.integers(300000, 39200000, n), \
    'bank_asset_value': rng.integers(0, 14900000, n), \
    'loan_status': rng.integers(0, 2, n), \
}); \
X = df.drop(columns=['loan_status']); y = df['loan_status']; \
model = Pipeline([('pre', build_preprocessor()), ('clf', LogisticRegression(max_iter=1000))]); \
model.fit(X, y); \
joblib.dump(model, 'models/model.joblib') \
"

# ── Stage 2 : image de production ────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY mlproject/ ./mlproject/

RUN pip install --no-cache-dir -e "."

COPY --from=builder /build/models/model.joblib ./models/model.joblib

ENV MODEL_PATH=models/model.joblib
ENV MODEL_NAME=loan-classifier
ENV MODEL_VERSION=latest

EXPOSE 8000

CMD ["uvicorn", "mlproject.api:app", "--host", "0.0.0.0", "--port", "8000"]

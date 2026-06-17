FROM python:3.11-slim

WORKDIR /app

# Copie uniquement ce qui est nécessaire au build des dépendances
COPY pyproject.toml ./
COPY mlproject/ ./mlproject/

# Installation sans les extras dev
RUN pip install --no-cache-dir -e "." \
    && pip install --no-cache-dir uvicorn[standard]

# Copie du modèle pré-entraîné s'il existe
COPY models/ ./models/

ENV MODEL_PATH=models/model.joblib
ENV MODEL_NAME=loan-classifier
ENV MODEL_VERSION=latest

EXPOSE 8000

CMD ["uvicorn", "mlproject.api:app", "--host", "0.0.0.0", "--port", "8000"]

PYTHON     ?= python3
UV         ?= uv
VENV       := .venv
PYTHONPATH := .

# Hyperparametres train.py
C        ?= 1.0
MAX_ITER ?= 1000

# Hyperparametres train_models.py / train_optuna.py
CV       ?= 5
SCORING  ?= roc_auc
N_TRIALS ?= 30

# Ports
MLFLOW_PORT  ?= 5001
API_PORT     ?= 8000

.PHONY: help install sync lock reset-env doctor \
        mlflow train train-models train-optuna evaluate \
        api frontend \
        lint format type check test

# ─── Aide ───────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  install        Creer le venv et installer les dependances"
	@echo "  sync           Synchroniser le venv avec pyproject.toml"
	@echo "  mlflow         Demarrer le serveur MLflow (port $(MLFLOW_PORT))"
	@echo "  train          Entrainer le baseline LogReg (S5)"
	@echo "  train-models   Comparer RF/XGB/LGBM par GridSearchCV (S7)"
	@echo "  train-optuna   Optimisation Optuna (S6)"
	@echo "  evaluate       Evaluer le modele du registry (S11)"
	@echo "  api            Lancer l'API FastAPI (S12)"
	@echo "  frontend       Lancer l'interface Streamlit"
	@echo "  lint           Verifier le style (ruff)"
	@echo "  format         Formater le code (ruff)"
	@echo "  type           Verifier les types (mypy)"
	@echo "  test           Lancer les tests (pytest)"
	@echo "  check          lint + type + test"
	@echo ""

# ─── Environnement ──────────────────────────────────────────────────────────

install:
	$(UV) venv $(VENV)
	$(UV) pip install -e ".[dev]"

sync:
	$(UV) pip install -e ".[dev]"

lock:
	$(UV) pip compile pyproject.toml -o requirements.lock

reset-env:
	rm -rf $(VENV)
	$(MAKE) install

doctor:
	@echo "Python  : $(shell $(PYTHON) --version)"
	@echo "MLflow  : $(shell PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c 'import mlflow; print(mlflow.__version__)' 2>/dev/null || echo 'non installe')"
	@echo "sklearn : $(shell PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c 'import sklearn; print(sklearn.__version__)' 2>/dev/null || echo 'non installe')"
	@echo "Venv    : $(VENV)"

# ─── MLflow ────────────────────────────────────────────────────────────────

mlflow:
	PYTHONPATH=$(PYTHONPATH) mlflow server \
		--host 0.0.0.0 \
		--port $(MLFLOW_PORT) \
		--backend-store-uri sqlite:///mlruns/mlflow.db \
		--default-artifact-root ./mlartifacts

# ─── Entrainement ───────────────────────────────────────────────────────────

train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m mlproject.train \
		--c $(C) --max-iter $(MAX_ITER)

train-models:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m mlproject.train_models \
		--cv $(CV) --scoring $(SCORING)

train-optuna:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m mlproject.train_optuna \
		--n-trials $(N_TRIALS) --cv $(CV)

evaluate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m mlproject.evaluate

# ─── Serving ────────────────────────────────────────────────────────────────

api:
	PYTHONPATH=$(PYTHONPATH) uvicorn mlproject.api:app \
		--host 0.0.0.0 --port $(API_PORT) --reload

frontend:
	PYTHONPATH=$(PYTHONPATH) streamlit run frontend/app.py

# ─── Qualite ────────────────────────────────────────────────────────────────

lint:
	ruff check mlproject/

format:
	ruff format mlproject/

type:
	mypy mlproject/

test:
	PYTHONPATH=$(PYTHONPATH) pytest tests/ -v --tb=short

check: lint type test

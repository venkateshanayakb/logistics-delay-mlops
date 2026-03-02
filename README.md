# 📦 Logistics Delay Prediction — MLOps Pipeline

[![Backend CI](https://github.com/venkateshanayakb/logistics-delay-mlops/actions/workflows/backend.yml/badge.svg)](https://github.com/venkateshanayakb/logistics-delay-mlops/actions/workflows/backend.yml)
[![Frontend CI](https://github.com/venkateshanayakb/logistics-delay-mlops/actions/workflows/frontend.yml/badge.svg)](https://github.com/venkateshanayakb/logistics-delay-mlops/actions/workflows/frontend.yml)

> End-to-end ML classification system that predicts whether a logistics shipment will arrive **Early**, **On-time**, or **Late**. Built with modern MLOps best practices — from data versioning to monitoring and deployment.

---

## 🏗️ Pipeline Architecture

```mermaid
graph LR
    A[Neon Postgres<br>Raw Data] --> B[Preprocessing<br>ColumnTransformer]
    B --> C[Training Pipeline<br>RandomizedSearch + BayesSearch]
    C --> D[W&B Experiment<br>Tracking]
    C --> E[Best Model<br>.joblib Artifact]
    E --> F[FastAPI Backend<br>/predict /health /metrics]
    F --> G[Streamlit<br>Frontend]
    F --> H[Prometheus<br>+ Grafana]
    I[GitHub Actions<br>CI/CD] --> F
    I --> G
    F --> J[Render<br>Deployment]
    G --> J
```

---

## 📊 Model Performance

**Best Model: LightGBM** (with `class_weight="balanced"`)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Early | 0.42 | 0.81 | 0.55 | 709 |
| On-time | 0.31 | 0.27 | 0.29 | 606 |
| Late | 0.85 | 0.57 | 0.68 | 1,795 |
| **Macro avg** | **0.53** | **0.55** | **0.51** | 3,110 |
| **Weighted avg** | **0.64** | **0.57** | **0.57** | 3,110 |

### Best Hyperparameters

Found via 2-phase tuning: **RandomizedSearchCV** (15 iterations, 3-fold CV) → **BayesSearchCV** (5 iterations, fine-tuning):

| Parameter | Value |
|-----------|-------|
| `n_estimators` | ~400 |
| `num_leaves` | ~80 |
| `learning_rate` | ~0.08 |
| `class_weight` | balanced |
| `subsample` | ~0.85 |
| `colsample_bytree` | ~0.75 |

### Additional Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 0.57 |
| F1 (macro) | 0.51 |
| ROC-AUC (macro, OVR) | ~0.73 |
| Training time | ~5 min |

---

## 🗂️ Project Structure

```
logistics-delay-mlops/
├── .github/workflows/
│   ├── backend.yml          # Backend CI: lint → test → deploy
│   └── frontend.yml         # Frontend CI: lint → deploy
├── api/
│   ├── main.py              # FastAPI app (POST /predict, GET /health, GET /metrics)
│   ├── model_service.py     # Model loading, inference, SHAP explainability
│   └── schemas.py           # Pydantic request/response models
├── data/
│   └── Logistics Delay.csv  # Raw dataset
├── frontend/
│   └── app.py               # Streamlit interactive UI
├── models/
│   └── best_pipeline.joblib # Saved preprocessor + model artifact
├── monitoring/
│   ├── docker-compose.yml   # Prometheus + Grafana stack
│   ├── prometheus.yml       # Prometheus scrape config
│   └── grafana/             # Dashboard provisioning + JSON
├── report/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── shap_summary.png
│   └── shap_bar.png
├── src/
│   ├── data_loader.py       # Load data from Neon Postgres
│   ├── preprocessing.py     # Feature engineering + ColumnTransformer
│   ├── train.py             # Training pipeline (RandomizedSearch + BayesSearch)
│   └── evaluate.py          # Evaluation plots, SHAP, W&B logging
├── tests/
│   ├── conftest.py           # Shared fixtures
│   ├── test_api.py           # API endpoint tests
│   ├── test_data_validation.py
│   ├── test_integration.py
│   ├── test_model.py         # Model service tests
│   └── test_preprocessing.py
├── Dockerfile               # Multi-stage Docker build for FastAPI
├── requirements.txt
├── lint_scores.txt          # Flake8 (0 errors) + Pylint (9.11/10)
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11+
- Neon Postgres account (for data layer)
- W&B account (for experiment tracking)

### Installation

```bash
git clone https://github.com/venkateshanayakb/logistics-delay-mlops.git
cd logistics-delay-mlops
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and fill in:

```env
DATABASE_URL=postgresql://user:pass@host/db
WANDB_API_KEY=your_key
WANDB_PROJECT=logistics-delay-mlops
```

---

## 🚀 Usage

### 1. Data Layer

```bash
# Load data from Neon Postgres
python -m src.data_loader

# Or use local CSV
python -m src.train --csv
```

### 2. Train Model

```bash
# Full training with W&B logging
python -m src.train --csv

# Quick smoke test (no W&B)
python -m src.train --csv --quick --no-wandb
```

### 3. Evaluate Model

```bash
python -m src.evaluate --csv --no-wandb
```

### 4. Run FastAPI Backend

```bash
uvicorn api.main:app --reload --port 8000
```

Endpoints:
- `POST /predict` — JSON payload → prediction with SHAP features
- `GET /health` — Model status and uptime
- `GET /metrics` — Prometheus-compatible metrics
- `GET /docs` — Interactive Swagger UI

### 5. Run Streamlit Frontend

```bash
streamlit run frontend/app.py
```

### 6. Docker + Monitoring

```bash
docker build -t logistics-api .
docker run -p 8000:8000 logistics-api

# Full stack with Prometheus + Grafana
cd monitoring
docker-compose up -d
```

---

## 🧪 Testing & Code Quality

### Tests (pytest)

```bash
pytest tests/ -v --tb=short --cov=api --cov=src
```

**5 test modules** covering:
- API endpoint validation (`test_api.py`)
- Data validation (`test_data_validation.py`)
- Model inference (`test_model.py`)
- Preprocessing pipeline (`test_preprocessing.py`)
- Integration tests (`test_integration.py`)

### Linting

```bash
flake8 api/ src/ tests/ frontend/ --max-line-length=120
pylint api src tests frontend
```

**Results:**
- **Flake8**: 0 errors, 0 warnings ✅
- **Pylint**: 9.11/10 ✅

Full scores saved in [`lint_scores.txt`](lint_scores.txt).

---

## 📈 W&B Experiment Tracking

All training experiments are tracked in **Weights & Biases**:
- Accuracy, F1 (macro/weighted), ROC-AUC, Precision, Recall
- Hyperparameters for each model
- Model artifacts (`.joblib`)
- Classification reports and plots

W&B Project: [logistics-delay-mlops](https://wandb.ai/venkateshnayak-iihmr-bangalore/logistics-delay-mlops)

---

## 📊 Monitoring (Prometheus + Grafana)

Three Grafana dashboards:
1. **Request Count** — Total predictions served, by class
2. **Prediction Latency** — P50, P95, P99 response times
3. **Model Confidence** — Distribution of prediction confidence scores

Prometheus scrapes `/metrics` endpoint every 15 seconds.

---

## 🚢 Deployment

| Service | URL |
|---------|-----|
| **Backend API** | [logistics-api-yjjb.onrender.com](https://logistics-api-yjjb.onrender.com) |
| **Frontend** | [logistics-frontend-4x43.onrender.com](https://logistics-frontend-4x43.onrender.com/) |
| **API Docs (Swagger)** | [logistics-api-yjjb.onrender.com/docs](https://logistics-api-yjjb.onrender.com/docs) |

- **CI/CD**: GitHub Actions — lint → test → deploy on push to `main`

---

## 💼 Business Value

### Problem
Logistics companies lose **12-15% of order value** on average due to delayed shipments through penalties, refunds, and customer churn.

### Solution
This ML system predicts shipment delays **before** they happen, enabling:

1. **Proactive intervention** — Flag high-risk shipments for expedited handling
2. **Resource optimization** — Allocate warehouse and transport resources based on predicted delay risk
3. **Customer communication** — Notify customers early about potential delays, improving satisfaction
4. **Cost savings** — Estimated **5-8% reduction** in delay-related losses through early detection

### Key Features for Business Users
- **Real-time predictions** via REST API (< 500ms latency)
- **SHAP explainability** — understand *why* each prediction was made
- **Monitoring dashboards** — track prediction volume, confidence, and model health
- **Scalable deployment** — Docker-based, runs on any cloud provider

---

## 📋 CI/CD Workflows

### Backend ([`backend.yml`](https://github.com/venkateshanayakb/logistics-delay-mlops/actions/workflows/backend.yml))
```
Push to main → Lint (flake8) → Test (pytest) → Deploy (Render)
```

### Frontend ([`frontend.yml`](https://github.com/venkateshanayakb/logistics-delay-mlops/actions/workflows/frontend.yml))
```
Push to main → Lint (flake8) → Syntax check → Deploy (Render)
```

---

## 📝 License

This project is part of the MTA-1 assessment for Advance Analytics II.

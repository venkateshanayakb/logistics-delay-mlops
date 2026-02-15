# 🚀 Render Deployment Guide

Complete step-by-step guide to deploy the **Logistics Delay Prediction** app (FastAPI + Streamlit) on [Render](https://render.com) for free.

---

## Architecture on Render

```
┌─────────────────────────────────────────────────────┐
│                    RENDER                            │
│                                                     │
│  ┌──────────────────┐     ┌──────────────────────┐  │
│  │  logistics-api   │     │ logistics-frontend   │  │
│  │  (FastAPI)        │◄────│  (Streamlit)          │  │
│  │  /predict         │     │  port $PORT           │  │
│  │  /health          │     │                      │  │
│  │  /metrics         │     │  API_URL=https://     │  │
│  │  port $PORT       │     │  logistics-api...     │  │
│  └──────────────────┘     └──────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## Prerequisites

1. A **GitHub account** with this repo pushed
2. A **Render account** (free tier) — sign up at [render.com](https://render.com)
3. The **model file** (`models/best_pipeline.joblib`) committed to the repo

---

## Step 0: Commit the Model File

The `.gitignore` excludes `models/*.joblib`. For Render deployment, we need the model in the repo.

```bash
# Force-add the model file (overrides .gitignore)
git add -f models/best_pipeline.joblib
git commit -m "feat: add trained model for deployment"
git push origin main
```

> **Why?** Render's free tier doesn't have persistent disks. The simplest approach is to include the model (~5-20 MB) directly in the repo. For production, you'd use cloud storage (S3/GCS) instead.

---

## Step 1: Deploy via Blueprint (Recommended)

The `render.yaml` file at the repo root defines both services automatically.

1. Go to [**Render Dashboard**](https://dashboard.render.com)
2. Click **New → Blueprint**
3. Connect your **GitHub** account (if not already)
4. Select your `logistics-delay-mlops` repository
5. Render auto-detects `render.yaml` and shows 2 services:
   - `logistics-api` (FastAPI)
   - `logistics-frontend` (Streamlit)
6. Click **Apply** → Render builds and deploys both services

### What happens automatically:
- ✅ Python 3.11 environment setup
- ✅ Dependencies installed
- ✅ Health check on `/health` for the API
- ✅ Frontend `API_URL` env var linked to the API service URL
- ✅ Auto-deploy on every `git push` to `main`

---

## Step 2: Manual Deployment (Alternative)

If you prefer manual setup instead of the Blueprint:

### 2a. Deploy the API

1. **New → Web Service** on Render Dashboard
2. Connect your GitHub repo
3. Configure:

| Setting | Value |
|---------|-------|
| **Name** | `logistics-api` |
| **Region** | Oregon (US West) |
| **Runtime** | Python |
| **Build Command** | `pip install -r requirements-docker.txt` |
| **Start Command** | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
| **Plan** | Free |

4. Add **Environment Variables**:

| Key | Value |
|-----|-------|
| `MODEL_PATH` | `models/best_pipeline.joblib` |
| `PYTHON_VERSION` | `3.11.7` |

5. Click **Create Web Service**

6. Wait for the build to finish (2-3 minutes). Copy the service URL:
   ```
   https://logistics-api-XXXX.onrender.com
   ```

### 2b. Deploy the Frontend

1. **New → Web Service** again
2. Connect the **same** GitHub repo
3. Configure:

| Setting | Value |
|---------|-------|
| **Name** | `logistics-frontend` |
| **Region** | Oregon (US West) |
| **Runtime** | Python |
| **Build Command** | `pip install streamlit requests plotly` |
| **Start Command** | `streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true` |
| **Plan** | Free |

4. Add **Environment Variables**:

| Key | Value |
|-----|-------|
| `API_URL` | `https://logistics-api-XXXX.onrender.com` ← paste API URL from Step 2a |
| `PYTHON_VERSION` | `3.11.7` |

5. Click **Create Web Service**

---

## Step 3: Verify Deployment

### Check the API

Open your API URL in a browser:

```
https://logistics-api-XXXX.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_name": "tuned_rf",
  "model_loaded": true,
  "uptime_seconds": 42.5
}
```

Also check the interactive docs:
```
https://logistics-api-XXXX.onrender.com/docs
```

### Check the Frontend

Open the frontend URL:
```
https://logistics-frontend-XXXX.onrender.com
```

You should see the **Logistics Delay Predictor** UI. Try a prediction!

---

## ⚠️ Important Notes

### Free Tier Behavior
- **Cold starts**: Free services spin down after 15 minutes of inactivity. First request takes ~30-60 seconds to wake up.
- **750 free hours/month**: Shared across all free services. 2 services × 24h × 31 days = 1,488 hours, so they won't both run all month. Consider upgrading one service if needed.
- **Build time**: ~2-4 minutes per deploy.

### Troubleshooting

| Issue | Fix |
|-------|-----|
| Frontend shows "API not reachable" | The API may be cold-starting. Wait 30s and refresh. Also check `API_URL` env var is set correctly. |
| Build fails with "No module named X" | Check `requirements-docker.txt` includes the dependency |
| Model not loading | Ensure you force-committed the `.joblib` file: `git add -f models/best_pipeline.joblib` |
| Port errors | Render sets `$PORT` automatically — never hardcode a port |
| `numpy` conflict | Already fixed: we use `numpy>=1.24,<2.1` |

### Keeping Services Awake (Optional)

To prevent cold starts, use a free cron job service like [cron-job.org](https://cron-job.org) to ping your API's `/health` endpoint every 14 minutes.

---

## File Changes Made

| File | Change |
|------|--------|
| `render.yaml` | **NEW** — Render Blueprint (IaC) defining both services |
| `frontend/app.py` | **MODIFIED** — `API_URL` now reads from `API_URL` env var (falls back to `localhost:8000`) |

---

## Quick Reference

| Service | Local URL | Render URL |
|---------|-----------|------------|
| API | http://localhost:8000 | https://logistics-api-XXXX.onrender.com |
| API Docs | http://localhost:8000/docs | https://logistics-api-XXXX.onrender.com/docs |
| Frontend | http://localhost:8501 | https://logistics-frontend-XXXX.onrender.com |
| Health Check | http://localhost:8000/health | https://logistics-api-XXXX.onrender.com/health |

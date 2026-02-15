"""
FastAPI Application
====================
Logistics Delay prediction API with Prometheus metrics
and SHAP-powered explainability.

Run:
    uvicorn api.main:app --reload --port 8000
"""

import time
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from api.model_service import model_service
from api.schemas import HealthResponse, PredictionRequest, PredictionResponse

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Prometheus Metrics ───────────────────────────────────────────
PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total predictions served",
    ["predicted_class"],
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time taken per prediction",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Max confidence of predictions",
    buckets=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)
MODEL_LOAD_TIME = Gauge(
    "model_load_time_seconds",
    "Time taken to load the model at startup",
)

# ── App ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Logistics Delay Prediction API",
    description=(
        "Predict whether a logistics shipment will be **Early**, **On-time**, or **Late** "
        "using a trained RandomForest model with SHAP explainability and Prometheus metrics."
    ),
    version="1.0.0",
)

APP_START_TIME = time.time()


# ── Startup ──────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Load model into memory when the server starts."""
    try:
        load_time = model_service.load_model()
        MODEL_LOAD_TIME.set(load_time)
        logger.info(f"🚀 API ready — model loaded in {load_time:.2f}s")
    except Exception as exc:
        logger.error(f"❌ Failed to load model: {exc}")
        logger.warning("API will start but /predict will return 503 until model is available")


# ── Routes ───────────────────────────────────────────────────────
@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint with API info."""
    return {
        "service": "Logistics Delay Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/health", "/predict", "/metrics"],
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    """Health check — model status and uptime."""
    return HealthResponse(
        status="healthy" if model_service.is_loaded else "degraded",
        model_name=model_service.model_name,
        model_loaded=model_service.is_loaded,
        uptime_seconds=round(time.time() - APP_START_TIME, 2),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict shipment delay class.

    Returns predicted label, confidence scores per class,
    and top-5 SHAP feature impacts.
    """
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    start = time.time()

    try:
        result = model_service.predict(request.model_dump())
    except Exception as exc:
        logger.error(f"Prediction failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(exc)}")

    latency = time.time() - start

    # Record Prometheus metrics
    PREDICTIONS_TOTAL.labels(predicted_class=result["class_name"]).inc()
    PREDICTION_LATENCY.observe(latency)
    PREDICTION_CONFIDENCE.observe(result["max_confidence"])

    logger.info(
        f"Predicted: {result['class_name']} "
        f"(conf={result['max_confidence']:.2f}, latency={latency:.3f}s)"
    )

    return PredictionResponse(**result)


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )

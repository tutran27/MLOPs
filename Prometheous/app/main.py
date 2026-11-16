import math
import time
from time import perf_counter

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

app = FastAPI(title="ML Prometheus Demo")

MODEL_NAME = "demo-logistic"
MODEL_VERSION = "1.0.0"
MODEL_LABEL = f"{MODEL_NAME}-v1"

# ============================================================
# Metrics
# ============================================================

REQUESTS = Counter(
    "ml_inference_requests_total",
    "Total number of ML inference requests",
    ["model", "status"],
)

IN_PROGRESS = Gauge(
    "ml_inference_in_progress",
    "Current number of inference requests being processed",
    ["model"],
)

LATENCY = Histogram(
    "ml_inference_duration_seconds",
    "ML inference latency in seconds",
    ["model"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)

PREDICTION_SCORE = Histogram(
    "ml_prediction_score",
    "Distribution of model prediction scores",
    ["model"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

MODEL_INFO = Info("ml_model", "Information about the loaded model")
MODEL_INFO.info({"name": MODEL_NAME, "version": MODEL_VERSION})

# Initialize known label values
REQUESTS.labels(model=MODEL_LABEL, status="success")
REQUESTS.labels(model=MODEL_LABEL, status="error")


class PredictRequest(BaseModel):
    value: float


@app.get("/")
def root():
    return {"message": "ML service is running", "model": MODEL_LABEL}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(data: PredictRequest):
    IN_PROGRESS.labels(model=MODEL_LABEL).inc()
    started_at = perf_counter()
    try:
        if data.value == -999:
            raise ValueError("Simulated model failure")

        # Simulate inference latency & compute score
        delay = min(abs(data.value) * 0.02, 0.3)
        time.sleep(delay)

        x = max(-20, min(20, data.value))
        score = 1 / (1 + math.exp(-x))
        prediction = int(score >= 0.5)

        PREDICTION_SCORE.labels(model=MODEL_LABEL).observe(score)
        REQUESTS.labels(model=MODEL_LABEL, status="success").inc()

        return {
            "prediction": prediction,
            "score": score,
            "model": MODEL_LABEL,
        }
    except Exception as exc:
        REQUESTS.labels(model=MODEL_LABEL, status="error").inc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        elapsed = perf_counter() - started_at
        LATENCY.labels(model=MODEL_LABEL).observe(elapsed)
        IN_PROGRESS.labels(model=MODEL_LABEL).dec()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
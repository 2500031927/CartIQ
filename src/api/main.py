"""CartIQ FastAPI service. Inference only: it loads saved artifacts and NEVER retrains.

Run from the project root:  uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Annotated, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src import config
from src.models.anomaly_detection import score_order
from src.models.segmentation import predict_segment
from src.monitoring.monitor import (
    PredictionLogger,
    RequestMetrics,
    compute_drift,
    configure_logging,
    read_model_metadata,
)

logger = logging.getLogger("cartiq.api")

NonNeg = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Pos = Annotated[float, Field(gt=0, allow_inf_nan=False)]

MODEL_FILES = {
    "basket_value": ("basket_value_model.joblib", "scripts/train_basket_value.py"),
    "reorder": ("reorder_model.joblib", "scripts/train_reorder.py"),
    "segmentation": ("customer_cluster_model.joblib", "scripts/train_segmentation.py"),
    "anomaly": ("anomaly_model.joblib", "scripts/train_anomaly_detection.py"),
}


# ------------------------------ schemas ------------------------------
class BasketRequest(BaseModel):
    customer_id: Optional[str] = Field(
        None, description="Optional; stored only as a salted hash"
    )
    prior_orders: int = Field(..., ge=1, description="Number of earlier orders")
    prior_total_spend: NonNeg
    prior_mean_value: NonNeg
    prior_median_value: NonNeg
    prior_max_value: NonNeg
    prior_std_value: NonNeg = 0.0
    last_order_value: NonNeg
    last_order_lines: int = Field(..., ge=1)
    last_order_items: int = Field(..., ge=1)
    tenure_days: NonNeg = Field(
        ..., description="Days from first order to most recent order"
    )
    avg_gap_days: Optional[NonNeg] = Field(
        None, description="Mean days between orders (null if 1 prior order)"
    )
    country: str = "United Kingdom"


class BasketResponse(BaseModel):
    predicted_basket_value: float
    currency_note: str = (
        "Same currency as the training data (GBP for UCI Online Retail)"
    )
    model_name: str
    model_version: str


class ReorderRequest(BaseModel):
    customer_id: Optional[str] = None
    recency_days: NonNeg
    frequency: int = Field(..., ge=1)
    monetary: NonNeg
    avg_order_value: NonNeg
    max_order_value: NonNeg
    total_items: NonNeg
    n_unique_products: int = Field(..., ge=1)
    tenure_days: NonNeg
    avg_gap_days: Optional[NonNeg] = None
    orders_last_30d: int = Field(0, ge=0)
    orders_last_90d: int = Field(0, ge=0)
    spend_last_90d: NonNeg = 0.0
    country: str = "United Kingdom"


class ReorderResponse(BaseModel):
    reorder_probability: float
    will_reorder_30d: bool
    decision_threshold: float
    calibrated: bool
    model_name: str
    model_version: str


class SegmentRequest(BaseModel):
    customer_id: Optional[str] = None
    recency_days: NonNeg
    frequency: int = Field(..., ge=1)
    monetary: Pos


class SegmentResponse(BaseModel):
    cluster_id: int
    segment_name: str
    recommended_action: str
    distance_to_centroid: float
    model_version: str


class AnomalyRequest(BaseModel):
    order_value: Pos
    n_lines: int = Field(..., ge=1)
    total_qty: Pos
    avg_unit_price: Pos


class AnomalyResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    model_version: str


# ------------------------------ registry ------------------------------
class ModelRegistry:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = Path(artifact_dir)
        self.bundles: dict = {}
        self.errors: dict = {}

    def load_all(self) -> None:
        self.bundles, self.errors = {}, {}
        for name, (fname, script) in MODEL_FILES.items():
            path = self.artifact_dir / fname
            if not path.exists():
                self.errors[name] = (
                    f"Model file {path} not found. "
                    f"Train it first: python {script}"
                )
                continue
            try:
                self.bundles[name] = joblib.load(path)
            except Exception as exc:
                self.errors[name] = (
                    f"Could not load {path}: {exc}. "
                    f"Re-train with python {script}"
                )

    def get(self, name: str) -> dict:
        if name not in self.bundles:
            raise HTTPException(
                status_code=503,
                detail=self.errors.get(
                    name, f"Model '{name}' is not loaded."
                ),
            )
        return self.bundles[name]


def _frame(payload: BaseModel, columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([payload.model_dump()])[columns]
    for c in df.columns:
        if c != "country":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def create_app(
    artifact_dir: Path | str | None = None,
    log_dir: Path | str | None = None,
) -> FastAPI:
    configure_logging()
    artifact_dir = Path(artifact_dir or config.ARTIFACT_DIR)
    log_dir = Path(log_dir or config.LOG_DIR)

    app = FastAPI(
        title="CartIQ API",
        version="1.0.0",
        description="E-commerce analytics: basket value, 30-day reorder, segments, anomalies.",
    )

    # ------------------------------ CORS ------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    registry = ModelRegistry(artifact_dir)
    registry.load_all()

    metrics = RequestMetrics()
    pred_log = PredictionLogger(log_dir / "predictions.jsonl")
    recent_reorder_inputs: deque = deque(maxlen=1000)

    app.state.registry, app.state.metrics = registry, metrics

    @app.exception_handler(RequestValidationError)
    async def validation_handler(
        request: Request,
        exc: RequestValidationError,
    ):
        details = [
            {
                "field": ".".join(
                    str(p) for p in e["loc"] if p != "body"
                ),
                "message": e["msg"],
            }
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "details": details,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.exception("unhandled error")
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "detail": "Unexpected server error; see server logs.",
            },
        )

    @app.middleware("http")
    async def track(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000

        metrics.record(
            request.url.path,
            response.status_code,
            ms,
        )

        logger.info(
            "request",
            extra={
                "extra_fields": {
                    "path": request.url.path,
                    "method": request.method,
                    "status": response.status_code,
                    "latency_ms": round(ms, 2),
                }
            },
        )

        response.headers["X-Process-Time-ms"] = f"{ms:.2f}"
        return response

    @app.get("/health")
    def health():
        models = {
            n: {
                "loaded": n in registry.bundles,
                "version": (
                    registry.bundles[n]["version"]
                    if n in registry.bundles
                    else None
                ),
                "error": registry.errors.get(n),
            }
            for n in MODEL_FILES
        }

        return {
            "status": "ok" if not registry.errors else "degraded",
            "models": models,
        }

    @app.post(
        "/predict/basket-value",
        response_model=BasketResponse,
    )
    def predict_basket(req: BasketRequest):
        b = registry.get("basket_value")

        pred = max(
            float(
                b["model"].predict(
                    _frame(req, b["feature_columns"])
                )[0]
            ),
            0.0,
        )

        out = BasketResponse(
            predicted_basket_value=round(pred, 2),
            model_name=b["name"],
            model_version=b["version"],
        )

        pred_log.log(
            "basket-value",
            b["version"],
            req.model_dump(exclude={"customer_id"}),
            {
                "predicted_basket_value": out.predicted_basket_value
            },
            req.customer_id,
        )

        return out

    @app.post(
        "/predict/reorder",
        response_model=ReorderResponse,
    )
    def predict_reorder(req: ReorderRequest):
        b = registry.get("reorder")

        X = _frame(req, b["feature_columns"])
        p = float(b["model"].predict_proba(X)[0, 1])

        recent_reorder_inputs.append(
            X.iloc[0].to_dict()
        )

        out = ReorderResponse(
            reorder_probability=round(p, 4),
            will_reorder_30d=p >= b["threshold"],
            decision_threshold=round(b["threshold"], 4),
            calibrated=b["calibrated"],
            model_name=b["name"],
            model_version=b["version"],
        )

        pred_log.log(
            "reorder",
            b["version"],
            req.model_dump(exclude={"customer_id"}),
            {
                "reorder_probability": out.reorder_probability
            },
            req.customer_id,
        )

        return out

    @app.post(
        "/predict/customer-segment",
        response_model=SegmentResponse,
    )
    def predict_customer_segment(req: SegmentRequest):
        b = registry.get("segmentation")

        res = predict_segment(
            b,
            req.recency_days,
            req.frequency,
            req.monetary,
        )

        pred_log.log(
            "customer-segment",
            b["version"],
            req.model_dump(exclude={"customer_id"}),
            {"segment": res["segment_name"]},
            req.customer_id,
        )

        return SegmentResponse(
            **res,
            model_version=b["version"],
        )

    @app.post(
        "/predict/anomaly",
        response_model=AnomalyResponse,
    )
    def predict_anomaly(req: AnomalyRequest):
        b = registry.get("anomaly")

        res = score_order(
            b,
            req.order_value,
            req.n_lines,
            req.total_qty,
            req.avg_unit_price,
        )

        pred_log.log(
            "anomaly",
            b["version"],
            req.model_dump(),
            res,
        )

        return AnomalyResponse(
            **res,
            model_version=b["version"],
        )

    @app.get("/metrics")
    def monitoring_summary():
        ref_path = artifact_dir / "drift_reference.json"

        if ref_path.exists():
            import json

            ref = json.loads(
                ref_path.read_text(encoding="utf-8")
            )["features"]

            drift = compute_drift(
                ref,
                pd.DataFrame(
                    list(recent_reorder_inputs)
                ),
            )
        else:
            drift = {
                "status": "no_reference",
                "detail": (
                    "Train the reorder model to create "
                    "drift_reference.json"
                ),
            }

        return {
            "requests": metrics.summary(),
            "models": read_model_metadata(artifact_dir),
            "reorder_input_drift": drift,
        }

    @app.post("/admin/reload-models")
    def reload_models(request: Request):
        """Hot-reload artifacts after a manual retraining run.
        Set CARTIQ_ADMIN_KEY to require a header.
        """
        key = os.environ.get("CARTIQ_ADMIN_KEY")

        if key and request.headers.get("X-Admin-Key") != key:
            raise HTTPException(
                status_code=403,
                detail="Invalid or missing X-Admin-Key header.",
            )

        registry.load_all()

        return {
            "loaded": sorted(registry.bundles),
            "errors": registry.errors,
        }

    return app


app = create_app()
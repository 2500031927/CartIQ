"""Monitoring: structured logs, request/latency metrics, prediction log, PSI drift, model metadata (M6)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": datetime.now(timezone.utc).isoformat(), "level": record.levelname,
                   "logger": record.name, "message": record.getMessage()}
        payload.update(getattr(record, "extra_fields", {}) or {})
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if any(getattr(h, "_cartiq", False) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler._cartiq = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)


def make_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


class RequestMetrics:
    """Thread-safe request counts and latency stats per endpoint."""

    def __init__(self, max_samples: int = 1000):
        self._lock = threading.Lock()
        self._counts: dict = defaultdict(int)
        self._errors: dict = defaultdict(int)
        self._lat: dict = defaultdict(lambda: deque(maxlen=max_samples))
        self._started = time.time()

    def record(self, endpoint: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self._counts[endpoint] += 1
            if status_code >= 400:
                self._errors[endpoint] += 1
            self._lat[endpoint].append(latency_ms)

    def summary(self) -> dict:
        with self._lock:
            eps = {}
            for ep, n in self._counts.items():
                lat = np.array(self._lat[ep])
                eps[ep] = {"count": n, "errors": self._errors[ep], "mean_ms": float(lat.mean()),
                           "p95_ms": float(np.percentile(lat, 95)), "max_ms": float(lat.max())}
            return {"total_requests": int(sum(self._counts.values())),
                    "uptime_seconds": round(time.time() - self._started, 1), "endpoints": eps}


class PredictionLogger:
    """Append-only JSONL log. Customer IDs are salted+hashed; no names/addresses are ever accepted."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._salt = os.environ.get("CARTIQ_HASH_SALT", "cartiq-dev-salt")

    def hash_id(self, customer_id: str | None) -> str | None:
        if not customer_id:
            return None
        return hashlib.sha256((self._salt + str(customer_id)).encode()).hexdigest()[:12]

    def log(self, endpoint: str, model_version: str, inputs: dict, outputs: dict,
            customer_id: str | None = None) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), "endpoint": endpoint,
                  "model_version": model_version, "customer_hash": self.hash_id(customer_id),
                  "inputs": inputs, "outputs": outputs}
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:
            logging.getLogger(__name__).warning("could not write prediction log: %s", exc)


# ----------------------------- drift (PSI) -----------------------------
def build_drift_reference(df: pd.DataFrame, columns: list[str], n_bins: int = 10) -> dict:
    """Quantile bins + expected proportions from TRAINING data, saved with the model."""
    ref = {}
    for c in columns:
        s = df[c].dropna().astype(float)
        if s.nunique() < 2:
            continue
        edges = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            continue
        inner = edges[1:-1]
        counts = np.bincount(np.searchsorted(inner, s.values, side="right"), minlength=len(edges) - 1)
        ref[c] = {"inner_edges": inner.tolist(), "expected": (counts / counts.sum()).tolist()}
    return ref


def population_stability_index(expected, actual, eps: float = 1e-4) -> float:
    e = np.clip(np.asarray(expected, float), eps, None)
    a = np.clip(np.asarray(actual, float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def compute_drift(reference: dict, recent: pd.DataFrame, min_rows: int = 50) -> dict:
    """PSI per feature: <0.1 stable, 0.1-0.25 moderate, >0.25 significant (common rule of thumb)."""
    n = len(recent)
    if n < min_rows:
        return {"status": "insufficient_data", "n_rows": n, "min_rows": min_rows, "features": {}}
    feats, worst = {}, 0.0
    for c, spec in reference.items():
        if c not in recent.columns:
            continue
        vals = pd.to_numeric(recent[c], errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        counts = np.bincount(np.searchsorted(spec["inner_edges"], vals, side="right"),
                             minlength=len(spec["expected"]))
        psi = population_stability_index(spec["expected"], counts / counts.sum())
        worst = max(worst, psi)
        feats[c] = {"psi": round(psi, 4),
                    "status": "stable" if psi < 0.1 else "moderate" if psi < 0.25 else "significant"}
    status = "stable" if worst < 0.1 else "moderate" if worst < 0.25 else "significant"
    return {"status": status, "n_rows": n, "max_psi": round(worst, 4), "features": feats}


# --------------------------- model metadata ---------------------------
def write_model_metadata(artifact_dir: Path | str, name: str, info: dict) -> None:
    path = Path(artifact_dir) / "model_metadata.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[name] = info
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def read_model_metadata(artifact_dir: Path | str) -> dict:
    path = Path(artifact_dir) / "model_metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

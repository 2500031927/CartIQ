"""Order-level anomaly detection with Isolation Forest (M4)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src import config
from src.evaluation.evaluate import save_json_report
from src.monitoring.monitor import make_version, write_model_metadata

log = logging.getLogger(__name__)
ORDER_FEATURES = ["order_value", "n_lines", "total_qty", "avg_unit_price"]
MODEL_FILE = "anomaly_model.joblib"


def train_anomaly_detection(orders: pd.DataFrame, artifact_dir: Path | str = config.ARTIFACT_DIR,
                            contamination: float = 0.01) -> dict:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    o = orders.sort_values("InvoiceDate").reset_index(drop=True)
    if len(o) < 100:
        raise ValueError("Need at least 100 orders to fit an anomaly detector.")
    cut = int(len(o) * 0.8)  # fit on the older 80%, check behaviour on the newest 20%
    train, later = o.iloc[:cut], o.iloc[cut:]
    pipe = Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                     ("scale", StandardScaler()),
                     ("iso", IsolationForest(n_estimators=200, contamination=contamination,
                                             random_state=config.RANDOM_STATE))]).fit(train[ORDER_FEATURES])
    flagged_later = pipe.predict(later[ORDER_FEATURES]) == -1
    scores = -pipe.decision_function(o[ORDER_FEATURES])  # higher = more anomalous
    o = o.assign(anomaly_score=scores, is_anomaly=pipe.predict(o[ORDER_FEATURES]) == -1)
    top = o.sort_values("anomaly_score", ascending=False).head(20)
    top[["InvoiceNo", "InvoiceDate"] + ORDER_FEATURES + ["anomaly_score"]].to_csv(
        artifact_dir / "top_anomalous_orders.csv", index=False)
    version = make_version()
    joblib.dump({"pipeline": pipe, "feature_columns": ORDER_FEATURES, "contamination": contamination,
                 "version": version}, artifact_dir / MODEL_FILE)
    report = {"orders": len(o), "contamination": contamination,
              "flagged_fraction_train": float(o.iloc[:cut]["is_anomaly"].mean()),
              "flagged_fraction_newest_20pct": float(flagged_later.mean()),
              "note": "Unsupervised: 'anomalous' means unusual, not fraudulent. Review flagged orders manually.",
              "top_anomalies": top[["InvoiceNo"] + ORDER_FEATURES + ["anomaly_score"]].to_dict(orient="records")}
    save_json_report(report, config.report_dir(artifact_dir) / "anomaly_report.json")
    write_model_metadata(artifact_dir, "anomaly", {
        "version": version, "trained_at": datetime.now(timezone.utc).isoformat(), "contamination": contamination})
    return report


def score_order(bundle: dict, order_value: float, n_lines: int, total_qty: float, avg_unit_price: float) -> dict:
    X = pd.DataFrame([{"order_value": order_value, "n_lines": n_lines, "total_qty": total_qty,
                       "avg_unit_price": avg_unit_price}])[bundle["feature_columns"]]
    pipe = bundle["pipeline"]
    return {"anomaly_score": float(-pipe.decision_function(X)[0]), "is_anomaly": bool(pipe.predict(X)[0] == -1)}

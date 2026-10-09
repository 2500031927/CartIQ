"""Metrics, time-aware CV, calibration summaries, plots and JSON reports (M5)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI needed (works in PyCharm/Windows terminals)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,  # noqa: E402
                             mean_absolute_error, mean_squared_error, precision_recall_curve,
                             r2_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV  # noqa: E402


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if len(y_true) == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None}
    return {"n": int(len(y_true)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None}


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Undefined metrics (e.g. ROC-AUC with one class) are returned as None, never crash."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two_classes = len(np.unique(y)) == 2
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else (0.0 if prec is not None and rec is not None else None)
    return {"n": int(len(y)), "prevalence": float(y.mean()) if len(y) else None, "threshold": float(threshold),
            "precision": prec, "recall": rec, "f1": f1,
            "roc_auc": float(roc_auc_score(y, p)) if two_classes else None,
            "pr_auc": float(average_precision_score(y, p)) if y.sum() > 0 else None,
            "brier": float(brier_score_loss(y, p)) if len(y) else None,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def best_f1_threshold(y_true, proba) -> float:
    """Pick the F1-maximising threshold on VALIDATION data only."""
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))]) if len(thr) else 0.5


def time_series_splits(groups, n_splits: int = 3):
    """Expanding-window CV: train on all groups before g, validate on group g (never on the past)."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        raise ValueError("Need at least 2 time groups for time-aware cross-validation.")
    n = min(n_splits, len(uniq) - 1)
    return [(np.where(groups < g)[0], np.where(groups == g)[0]) for g in uniq[-n:]]


def fit_candidate(estimator, grid: dict, X, y, cv, scoring: str):
    """Hyper-parameter search (GridSearchCV with the time-aware splits) or a plain fit."""
    if grid:
        search = GridSearchCV(estimator, grid, cv=cv, scoring=scoring, n_jobs=1, refit=True)
        search.fit(X, y)
        return search.best_estimator_, search.best_params_, float(search.best_score_)
    estimator.fit(X, y)
    return estimator, {}, None


def calibration_summary(y_true, proba, n_bins: int = 10) -> dict:
    y, p = np.asarray(y_true).astype(int), np.asarray(proba, float)
    if len(y) == 0:
        return {"bins": [], "ece": None}
    n_bins = max(2, min(n_bins, len(y) // 20))
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, max(len(edges) - 2, 0))
    bins, ece = [], 0.0
    for b in np.unique(idx):
        m = idx == b
        mp, fp_ = float(p[m].mean()), float(y[m].mean())
        bins.append({"mean_predicted": mp, "fraction_positive": fp_, "count": int(m.sum())})
        ece += m.sum() / len(y) * abs(mp - fp_)
    return {"bins": bins, "ece": float(ece)}


def plot_reliability(curves: dict, path: Path | str) -> None:
    """curves: {label: (y_true, proba)} -> reliability diagram PNG."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    for label, (y, p) in curves.items():
        bins = calibration_summary(y, p)["bins"]
        ax.plot([b["mean_predicted"] for b in bins], [b["fraction_positive"] for b in bins], "o-", label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction of reorders")
    ax.set_title("Reliability diagram (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def rank_features(names, values, top: int = 10) -> list[dict]:
    order = np.argsort(-np.abs(np.asarray(values, float)))[:top]
    return [{"feature": str(names[i]), "value": float(values[i])} for i in order]


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(float(o)) or math.isinf(float(o))) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o if isinstance(o, (str, int, bool)) or o is None else str(o)


def save_json_report(report: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    return path

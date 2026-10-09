"""30-day reorder classification (M2 logistic, M3 trees, M5 imbalance/calibration/selection)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.evaluation.evaluate import (best_f1_threshold, calibration_summary, classification_metrics,
                                     fit_candidate, plot_reliability, rank_features, save_json_report,
                                     time_series_splits)
from src.features.preprocessing import (CUSTOMER_FEATURES, CUSTOMER_NUMERIC, build_reorder_dataset,
                                        customer_preprocessor, split_by_cutoff)
from src.monitoring.monitor import build_drift_reference, make_version, write_model_metadata

log = logging.getLogger(__name__)
RS = config.RANDOM_STATE
MODEL_FILE = "reorder_model.joblib"


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class CalibratedClassifier:
    """Wraps a fitted pipeline + a Platt-scaling (sigmoid) calibrator fitted on VALIDATION data."""

    def __init__(self, base, calibrator=None):
        self.base = base
        self.calibrator = calibrator

    def predict_proba(self, X) -> np.ndarray:
        p = self.base.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return np.column_stack([1 - p, p])


def _platt(p, y) -> LogisticRegression:
    return LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)


def candidate_models(quick: bool = False) -> dict:
    n = 40 if quick else 150
    models = {
        "baseline_prior": (DummyClassifier(strategy="prior"), {}),
        "logreg_l2": (LogisticRegression(penalty="l2", class_weight="balanced", max_iter=3000),
                      {"C": [0.01, 0.1, 1, 10]}),
        "logreg_l1": (LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced",
                                         max_iter=3000), {"C": [0.05, 0.1, 1]}),
        "decision_tree": (DecisionTreeClassifier(class_weight="balanced", random_state=RS),
                          {"max_depth": [3, 5, 8], "min_samples_leaf": [50, 100]}),
        "random_forest": (RandomForestClassifier(n_estimators=n, class_weight="balanced_subsample",
                                                 random_state=RS, n_jobs=-1),
                          {"max_depth": [6, 12], "min_samples_leaf": [20, 50]}),
        "gradient_boosting": (GradientBoostingClassifier(n_estimators=n, random_state=RS),
                              {"learning_rate": [0.05, 0.1], "max_depth": [2, 3]}),
        "hist_gradient_boosting": (HistGradientBoostingClassifier(max_iter=n * 2, class_weight="balanced",
                                                                  random_state=RS),
                                   {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]}),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = (XGBClassifier(n_estimators=n, random_state=RS, n_jobs=1, verbosity=0),
                             {"max_depth": [3, 5], "learning_rate": [0.05, 0.1]})
    except ImportError:
        log.info("xgboost not installed - skipping (optional).")
    if quick:
        models = {k: (e, {p: v[:1] for p, v in g.items()}) for k, (e, g) in models.items()}
    return models


def _pipeline(est) -> Pipeline:
    return Pipeline([("prep", customer_preprocessor()), ("model", est)])


def train_reorder(lines: pd.DataFrame, artifact_dir: Path | str = config.ARTIFACT_DIR,
                  quick: bool = False) -> dict:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ds = build_reorder_dataset(lines)
    train, val, test = split_by_cutoff(ds)
    Xtr, ytr = train[CUSTOMER_FEATURES], train["reorder_30d"]
    Xva, yva = val[CUSTOMER_FEATURES], val["reorder_30d"]
    Xte, yte = test[CUSTOMER_FEATURES], test["reorder_30d"]
    if ytr.nunique() < 2:
        raise ValueError("Training labels contain a single class - cannot train a classifier.")
    cv = time_series_splits(train["cutoff"].values, n_splits=3)

    results, fitted, val_proba = {}, {}, {}
    for name, (est, grid) in candidate_models(quick).items():
        pgrid = {f"model__{k}": v for k, v in grid.items()}
        pipe, params, cv_score = fit_candidate(_pipeline(est), pgrid, Xtr, ytr, cv, "average_precision")
        fitted[name] = pipe
        val_proba[name] = pipe.predict_proba(Xva)[:, 1]
        results[name] = {"best_params": {k.replace("model__", ""): v for k, v in params.items()},
                         "cv_pr_auc": cv_score,
                         "train": classification_metrics(ytr, pipe.predict_proba(Xtr)[:, 1]),
                         "validation": classification_metrics(yva, val_proba[name])}
        log.info("%s val PR-AUC=%s", name, results[name]["validation"]["pr_auc"])

    contenders = {k: v for k, v in results.items() if k != "baseline_prior"}
    selected = max(contenders, key=lambda k: contenders[k]["validation"]["pr_auc"] or -1)
    best = fitted[selected]

    # ---- calibration (Platt scaling) fitted on validation, checked by cross-fitting across val cutoffs
    p_val = val_proba[selected]
    calibration = {"applied": False, "reason": ""}
    calibrator = None
    val_cutoffs = np.sort(val["cutoff"].unique())
    enough = (len(yva) >= 200 and yva.sum() >= 30 and (len(yva) - yva.sum()) >= 30 and len(val_cutoffs) >= 2)
    if not enough:
        calibration["reason"] = "validation set too small (need >=200 rows, >=30 per class, 2 cutoffs)"
    else:
        a, b = (val["cutoff"] == val_cutoffs[0]).values, (val["cutoff"] != val_cutoffs[0]).values
        y_arr = yva.values
        raw, cal, ys = [], [], []
        for fit_m, ev_m in ((a, b), (b, a)):
            if len(np.unique(y_arr[fit_m])) < 2:
                continue
            c = _platt(p_val[fit_m], y_arr[fit_m])
            raw.append(p_val[ev_m])
            cal.append(c.predict_proba(_logit(p_val[ev_m]).reshape(-1, 1))[:, 1])
            ys.append(y_arr[ev_m])
        if ys:
            ys, raw, cal = np.concatenate(ys), np.concatenate(raw), np.concatenate(cal)
            b_raw = float(np.mean((raw - ys) ** 2))
            b_cal = float(np.mean((cal - ys) ** 2))
            calibration.update({"cross_fit_brier_uncalibrated": b_raw, "cross_fit_brier_calibrated": b_cal})
            if b_cal < b_raw:
                calibrator = _platt(p_val, y_arr)
                calibration.update({"applied": True, "reason": "Brier score improved in cross-fitting"})
            else:
                calibration["reason"] = "calibration did not improve Brier score; kept raw probabilities"
        else:
            calibration["reason"] = "a validation cutoff had a single class"
    final = CalibratedClassifier(best, calibrator)
    final_val = final.predict_proba(Xva)[:, 1]
    threshold = best_f1_threshold(yva, final_val)  # chosen on validation only

    # ---- interpretation
    names = best.named_steps["prep"].get_feature_names_out()
    inner = best.named_steps["model"]
    interpretation: dict = {}
    if hasattr(inner, "coef_"):
        interpretation["coefficients_log_odds"] = rank_features(names, np.ravel(inner.coef_))
    if hasattr(inner, "feature_importances_"):
        interpretation["impurity_importance"] = rank_features(names, inner.feature_importances_)
    lr = fitted["logreg_l2"].named_steps["model"]
    interpretation["logreg_l2_odds_ratios"] = [
        {"feature": str(n), "odds_ratio": float(np.exp(c))}
        for n, c in sorted(zip(names, np.ravel(lr.coef_)), key=lambda t: -abs(t[1]))[:10]]
    try:
        perm = permutation_importance(best, Xva, yva, n_repeats=3 if quick else 5, random_state=RS,
                                      scoring="average_precision", n_jobs=1)
        interpretation["permutation_importance_val_pr_auc"] = rank_features(list(Xva.columns),
                                                                            perm.importances_mean)
    except ValueError as exc:
        interpretation["permutation_importance_val_pr_auc"] = f"skipped: {exc}"

    # ---- final, one-time test evaluation
    p_test_final = final.predict_proba(Xte)[:, 1]
    p_test_raw = best.predict_proba(Xte)[:, 1]
    lr_thr = best_f1_threshold(yva, val_proba["logreg_l2"])
    test_metrics = {
        f"{selected}_final": classification_metrics(yte, p_test_final, threshold),
        "logreg_l2_linear_baseline": classification_metrics(
            yte, fitted["logreg_l2"].predict_proba(Xte)[:, 1], lr_thr),
        "baseline_prior": classification_metrics(yte, fitted["baseline_prior"].predict_proba(Xte)[:, 1], 0.5)}
    plot_reliability({"uncalibrated": (yte, p_test_raw), "final": (yte, p_test_final)},
                     config.report_dir(artifact_dir) / "reorder_reliability.png")

    report = {
        "task": "reorder_30d (will the customer place an order within 30 days after the cutoff)",
        "label_design": {"horizon_days": config.HORIZON_DAYS,
                         "features": "transactions strictly before each cutoff",
                         "cutoffs_kept": "only cutoffs with a fully observed 30-day window",
                         "split": "chronological by cutoff (train < validation < test)"},
        "data": {"rows": len(ds), "train": len(train), "validation": len(val), "test": len(test),
                 "prevalence": {"train": float(ytr.mean()), "validation": float(yva.mean()),
                                "test": float(yte.mean())},
                 "train_cutoffs": [str(c) for c in np.sort(train["cutoff"].unique())],
                 "val_cutoffs": [str(c) for c in val_cutoffs],
                 "test_cutoffs": [str(c) for c in np.sort(test["cutoff"].unique())]},
        "imbalance_handling": "class_weight='balanced' where supported + F1-optimal threshold chosen on validation",
        "selection_rule": "highest validation PR-AUC (excluding baseline); test untouched until now",
        "candidates": results, "selected_model": selected, "calibration": calibration,
        "validation_calibration_summary": calibration_summary(yva, final_val),
        "test_calibration_summary": {"uncalibrated": calibration_summary(yte, p_test_raw),
                                     "final": calibration_summary(yte, p_test_final)},
        "decision_threshold": threshold, "final_test_metrics": test_metrics,
        "interpretation": interpretation,
        "note": "Customers appear in several cutoffs; splits are time-based, so customers can appear in "
                "train and test with different (later) snapshots - this mimics real deployment.",
    }
    version = make_version()
    joblib.dump({"model": final, "threshold": threshold, "feature_columns": CUSTOMER_FEATURES,
                 "numeric_columns": CUSTOMER_NUMERIC, "name": selected, "calibrated": calibrator is not None,
                 "version": version}, artifact_dir / MODEL_FILE)
    # Reference copy of the fitted shared preprocessor + drift reference built from TRAINING data only.
    prep = customer_preprocessor().fit(Xtr)
    joblib.dump(prep, artifact_dir / "preprocessor.joblib")
    save_json_report({"features": build_drift_reference(train, CUSTOMER_NUMERIC), "version": version},
                     artifact_dir / "drift_reference.json")
    save_json_report(report, config.report_dir(artifact_dir) / "reorder_report.json")
    write_model_metadata(artifact_dir, "reorder", {
        "version": version, "model": selected, "calibrated": calibrator is not None, "threshold": threshold,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "validation_pr_auc": results[selected]["validation"]["pr_auc"],
        "test_pr_auc": test_metrics[f"{selected}_final"]["pr_auc"]})
    return report

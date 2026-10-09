"""Basket value regression (M2 linear models, M3 trees, M5 selection).

One prediction = expected total value of a customer's NEXT order, given their order history.
Target is log-transformed during training (order values are heavy-tailed); metrics are
always computed on the original currency scale.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src import config
from src.evaluation.evaluate import (fit_candidate, rank_features, regression_metrics, save_json_report,
                                     time_series_splits)
from src.features.preprocessing import (BASKET_FEATURES, BASKET_NUMERIC, basket_preprocessor,
                                        build_basket_dataset, split_by_time)
from src.monitoring.monitor import make_version, write_model_metadata

log = logging.getLogger(__name__)
RS = config.RANDOM_STATE
MODEL_FILE = "basket_value_model.joblib"

BIAS_VARIANCE_NOTE = (
    "Linear models have high bias (cannot bend) but low variance; deep trees have low bias but high "
    "variance (memorise noise). Compare train vs validation error: a big gap = overfitting. Regularisation "
    "(Ridge/Lasso/ElasticNet alpha, tree depth, min_samples_leaf) trades variance for bias.")


def candidate_models(quick: bool = False) -> dict:
    n = 40 if quick else 150
    models = {
        "baseline_median": (DummyRegressor(strategy="median"), {}),
        "linear_regression": (LinearRegression(), {}),
        "ridge": (Ridge(), {"alpha": [0.1, 1, 10, 100]}),
        "lasso": (Lasso(max_iter=20000), {"alpha": [0.0005, 0.005, 0.05]}),
        "elastic_net": (ElasticNet(max_iter=20000), {"alpha": [0.001, 0.01], "l1_ratio": [0.2, 0.5, 0.8]}),
        "decision_tree": (DecisionTreeRegressor(random_state=RS),
                          {"max_depth": [3, 5, 8], "min_samples_leaf": [20, 50]}),
        "random_forest": (RandomForestRegressor(n_estimators=n, random_state=RS, n_jobs=-1),
                          {"max_depth": [6, 12], "min_samples_leaf": [5, 20]}),
        "gradient_boosting": (GradientBoostingRegressor(n_estimators=n, random_state=RS),
                              {"learning_rate": [0.05, 0.1], "max_depth": [2, 3]}),
        "hist_gradient_boosting": (HistGradientBoostingRegressor(max_iter=n * 2, random_state=RS),
                                   {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]}),
    }
    try:  # optional dependency
        from xgboost import XGBRegressor
        models["xgboost"] = (XGBRegressor(n_estimators=n, random_state=RS, n_jobs=1, verbosity=0),
                             {"max_depth": [3, 5], "learning_rate": [0.05, 0.1]})
    except ImportError:
        log.info("xgboost not installed - skipping (optional).")
    if quick:  # smoke-test mode: first value of each grid only
        models = {k: (e, {p: v[:1] for p, v in g.items()}) for k, (e, g) in models.items()}
    return models


def _pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", basket_preprocessor()),
                     ("model", TransformedTargetRegressor(regressor=estimator, func=np.log1p,
                                                          inverse_func=np.expm1))])


def train_basket_value(orders: pd.DataFrame, artifact_dir: Path | str = config.ARTIFACT_DIR,
                       quick: bool = False) -> dict:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ds = build_basket_dataset(orders)
    train, val, test = split_by_time(ds)
    Xtr, ytr = train[BASKET_FEATURES], train["next_order_value"]
    Xva, yva = val[BASKET_FEATURES], val["next_order_value"]
    Xte, yte = test[BASKET_FEATURES], test["next_order_value"]
    cv = time_series_splits(train["InvoiceDate"].dt.to_period("M").astype(str).values, n_splits=3)

    results, fitted = {}, {}
    for name, (est, grid) in candidate_models(quick).items():
        pgrid = {f"model__regressor__{k}": v for k, v in grid.items()}
        # Selection uses CV on TRAIN only, then validation - the test set is untouched.
        pipe, params, cv_score = fit_candidate(_pipeline(est), pgrid, Xtr, ytr, cv, "neg_mean_absolute_error")
        fitted[name] = pipe
        results[name] = {"best_params": {k.replace("model__regressor__", ""): v for k, v in params.items()},
                         "cv_mae": None if cv_score is None else -cv_score,
                         "train": regression_metrics(ytr, pipe.predict(Xtr)),
                         "validation": regression_metrics(yva, pipe.predict(Xva))}
        log.info("%s val MAE=%.2f", name, results[name]["validation"]["mae"])

    contenders = {k: v for k, v in results.items() if k != "baseline_median"}
    selected = min(contenders, key=lambda k: contenders[k]["validation"]["mae"])
    best = fitted[selected]

    names = best.named_steps["prep"].get_feature_names_out()
    inner = best.named_steps["model"].regressor_
    interpretation: dict = {}
    if hasattr(inner, "coef_"):  # linear: coefficients are in log-target units
        interpretation["coefficients_log_target"] = rank_features(names, np.ravel(inner.coef_))
    if hasattr(inner, "feature_importances_"):
        interpretation["impurity_importance"] = rank_features(names, inner.feature_importances_)
    ridge_inner = fitted["ridge"].named_steps["model"].regressor_
    interpretation["ridge_coefficients_log_target"] = rank_features(names, np.ravel(ridge_inner.coef_))
    perm = permutation_importance(best, Xva, yva, n_repeats=3 if quick else 5, random_state=RS,
                                  scoring="neg_mean_absolute_error", n_jobs=1)
    interpretation["permutation_importance_val_mae"] = rank_features(list(Xva.columns), perm.importances_mean)

    report = {
        "task": "basket_value (expected value of the customer's next order)",
        "data": {"rows": len(ds), "train": len(train), "validation": len(val), "test": len(test),
                 "train_dates": [str(train["InvoiceDate"].min()), str(train["InvoiceDate"].max())],
                 "val_dates": [str(val["InvoiceDate"].min()), str(val["InvoiceDate"].max())],
                 "test_dates": [str(test["InvoiceDate"].min()), str(test["InvoiceDate"].max())]},
        "selection_rule": "lowest validation MAE (excluding the dummy baseline); test untouched until now",
        "candidates": results,
        "selected_model": selected,
        "final_test_metrics": {  # computed once, after selection
            selected: regression_metrics(yte, best.predict(Xte)),
            "ridge_linear_baseline": regression_metrics(yte, fitted["ridge"].predict(Xte)),
            "baseline_median": regression_metrics(yte, fitted["baseline_median"].predict(Xte))},
        "interpretation": interpretation,
        "bias_variance_note": BIAS_VARIANCE_NOTE,
    }
    version = make_version()
    joblib.dump({"model": best, "feature_columns": BASKET_FEATURES, "numeric_columns": BASKET_NUMERIC,
                 "name": selected, "version": version}, artifact_dir / MODEL_FILE)
    save_json_report(report, config.report_dir(artifact_dir) / "basket_value_report.json")
    write_model_metadata(artifact_dir, "basket_value", {
        "version": version, "model": selected, "trained_at": datetime.now(timezone.utc).isoformat(),
        "validation_mae": results[selected]["validation"]["mae"],
        "test_mae": report["final_test_metrics"][selected]["mae"]})
    return report

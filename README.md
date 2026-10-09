# CartIQ — E-commerce Customer Analytics and Personalization System

CartIQ learns from purchase history to (1) predict a customer's **next order value**, (2) predict whether they
will **reorder within 30 days**, (3) **segment** customers (RFM + K-means), and (4) **flag unusual orders**
(Isolation Forest). Models are served by FastAPI with monitoring (latency, drift, versions).

> **Honest status.** The code was written and tested in a sandbox that could **not reach the UCI website**
> (HTTP 403 `host_not_allowed`). So: tests and a live API smoke-run passed on a clearly-labelled **synthetic**
> fixture, but **no model has been trained on the real UCI data yet and no real-world metric exists.**
> You will produce those numbers yourself with the commands below. Example outputs in this README are
> *illustrative* unless marked "actual run (synthetic)".

## 1. Business value
Predicted next-basket value → size offers/free-shipping thresholds. Reorder probability → target win-back emails
only where they change behaviour. Segments → different campaigns per group (the API returns a recommended action).
Anomaly flags → review mistaken or suspicious orders (unusual ≠ fraud).

## 2. Dataset
**UCI Online Retail** (id 352) — https://archive.ics.uci.edu/dataset/352/online+retail — UK online gift retailer,
2010-12-01 to 2011-12-09, ~541k lines, one `Online Retail.xlsx` file.

| Column | Meaning |
|---|---|
| InvoiceNo | order id; prefix `C` = cancellation/return |
| StockCode | product id (non-product codes like POST, BANK CHARGES exist) |
| Description | product name |
| Quantity | units (negative on returns) |
| InvoiceDate | timestamp |
| UnitPrice | GBP per unit |
| CustomerID | customer id (**~25% missing**) |
| Country | customer country |

**Cleaning rules** (`src/data/ingestion.py::clean_transactions`, counts saved in `data/processed/cleaning_report.json`):
drop missing CustomerID → drop `C` invoices → drop Quantity ≤ 0 → drop UnitPrice ≤ 0 → drop non-product
StockCodes (must start with 5 digits) → drop exact duplicates. **Limitations:** ~1 year only (weak seasonality
estimates); returns are removed, not netted; no customer demographics; guest checkouts (no CustomerID) are lost;
a wholesale-heavy customer mix means heavy-tailed spend. If CustomerID were missing entirely, `validate_schema`
stops with an explanation: only order-level anomaly detection would remain possible.
Fallback dataset: UCI **Online Retail II** (id 502) — the loader maps its column names automatically.

## 3. Architecture
```
data/raw ──prepare_data──▶ data/processed/transactions_clean.csv.gz
      │ (src/data/ingestion.py)
      ▼
src/features/preprocessing.py  (ONE shared feature + preprocessor definition)
      ▼                         ▼
 train_*.py scripts ──▶ artifacts/*.joblib + artifacts/reports/*.json
      (src/models/*)             ▼
                         src/api/main.py (inference only) ──▶ logs/predictions.jsonl, /metrics
```
Training (`scripts/`, `src/models/`) and inference (`src/api/`) are separate; the API never trains.

## 4. Modelling design (leakage prevention)
**Reorder (30 days).** Cutoffs every 30 days starting 60 days after the first transaction. For each cutoff:
features use only transactions **before** the cutoff; label = customer placed ≥1 order in
`[cutoff, cutoff+30d)`. Only cutoffs whose full 30-day window lies inside the data are kept (so "no future data"
is never mistaken for "no reorder"). Split by cutoff: oldest → train, next 2 → validation, newest 2 → test.
Non-overlapping windows mean no label leaks across splits.

**Basket value.** One prediction = expected value of a customer's **next order**. Row = a customer's i-th order
(i ≥ 2); target = its order total; features come from orders 1…i-1 only (cumulative stats on shifted values).
Time since the previous order is deliberately **not** a feature (unknown before the next order happens).
Split by order date 70/15/15. Target is `log1p`-transformed in training; metrics are on the money scale.

**Segmentation.** Customer-level RFM as of the day after the last transaction; `log1p` + `StandardScaler`;
K-means++ with k chosen by silhouette (k ≥ 3), elbow plot saved; DBSCAN and PCA for comparison/visualisation.
Cluster names are **derived from measured medians** vs overall quartiles (never from cluster numbers).
Segments are descriptive, not used as features for labels.

## 5. Course mapping

| Course concept | Python file | Implementation | How to run / verify |
|---|---|---|---|
| M1 lifecycle: ingestion, validation, cleaning | `src/data/ingestion.py`, `scripts/download_data.py`, `scripts/prepare_data.py` | schema check, documented cleaning, processed CSV | `python scripts/prepare_data.py`; see `cleaning_report.json` |
| M1 retraining workflow | `scripts/retrain_all.py` | re-prepare + retrain all, print versions | `python scripts/retrain_all.py` |
| M1 train/inference separation | `src/models/*` vs `src/api/main.py` | API loads joblib only | `GET /health` |
| M2 Linear/Ridge/Lasso/ElasticNet | `src/models/basket_value.py` | log-target linear family + coefficient ranking | `artifacts/reports/basket_value_report.json` → `interpretation` |
| M2 Logistic (L2, L1), sigmoid, probabilities | `src/models/reorder.py` | class-weighted logistic, odds ratios | `reorder_report.json` → `logreg_l2_odds_ratios` |
| M2 scaling, encoding, imputation | `src/features/preprocessing.py::build_preprocessor` | median impute, log1p, StandardScaler, one-hot | `pytest tests/test_features.py` |
| M3 trees, forests, boosting, HistGB, XGBoost (optional) | `basket_value.py`, `reorder.py` | grid-searched, compared with linear baseline | `candidates` table in both reports |
| M3 feature / permutation importance | same | impurity + permutation on validation | `interpretation` in reports |
| M3 bias–variance | same | train-vs-validation metrics per model + note | `bias_variance_note`, `candidates.*.train/validation` |
| M4 RFM, K-means++, elbow, silhouette, PCA, DBSCAN | `src/models/segmentation.py` | see §4 | `artifacts/reports/segmentation_plots.png`, `segmentation_report.json` |
| M4 Isolation Forest | `src/models/anomaly_detection.py` | order-level detector | `artifacts/top_anomalous_orders.csv` |
| M5 chronological splits + time-aware CV | `preprocessing.py::split_by_*`, `evaluate.py::time_series_splits` | expanding-window CV inside train only | `pytest` (`test_chronological_*`) |
| M5 metrics, imbalance, calibration, reports | `src/evaluation/evaluate.py`, `reorder.py` | confusion matrix, P/R/F1, ROC/PR-AUC, MAE/RMSE/R², Platt calibration, reliability PNG, JSON | `artifacts/reports/*.json`, `reorder_reliability.png` |
| M6 FastAPI + Pydantic + validation | `src/api/main.py` | 4 predict endpoints, clear 422/503 errors | `http://127.0.0.1:8000/docs`, `pytest tests/test_api.py` |
| M6 monitoring: latency, counts, drift (PSI), versions, prediction log | `src/monitoring/monitor.py` | `/metrics`, `logs/predictions.jsonl` (customer ids hashed) | `GET /metrics` |
| M6 tests | `tests/` | 16 tests | `pytest` |

**Model selection rule:** pick by *validation* score (basket: lowest MAE; reorder: highest PR-AUC); test metrics
are computed once afterwards. The F1 threshold and calibrator are also fitted on validation only. Logistic
calibration is applied only if validation has ≥200 rows, ≥30 per class and 2 cutoffs, **and** cross-fitting
across the two validation cutoffs shows a better Brier score; otherwise raw probabilities are kept and the report says why.

**Training–serving skew** = features computed differently in training and production (different imputation,
scaling, category handling). Here the preprocessing `ColumnTransformer` is defined once
(`build_preprocessor`), fitted inside each saved model pipeline, and the API passes **raw** feature rows to that
same pipeline, so the exact fitted transformations run at inference. `preprocessor.joblib` is a reference copy
(fitted on reorder training data) for inspection; models use their embedded copies. One caveat: the *client*
must compute input features with the same definitions (e.g. `recency_days` = days since last order) — see §8.

## 6. Run from scratch (Windows, PowerShell, PyCharm)
1. **Open in PyCharm:** File → Open → select the `CartIQ` folder (Python 3.11 recommended: python.org installer, tick "Add to PATH").
2. **Terminal in the project root** (PyCharm: View → Tool Windows → Terminal):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
   *If activation is blocked* ("running scripts is disabled"): run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`,
   **or** skip activation and call the venv's Python directly: `.\.venv\Scripts\python.exe scripts\prepare_data.py`
   (use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `-m uvicorn ...` the same way).
   In PyCharm: Settings → Project → Python Interpreter → Add → Existing → `.venv\Scripts\python.exe`.
3. **Dataset:** `python scripts/download_data.py` — success prints "Verified dataset file ...". If it fails, download
   manually (instructions printed; also in §2) into `data\raw\`.
4. **Prepare/validate:** `python scripts/prepare_data.py` (~1–2 min for the xlsx) → prints a cleaning report.
5. **Train (each writes artifacts + a JSON report):**
```powershell
python scripts/train_basket_value.py
python scripts/train_reorder.py
python scripts/train_segmentation.py
python scripts/train_anomaly_detection.py
```
   Expect several minutes for the grid searches. Add `--quick` (basket/reorder) only for a fast smoke test.
   Artifacts: `artifacts\basket_value_model.joblib`, `reorder_model.joblib`, `customer_cluster_model.joblib`,
   `preprocessor.joblib`, plus `anomaly_model.joblib`, `drift_reference.json`, `model_metadata.json`, `reports\`.
6. **Reports:** open `artifacts\reports\*.json` (`basket_value_report.json`, `reorder_report.json`,
   `segmentation_report.json`, `anomaly_report.json`) and the PNGs `reorder_reliability.png`, `segmentation_plots.png`.
7. **Start the API:** `python -m uvicorn src.api.main:app --reload` → open **http://127.0.0.1:8000/docs** (Swagger UI).
8. **Sample requests** (PowerShell: use `curl.exe`, not the `curl` alias; or use "Try it out" in Swagger):
```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST http://127.0.0.1:8000/predict/reorder -H "Content-Type: application/json" -d "{\"recency_days\":20,\"frequency\":4,\"monetary\":400,\"avg_order_value\":100,\"max_order_value\":180,\"total_items\":120,\"n_unique_products\":25,\"tenure_days\":150,\"avg_gap_days\":35,\"orders_last_30d\":1,\"orders_last_90d\":2,\"spend_last_90d\":200}"
curl.exe -X POST http://127.0.0.1:8000/predict/basket-value -H "Content-Type: application/json" -d "{\"prior_orders\":3,\"prior_total_spend\":300,\"prior_mean_value\":100,\"prior_median_value\":90,\"prior_max_value\":150,\"prior_std_value\":20,\"last_order_value\":95,\"last_order_lines\":8,\"last_order_items\":30,\"tenure_days\":120,\"avg_gap_days\":40}"
curl.exe -X POST http://127.0.0.1:8000/predict/customer-segment -H "Content-Type: application/json" -d "{\"recency_days\":10,\"frequency\":5,\"monetary\":500}"
curl.exe -X POST http://127.0.0.1:8000/predict/anomaly -H "Content-Type: application/json" -d "{\"order_value\":99999,\"n_lines\":1,\"total_qty\":80000,\"avg_unit_price\":1.0}"
curl.exe http://127.0.0.1:8000/metrics
```
   *Actual run (synthetic fixture, NOT real performance), reorder:*
   `{"reorder_probability":0.5455,"will_reorder_30d":true,"decision_threshold":0.4452,"calibrated":true,"model_name":"hist_gradient_boosting","model_version":"20261006-085634"}`
   *Actual run (synthetic), invalid request:* `{"error":"validation_error","details":[{"field":"order_value","message":"Input should be greater than 0"}, ...]}`.
   If a model isn't trained you get HTTP 503 with the exact training command to run.
9. **Tests:** `python -m pytest -q` (tests train tiny models on synthetic data in a temp folder; no dataset needed).
   *Actual result in the build sandbox:* `16 passed`.
10. **Monitoring:** `GET /metrics` (request counts, mean/p95 latency, model versions, PSI drift of recent reorder
    inputs vs training; drift shows `insufficient_data` until ≥50 requests). Prediction log: `logs\predictions.jsonl`
    (customer ids salted-hashed; set `CARTIQ_HASH_SALT`).
11. **Retrain safely (manual or scheduled — never per request):**
```powershell
python scripts/retrain_all.py                 # re-prepare data + retrain everything
python scripts/retrain_all.py --skip-prepare  # reuse the processed file
curl.exe -X POST http://127.0.0.1:8000/admin/reload-models   # or just restart uvicorn
```
    Compare new reports with old before reloading; keep a copy of the previous `artifacts\` folder for rollback.
    Windows Task Scheduler can run `.venv\Scripts\python.exe scripts\retrain_all.py` monthly.
12. **Try the pipeline without the real data (synthetic, for smoke tests only):**
```powershell
python scripts/make_synthetic_fixture.py
python scripts/train_reorder.py --input data/processed/SYNTHETIC_transactions.csv.gz --artifact-dir artifacts_synth --quick
```

## 7. Example request trace (reorder)
`POST /predict/reorder` → Pydantic validates types/ranges (negative recency → 422) → `ModelRegistry.get("reorder")`
(503 if not trained) → one-row DataFrame in the saved `feature_columns` order → saved pipeline
(impute → log1p → scale → one-hot → classifier → Platt calibration if applied) → probability → compared with the
validation-chosen threshold → JSON response + prediction-log line + latency recorded + inputs added to the drift buffer.

## 8. Input definitions for the API (must match training)
Reorder/segment features are computed over a customer's **entire history up to now**: `recency_days` = days since last
order; `frequency` = number of distinct orders; `monetary` = total spend; `tenure_days` = days since first order;
`avg_gap_days` = (last − first order date) / (orders − 1), null if one order; `orders_last_30d/90d`, `spend_last_90d`
over those windows. Basket-value inputs describe orders **before** the one being predicted; `tenure_days` there is
first-order → most recent order.

## 9. Limitations and ethical use
* One year of data from one retailer; results may not transfer. Real metrics are unknown until you train.
* Reorder positives include any order; no causal claim that a campaign changes behaviour (that needs an A/B test).
* Customers repeat across cutoffs; splits are chronological, but no customer-grouped hold-out is used.
* Segments/anomalies are unsupervised and descriptive; "anomalous" ≠ fraud — review manually.
* Use customer data lawfully (consent, minimisation). The API accepts no names/addresses; ids are hashed in logs.
  Avoid using predictions for discriminatory pricing or excluding groups from service; audit by country/segment.
* No online learning; drift alerts are advisory (PSI < 0.1 stable, 0.1–0.25 moderate, > 0.25 significant).

## 10. Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError: src` | run from the project root; scripts use `scripts/_bootstrap.py`; in PyCharm mark `CartIQ` as Sources Root |
| `...Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` or use `.venv\Scripts\python.exe` directly |
| `pip` build errors on very new Python (3.13+) | install Python 3.11/3.12 and recreate `.venv` |
| `Processed data ... not found` | run `prepare_data.py` first |
| 503 from API | train the model named in the error, then restart or `POST /admin/reload-models` |
| Download 403/timeout | manual download (§2); corporate proxies often block it |
| `Address already in use` | `--port 8001` |
| xlsx loading slow / `openpyxl` missing | normal (~1–2 min); `pip install openpyxl` |
| `curl` in PowerShell behaves oddly | use `curl.exe` or Swagger UI |
| Unpickling errors after upgrading sklearn | retrain with the pinned versions |

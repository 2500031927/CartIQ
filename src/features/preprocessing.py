"""Feature engineering shared by training AND serving (prevents training-serving skew).

Leakage rules used everywhere:
  * Every feature is computed only from transactions strictly BEFORE the prediction time.
  * Reorder labels use the window [cutoff, cutoff+30d) and only cutoffs whose full window
    lies inside the observed data are kept (no "unknown = no reorder" mistakes).
  * Basket target = value of order i; its features use orders 0..i-1 of that customer only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config

# ---------- customer snapshot features (reorder model, segmentation, drift) ----------
CUSTOMER_NUMERIC = ["recency_days", "frequency", "monetary", "avg_order_value", "max_order_value",
                    "total_items", "n_unique_products", "tenure_days", "avg_gap_days",
                    "orders_last_30d", "orders_last_90d", "spend_last_90d"]
CUSTOMER_SKEWED = ["frequency", "monetary", "avg_order_value", "max_order_value", "total_items",
                   "n_unique_products", "avg_gap_days", "spend_last_90d"]
CUSTOMER_CATEGORICAL = ["country"]
CUSTOMER_FEATURES = CUSTOMER_NUMERIC + CUSTOMER_CATEGORICAL

# ---------- next-order (basket value) features ----------
BASKET_NUMERIC = ["prior_orders", "prior_total_spend", "prior_mean_value", "prior_median_value",
                  "prior_max_value", "prior_std_value", "last_order_value", "last_order_lines",
                  "last_order_items", "tenure_days", "avg_gap_days"]
BASKET_SKEWED = [c for c in BASKET_NUMERIC if c != "tenure_days"]
BASKET_CATEGORICAL = ["country"]
BASKET_FEATURES = BASKET_NUMERIC + BASKET_CATEGORICAL


def build_preprocessor(numeric: list[str], skewed: list[str], categorical: list[str]) -> ColumnTransformer:
    """The ONE preprocessing definition: median impute -> (log1p for skewed) -> scale; one-hot for categories."""
    transformers = []
    skew = [c for c in numeric if c in skewed]
    other = [c for c in numeric if c not in skewed]
    if skew:
        transformers.append(("skewed_num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),  # tames heavy right tails
            ("scale", StandardScaler())]), skew))
    if other:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())]), other))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10,
                                     sparse_output=False))]), categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def customer_preprocessor() -> ColumnTransformer:
    return build_preprocessor(CUSTOMER_NUMERIC, CUSTOMER_SKEWED, CUSTOMER_CATEGORICAL)


def basket_preprocessor() -> ColumnTransformer:
    return build_preprocessor(BASKET_NUMERIC, BASKET_SKEWED, BASKET_CATEGORICAL)


# ------------------------------- snapshots -------------------------------
def build_customer_snapshot(lines: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """One row per customer using ONLY lines with InvoiceDate < cutoff."""
    cutoff = pd.Timestamp(cutoff)
    hist = lines[lines["InvoiceDate"] < cutoff]
    if hist.empty:
        return pd.DataFrame(columns=["CustomerID"] + CUSTOMER_FEATURES)
    orders = (hist.groupby(["CustomerID", "InvoiceNo"], as_index=False)
              .agg(order_date=("InvoiceDate", "min"), order_value=("LineTotal", "sum"),
                   items=("Quantity", "sum")))
    g = orders.groupby("CustomerID")
    snap = g.agg(first=("order_date", "min"), last=("order_date", "max"),
                 frequency=("InvoiceNo", "nunique"), monetary=("order_value", "sum"),
                 avg_order_value=("order_value", "mean"), max_order_value=("order_value", "max"),
                 total_items=("items", "sum"))
    day = pd.Timedelta(days=1)
    snap["recency_days"] = (cutoff - snap["last"]) / day
    snap["tenure_days"] = (cutoff - snap["first"]) / day
    span = (snap["last"] - snap["first"]) / day
    snap["avg_gap_days"] = np.where(snap["frequency"] > 1, span / (snap["frequency"] - 1).clip(lower=1), np.nan)
    snap["n_unique_products"] = hist.groupby("CustomerID")["StockCode"].nunique()
    for days in (30, 90):
        recent = orders[orders["order_date"] >= cutoff - pd.Timedelta(days=days)]
        snap[f"orders_last_{days}d"] = recent.groupby("CustomerID").size()
        if days == 90:
            snap["spend_last_90d"] = recent.groupby("CustomerID")["order_value"].sum()
    for c in ("orders_last_30d", "orders_last_90d", "spend_last_90d"):
        snap[c] = snap[c].fillna(0.0)
    country = hist.sort_values("InvoiceDate").groupby("CustomerID")["Country"].last()
    snap["country"] = country.astype(str)
    snap = snap.reset_index()
    return snap[["CustomerID"] + CUSTOMER_FEATURES]


def make_cutoffs(min_date: pd.Timestamp, max_date: pd.Timestamp, horizon_days: int = config.HORIZON_DAYS,
                 step_days: int | None = None, min_history_days: int = 60) -> list[pd.Timestamp]:
    """Prediction cutoffs whose full label window [cutoff, cutoff+horizon) is observed."""
    step = step_days or horizon_days  # step == horizon => label windows never overlap
    t = pd.Timestamp(min_date).normalize() + pd.Timedelta(days=min_history_days)
    cutoffs = []
    while t + pd.Timedelta(days=horizon_days) <= pd.Timestamp(max_date):
        cutoffs.append(t)
        t += pd.Timedelta(days=step)
    return cutoffs


def build_reorder_dataset(lines: pd.DataFrame, horizon_days: int = config.HORIZON_DAYS,
                          min_history_days: int = 60) -> pd.DataFrame:
    """Stack customer snapshots at several cutoffs; label = any order in the next 30 days."""
    cutoffs = make_cutoffs(lines["InvoiceDate"].min(), lines["InvoiceDate"].max(),
                           horizon_days, None, min_history_days)
    if len(cutoffs) < 5:
        raise ValueError(f"Only {len(cutoffs)} valid cutoffs - need >=5 for train/val/test. "
                         "The data span is too short for a 30-day reorder task.")
    parts = []
    for cutoff in cutoffs:
        snap = build_customer_snapshot(lines, cutoff)
        end = cutoff + pd.Timedelta(days=horizon_days)
        future = lines[(lines["InvoiceDate"] >= cutoff) & (lines["InvoiceDate"] < end)]
        snap["reorder_30d"] = snap["CustomerID"].isin(set(future["CustomerID"].unique())).astype(int)
        snap["cutoff"] = cutoff
        parts.append(snap)
    return pd.concat(parts, ignore_index=True)


def split_by_cutoff(df: pd.DataFrame, n_val: int = 2, n_test: int = 2):
    """Chronological split by cutoff: oldest cutoffs train, then validation, newest test."""
    uniq = np.sort(df["cutoff"].unique())
    if len(uniq) < n_val + n_test + 1:
        raise ValueError("Not enough cutoffs for a chronological train/val/test split.")
    train_c, val_c, test_c = uniq[:-(n_val + n_test)], uniq[-(n_val + n_test):-n_test], uniq[-n_test:]
    return (df[df["cutoff"].isin(train_c)].reset_index(drop=True),
            df[df["cutoff"].isin(val_c)].reset_index(drop=True),
            df[df["cutoff"].isin(test_c)].reset_index(drop=True))


def build_basket_dataset(orders: pd.DataFrame) -> pd.DataFrame:
    """Target = value of a customer's i-th order (i>=2); features = that customer's earlier orders only."""
    o = orders.sort_values(["CustomerID", "InvoiceDate", "InvoiceNo"]).reset_index(drop=True)
    o["prior_orders"] = o.groupby("CustomerID").cumcount()
    o["prior_total_spend"] = o.groupby("CustomerID")["order_value"].cumsum() - o["order_value"]
    o["prior_mean_value"] = o["prior_total_spend"] / o["prior_orders"].replace(0, np.nan)
    o["_prev"] = o.groupby("CustomerID")["order_value"].shift(1)  # shifted: excludes the target order
    gp = o.groupby("CustomerID")["_prev"]
    o["prior_median_value"] = gp.transform(lambda s: s.expanding().median())
    o["prior_max_value"] = gp.cummax()
    o["prior_std_value"] = gp.transform(lambda s: s.expanding().std()).fillna(0.0)
    o["last_order_value"] = o["_prev"]
    o["last_order_lines"] = o.groupby("CustomerID")["n_lines"].shift(1)
    o["last_order_items"] = o.groupby("CustomerID")["total_qty"].shift(1)
    prev_date = o.groupby("CustomerID")["InvoiceDate"].shift(1)
    first_date = o.groupby("CustomerID")["InvoiceDate"].transform("first")
    o["tenure_days"] = (prev_date - first_date) / pd.Timedelta(days=1)
    o["avg_gap_days"] = np.where(o["prior_orders"] >= 2,
                                 o["tenure_days"] / (o["prior_orders"] - 1).clip(lower=1), np.nan)
    o["country"] = o["Country"].astype(str)
    o["next_order_value"] = o["order_value"]
    o = o[o["prior_orders"] >= 1]
    cols = ["CustomerID", "InvoiceNo", "InvoiceDate"] + BASKET_FEATURES + ["next_order_value"]
    return o[cols].sort_values("InvoiceDate").reset_index(drop=True)


def split_by_time(df: pd.DataFrame, date_col: str = "InvoiceDate", train_frac: float = 0.7,
                  val_frac: float = 0.15):
    """Chronological split by date quantiles (train oldest, test newest)."""
    t1 = df[date_col].quantile(train_frac)
    t2 = df[date_col].quantile(train_frac + val_frac)
    train = df[df[date_col] < t1]
    val = df[(df[date_col] >= t1) & (df[date_col] < t2)]
    test = df[df[date_col] >= t2]
    for name, part in (("train", train), ("validation", val), ("test", test)):
        if len(part) < 30:
            raise ValueError(f"{name} split has only {len(part)} rows - not enough data.")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

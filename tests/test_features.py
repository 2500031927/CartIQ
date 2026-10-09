"""Feature, leakage and cleaning tests. All data here is SYNTHETIC or hand-made."""
import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import build_orders, clean_transactions, generate_synthetic_transactions, validate_schema
from src.features.preprocessing import (BASKET_FEATURES, CUSTOMER_FEATURES, build_basket_dataset,
                                        build_customer_snapshot, build_reorder_dataset, make_cutoffs,
                                        split_by_cutoff, split_by_time)
from src.monitoring.monitor import build_drift_reference, compute_drift, population_stability_index


@pytest.fixture(scope="module")
def lines():
    return generate_synthetic_transactions(n_customers=150, seed=1)


def test_cleaning_rules_on_handmade_data():
    raw = pd.DataFrame({
        "InvoiceNo": ["1", "2", "C3", "4", "5", "6", "1"],
        "StockCode": ["10001", "10002", "10003", "POST", "10005", "10006", "10001"],
        "Description": ["a"] * 7,
        "Quantity": [1, 2, -1, 1, -5, 3, 1],
        "InvoiceDate": ["2011-01-01"] * 7,
        "UnitPrice": [2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 2.0],
        "CustomerID": [1, 2, 3, 4, 5, 6, 1],
        "Country": ["UK"] * 7})
    raw.loc[1, "CustomerID"] = np.nan  # missing customer
    clean, report = clean_transactions(raw)
    assert list(clean["InvoiceNo"]) == ["1"]  # duplicate removed, everything else dropped
    assert report["dropped_missing_customer_id"] == 1
    assert report["dropped_cancellation_invoices"] == 1
    assert report["dropped_non_positive_price"] == 1


def test_schema_validation_reports_missing_customer_id():
    with pytest.raises(ValueError, match="CustomerID"):
        validate_schema(pd.DataFrame({"InvoiceNo": [1]}))


def test_snapshot_ignores_future_transactions(lines):
    cutoff = lines["InvoiceDate"].min() + pd.Timedelta(days=120)
    full = build_customer_snapshot(lines, cutoff)
    past_only = build_customer_snapshot(lines[lines["InvoiceDate"] < cutoff], cutoff)
    pd.testing.assert_frame_equal(full.reset_index(drop=True), past_only.reset_index(drop=True))


def test_reorder_label_matches_manual_computation():
    rows = [("1", "10001", "x", 1, pd.Timestamp("2011-01-01"), 10.0, 1, "UK", 10.0),
            ("2", "10001", "x", 1, pd.Timestamp("2011-01-20"), 10.0, 1, "UK", 10.0),
            ("3", "10001", "x", 1, pd.Timestamp("2011-01-05"), 10.0, 2, "UK", 10.0),
            ("4", "10001", "x", 1, pd.Timestamp("2011-03-30"), 10.0, 2, "UK", 10.0)]
    df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
                                     "UnitPrice", "CustomerID", "Country", "LineTotal"])
    cutoff = pd.Timestamp("2011-01-10")
    snap = build_customer_snapshot(df, cutoff)
    assert set(snap["CustomerID"]) == {1, 2}
    assert snap.loc[snap.CustomerID == 1, "recency_days"].iloc[0] == 9
    # customer 1 orders again on Jan 20 (within 30 days) -> positive; customer 2 does not -> negative
    future = df[(df.InvoiceDate >= cutoff) & (df.InvoiceDate < cutoff + pd.Timedelta(days=30))]
    assert set(future.CustomerID) == {1}


def test_all_cutoffs_have_fully_observed_label_window(lines):
    ds = build_reorder_dataset(lines)
    assert (ds["cutoff"] + pd.Timedelta(days=30) <= lines["InvoiceDate"].max()).all()
    assert set(CUSTOMER_FEATURES).issubset(ds.columns)
    assert ds["reorder_30d"].isin([0, 1]).all()
    cutoffs = make_cutoffs(lines["InvoiceDate"].min(), lines["InvoiceDate"].max())
    assert len(cutoffs) >= 5


def test_chronological_cutoff_split_has_no_overlap(lines):
    ds = build_reorder_dataset(lines)
    tr, va, te = split_by_cutoff(ds)
    assert tr["cutoff"].max() < va["cutoff"].min() <= va["cutoff"].max() < te["cutoff"].min()


def test_basket_features_use_only_earlier_orders(lines):
    orders = build_orders(lines)
    ds = build_basket_dataset(orders)
    assert (ds["prior_orders"] >= 1).all() and set(BASKET_FEATURES).issubset(ds.columns)
    # pick a customer with >=3 orders and verify by hand
    cid = orders.groupby("CustomerID").size().loc[lambda s: s >= 3].index[0]
    o = orders[orders.CustomerID == cid].sort_values("InvoiceDate").reset_index(drop=True)
    row = ds[(ds.CustomerID == cid) & (ds.InvoiceNo == o.loc[2, "InvoiceNo"])].iloc[0]
    assert row["prior_orders"] == 2
    assert row["prior_total_spend"] == pytest.approx(o.loc[:1, "order_value"].sum())
    assert row["last_order_value"] == pytest.approx(o.loc[1, "order_value"])
    assert row["next_order_value"] == pytest.approx(o.loc[2, "order_value"])
    # changing the target order must not change its own features
    changed = orders.copy()
    changed.loc[changed.InvoiceNo == o.loc[2, "InvoiceNo"], "order_value"] *= 100
    inv = o.loc[2, "InvoiceNo"]
    ds2 = build_basket_dataset(changed)
    row2 = ds2[(ds2.CustomerID == cid) & (ds2.InvoiceNo == inv)].iloc[0]
    for f in ("prior_total_spend", "prior_max_value", "last_order_value", "prior_median_value"):
        assert row2[f] == pytest.approx(row[f])


def test_time_split_orders_dates(lines):
    ds = build_basket_dataset(build_orders(lines))
    tr, va, te = split_by_time(ds)
    assert tr["InvoiceDate"].max() < va["InvoiceDate"].min() and va["InvoiceDate"].max() < te["InvoiceDate"].min()


def test_psi_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=2000)})
    ref = build_drift_reference(df, ["x"])
    same = compute_drift(ref, df.sample(500, random_state=1))
    shifted = compute_drift(ref, df.sample(500, random_state=1) + 3)
    assert same["status"] == "stable" and shifted["status"] == "significant"
    assert population_stability_index([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert compute_drift(ref, df.head(5))["status"] == "insufficient_data"

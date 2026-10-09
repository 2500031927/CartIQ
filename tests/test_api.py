"""API tests. Models are trained in quick mode on SYNTHETIC data in a temp dir; metrics are meaningless."""
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.data.ingestion import build_orders, generate_synthetic_transactions
from src.models.anomaly_detection import train_anomaly_detection
from src.models.basket_value import train_basket_value
from src.models.reorder import train_reorder
from src.models.segmentation import train_segmentation

BASKET = {"prior_orders": 3, "prior_total_spend": 300.0, "prior_mean_value": 100.0, "prior_median_value": 90.0,
          "prior_max_value": 150.0, "prior_std_value": 20.0, "last_order_value": 95.0, "last_order_lines": 8,
          "last_order_items": 30, "tenure_days": 120.0, "avg_gap_days": 40.0, "country": "United Kingdom"}
REORDER = {"recency_days": 20.0, "frequency": 4, "monetary": 400.0, "avg_order_value": 100.0,
           "max_order_value": 180.0, "total_items": 120, "n_unique_products": 25, "tenure_days": 150.0,
           "avg_gap_days": 35.0, "orders_last_30d": 1, "orders_last_90d": 2, "spend_last_90d": 200.0,
           "country": "United Kingdom"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("artifacts")
    lines = generate_synthetic_transactions(n_customers=250, seed=3)
    orders = build_orders(lines)
    train_basket_value(orders, d, quick=True)
    train_reorder(lines, d, quick=True)
    train_segmentation(lines, d)
    train_anomaly_detection(orders, d)
    return TestClient(create_app(d, tmp_path_factory.mktemp("logs")))


def test_health_all_models_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and all(m["loaded"] for m in body["models"].values())


def test_basket_value(client):
    r = client.post("/predict/basket-value", json=BASKET)
    assert r.status_code == 200 and r.json()["predicted_basket_value"] >= 0
    r2 = client.post("/predict/basket-value", json={**BASKET, "avg_gap_days": None, "prior_orders": 1})
    assert r2.status_code == 200  # missing avg_gap_days is imputed by the shared pipeline


def test_reorder(client):
    r = client.post("/predict/reorder", json=REORDER)
    body = r.json()
    assert r.status_code == 200 and 0.0 <= body["reorder_probability"] <= 1.0
    assert isinstance(body["will_reorder_30d"], bool)


def test_segment_and_anomaly(client):
    seg = client.post("/predict/customer-segment", json={"recency_days": 10, "frequency": 5, "monetary": 500})
    assert seg.status_code == 200 and seg.json()["segment_name"]
    an = client.post("/predict/anomaly", json={"order_value": 99999, "n_lines": 1, "total_qty": 80000,
                                               "avg_unit_price": 1.0})
    assert an.status_code == 200 and "is_anomaly" in an.json()


def test_invalid_requests_return_clear_errors(client):
    r = client.post("/predict/reorder", json={**REORDER, "recency_days": -5})
    assert r.status_code == 422 and r.json()["error"] == "validation_error"
    assert any(d["field"] == "recency_days" for d in r.json()["details"])
    assert client.post("/predict/anomaly", json={"order_value": 10}).status_code == 422


def test_metrics_endpoint_counts_requests(client):
    client.post("/predict/reorder", json=REORDER)
    m = client.get("/metrics").json()
    assert m["requests"]["endpoints"]["/predict/reorder"]["count"] >= 1
    assert "reorder" in m["models"] and "reorder_input_drift" in m


def test_untrained_models_give_503(tmp_path):
    c = TestClient(create_app(tmp_path / "empty", tmp_path / "logs"))
    r = c.post("/predict/reorder", json=REORDER)
    assert r.status_code == 503 and "train_reorder.py" in r.json()["detail"]
    assert c.get("/health").json()["status"] == "degraded"

"""Write a SYNTHETIC transactions file for smoke tests (never real performance!).

Usage:  python scripts/make_synthetic_fixture.py
Then:   python scripts/train_reorder.py --input data/processed/SYNTHETIC_transactions.csv.gz --artifact-dir artifacts_synthetic --quick
"""
import _bootstrap  # noqa: F401

from src import config
from src.data.ingestion import generate_synthetic_transactions, save_processed

if __name__ == "__main__":
    path = save_processed(generate_synthetic_transactions(), config.PROCESSED_DIR / "SYNTHETIC_transactions.csv.gz")
    print(f"SYNTHETIC data written to {path}. Do not report scores trained on it as real-world performance.")

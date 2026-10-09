"""Central paths and constants. Paths can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = Path(os.environ.get("CARTIQ_ARTIFACT_DIR", ROOT / "artifacts"))
LOG_DIR = Path(os.environ.get("CARTIQ_LOG_DIR", ROOT / "logs"))

PROCESSED_TRANSACTIONS = PROCESSED_DIR / "transactions_clean.csv.gz"
CLEANING_REPORT = PROCESSED_DIR / "cleaning_report.json"

RANDOM_STATE = 42
HORIZON_DAYS = 30  # reorder label window


def report_dir(artifact_dir: Path | str | None = None) -> Path:
    d = Path(artifact_dir or ARTIFACT_DIR) / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d

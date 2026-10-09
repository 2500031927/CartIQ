"""Train the segmentation model from data/processed/ and save artifacts + a JSON report."""
import _bootstrap  # noqa: F401
import argparse
import json
import logging
import sys

from src import config
from src.data.ingestion import load_processed
from src.models.segmentation import train_segmentation

SUMMARY_KEYS = ("selected_model", "final_test_metrics", "calibration", "chosen_k", "k_selection", "dbscan",
                "flagged_fraction_train", "flagged_fraction_newest_20pct")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(config.PROCESSED_TRANSACTIONS))
    p.add_argument("--artifact-dir", default=str(config.ARTIFACT_DIR))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    lines = load_processed(args.input)
    data = lines
    report = train_segmentation(data, args.artifact_dir)
    print(json.dumps({k: v for k, v in report.items() if k in SUMMARY_KEYS}, indent=2, default=str))
    print(f"Artifacts + reports saved under {args.artifact_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

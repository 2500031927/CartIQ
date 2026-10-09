"""Load raw data, validate, clean, and write data/processed/transactions_clean.csv.gz."""
import _bootstrap  # noqa: F401
import argparse
import json
import sys

from src import config
from src.data.ingestion import (MANUAL_INSTRUCTIONS, clean_transactions, find_raw_file,
                                load_raw_transactions, save_processed)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None, help="raw file path (default: first file in data/raw)")
    args = p.parse_args()
    path = args.input or find_raw_file(config.RAW_DIR)
    if not path:
        print(f"No raw dataset found in {config.RAW_DIR}.\n{MANUAL_INSTRUCTIONS}")
        return 1
    print(f"Loading {path} (the .xlsx takes about a minute) ...")
    raw = load_raw_transactions(path)
    lines, report = clean_transactions(raw)
    out = save_processed(lines)
    config.CLEANING_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

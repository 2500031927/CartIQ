"""Download the UCI Online Retail dataset into data/raw/ and verify its schema."""
import _bootstrap  # noqa: F401
import argparse
import sys

from src import config
from src.data.ingestion import DatasetDownloadError, MANUAL_INSTRUCTIONS, download_dataset, read_header


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="download again even if a file exists")
    args = p.parse_args()
    try:
        path = download_dataset(config.RAW_DIR, force=args.force)
        head = read_header(path)  # real verification: file opens and has the required columns
    except DatasetDownloadError as exc:
        print(exc)
        return 1
    except Exception as exc:
        print(f"Downloaded/found file could not be verified: {exc}\n{MANUAL_INSTRUCTIONS}")
        return 1
    print(f"Verified dataset file: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print("Columns:", list(head.columns))
    return 0


if __name__ == "__main__":
    sys.exit(main())

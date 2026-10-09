"""Manual / scheduled retraining: (optionally) re-prepare data, retrain every model, show versions.

Safe by design: models are NEVER retrained inside the API. After this finishes, either restart
uvicorn or call POST /admin/reload-models. Schedule with Windows Task Scheduler if desired.
"""
import _bootstrap  # noqa: F401
import argparse
import subprocess
import sys
from pathlib import Path

from src import config
from src.monitoring.monitor import read_model_metadata

SCRIPTS = Path(__file__).resolve().parent
STEPS = ["train_basket_value.py", "train_reorder.py", "train_segmentation.py", "train_anomaly_detection.py"]


def run(script: str, extra: list) -> None:
    print(f"\n=== {script} ===")
    subprocess.run([sys.executable, str(SCRIPTS / script), *extra], check=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if not args.skip_prepare:
        run("prepare_data.py", [])
    for s in STEPS:
        run(s, ["--quick"] if args.quick and s in STEPS[:2] else [])
    for name, info in read_model_metadata(config.ARTIFACT_DIR).items():
        print(name, "->", info.get("version"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

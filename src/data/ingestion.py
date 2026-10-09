"""Dataset acquisition, validation, cleaning and order-level aggregation.

Primary dataset: UCI "Online Retail" (id 352), UK online gift retailer,
2010-12-01 to 2011-12-09. Fallback: UCI "Online Retail II" (id 502) - the
loader maps both column schemas to one canonical schema.
"""
from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)

UCI_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
UCI_PAGE = "https://archive.ics.uci.edu/dataset/352/online+retail"

MANUAL_INSTRUCTIONS = f"""
MANUAL DOWNLOAD FALLBACK
1. Open {UCI_PAGE} in your browser and click "Download".
2. Unzip it and put 'Online Retail.xlsx' (or the .zip itself) into: {config.RAW_DIR}
3. Re-run:  python scripts/prepare_data.py
Alternative dataset: UCI Online Retail II (https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- place 'online_retail_II.xlsx' in the same folder; the loader understands its columns.
"""

# Canonical schema and accepted aliases (lower-case, stripped).
COLUMN_ALIASES = {
    "invoiceno": "InvoiceNo", "invoice": "InvoiceNo",
    "stockcode": "StockCode",
    "description": "Description",
    "quantity": "Quantity",
    "invoicedate": "InvoiceDate",
    "unitprice": "UnitPrice", "price": "UnitPrice",
    "customerid": "CustomerID", "customer id": "CustomerID",
    "country": "Country",
}
REQUIRED_COLUMNS = ["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
LINE_COLUMNS = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
                "UnitPrice", "CustomerID", "Country", "LineTotal"]


class DatasetDownloadError(RuntimeError):
    pass


def find_raw_file(raw_dir: Path | str = config.RAW_DIR) -> Path | None:
    """Return the first dataset file in raw_dir (extracting a .zip if needed)."""
    raw_dir = Path(raw_dir)
    for z in sorted(raw_dir.glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if m.lower().endswith((".xlsx", ".csv")):
                    target = raw_dir / Path(m).name
                    if not target.exists():
                        target.write_bytes(zf.read(m))
    for pattern in ("*.xlsx", "*.xls", "*.csv"):
        files = sorted(raw_dir.glob(pattern))
        if files:
            return files[0]
    return None


def download_dataset(raw_dir: Path | str = config.RAW_DIR, url: str = UCI_URL,
                     timeout: int = 120, force: bool = False) -> Path:
    """Download the UCI zip and extract the spreadsheet. Raises with manual steps on failure."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = find_raw_file(raw_dir)
    if existing and not force:
        log.info("Dataset already present: %s", existing)
        return existing
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CartIQ/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
    except Exception as exc:  # network, DNS, proxy, HTTP errors
        raise DatasetDownloadError(f"Automatic download failed: {exc}\n{MANUAL_INSTRUCTIONS}") from exc
    if len(payload) < 1_000_000:
        raise DatasetDownloadError(
            f"Downloaded file is only {len(payload)} bytes - not the dataset.\n{MANUAL_INSTRUCTIONS}")
    try:
        zf = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise DatasetDownloadError(f"Download is not a valid zip.\n{MANUAL_INSTRUCTIONS}") from exc
    members = [m for m in zf.namelist() if m.lower().endswith((".xlsx", ".csv"))]
    if not members:
        raise DatasetDownloadError(f"Zip has no .xlsx/.csv file: {zf.namelist()}\n{MANUAL_INSTRUCTIONS}")
    target = raw_dir / Path(members[0]).name
    target.write_bytes(zf.read(members[0]))
    return target


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: COLUMN_ALIASES[str(c).strip().lower()] for c in df.columns
              if str(c).strip().lower() in COLUMN_ALIASES}
    return df.rename(columns=rename)


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing essential columns {missing}. Found: {list(df.columns)}. "
            "Without CustomerID no customer-level task (RFM, reorder, next-basket) is possible; "
            "only order-level anomaly detection would remain.")


def read_header(path: Path | str, nrows: int = 5) -> pd.DataFrame:
    """Cheap schema check used right after download (reads only a few rows)."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, nrows=nrows)
    else:
        df = pd.read_csv(path, nrows=nrows, encoding="ISO-8859-1")
    df = normalize_columns(df)
    validate_schema(df)
    return df


def load_raw_transactions(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/download_data.py or place the file manually.\n"
                                f"{MANUAL_INSTRUCTIONS}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)  # Online Retail II has 2 sheets
        df = pd.concat(sheets.values(), ignore_index=True)
    else:
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="ISO-8859-1")
    df = normalize_columns(df)
    validate_schema(df)
    return df


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply documented cleaning rules; returns (clean line items, report).

    Rules (in order): missing CustomerID dropped; cancellation invoices ('C' prefix)
    dropped; Quantity<=0 dropped; UnitPrice<=0 dropped; non-product StockCodes
    (POST, DOT, M, BANK CHARGES, ...) dropped by requiring a 5-digit prefix; exact
    duplicate lines dropped. Returns are NOT netted against orders (limitation).
    """
    validate_schema(raw)
    df = raw.copy()
    if "Description" not in df.columns:
        df["Description"] = ""
    report: dict = {"rows_raw": int(len(df))}

    def drop(mask: pd.Series, key: str) -> None:
        nonlocal df
        report[key] = int(mask.sum())
        df = df.loc[~mask].copy()

    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")

    drop(df["InvoiceDate"].isna() | df["Quantity"].isna() | df["UnitPrice"].isna(), "dropped_unparseable")
    drop(df["CustomerID"].isna(), "dropped_missing_customer_id")
    drop(df["InvoiceNo"].str.upper().str.startswith("C"), "dropped_cancellation_invoices")
    drop(df["Quantity"] <= 0, "dropped_non_positive_quantity")
    drop(df["UnitPrice"] <= 0, "dropped_non_positive_price")
    drop(~df["StockCode"].str.match(r"^\d{5}"), "dropped_non_product_stockcodes")
    before = len(df)
    df = df.drop_duplicates()
    report["dropped_duplicates"] = int(before - len(df))

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["Country"] = df["Country"].astype(str).str.strip()
    df["LineTotal"] = df["Quantity"] * df["UnitPrice"]
    df = df[LINE_COLUMNS].sort_values(["InvoiceDate", "InvoiceNo"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No rows left after cleaning - check the dataset.")
    report.update({
        "rows_clean": int(len(df)),
        "customers": int(df["CustomerID"].nunique()),
        "orders": int(df["InvoiceNo"].nunique()),
        "date_min": str(df["InvoiceDate"].min()),
        "date_max": str(df["InvoiceDate"].max()),
    })
    return df, report


def build_orders(lines: pd.DataFrame) -> pd.DataFrame:
    """Aggregate line items to one row per invoice (order)."""
    orders = (lines.groupby(["CustomerID", "InvoiceNo"], as_index=False)
              .agg(InvoiceDate=("InvoiceDate", "min"), order_value=("LineTotal", "sum"),
                   n_lines=("StockCode", "size"), total_qty=("Quantity", "sum"),
                   avg_unit_price=("UnitPrice", "mean"), Country=("Country", "first")))
    return orders.sort_values(["InvoiceDate", "InvoiceNo"]).reset_index(drop=True)


def save_processed(lines: pd.DataFrame, path: Path | str = config.PROCESSED_TRANSACTIONS) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines.to_csv(path, index=False, compression="gzip")
    return path


def load_processed(path: Path | str = config.PROCESSED_TRANSACTIONS) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed data {path} not found. Run: python scripts/prepare_data.py")
    df = pd.read_csv(path, parse_dates=["InvoiceDate"], dtype={"InvoiceNo": str, "StockCode": str})
    validate_schema(df)
    return df


def generate_synthetic_transactions(n_customers: int = 400, n_days: int = 330, seed: int = 42,
                                    start: str = "2011-01-01") -> pd.DataFrame:
    """SYNTHETIC smoke-test data (already in cleaned schema). Never report scores from it as real."""
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    codes = np.arange(10000, 10060)
    prices = np.clip(rng.lognormal(0.8, 0.6, len(codes)).round(2), 0.2, None)
    countries = ["United Kingdom"] * 8 + ["Germany", "France"]
    rows, invoice = [], 900000
    for cid in range(20000, 20000 + n_customers):
        mean_gap = rng.uniform(10, 60)
        country = str(rng.choice(countries))
        t = rng.uniform(0, mean_gap)
        while t < n_days:
            invoice += 1
            ts = (start_ts + pd.Timedelta(days=float(t))).floor("min")
            for p in rng.choice(len(codes), size=int(rng.integers(1, 9)), replace=False):
                rows.append((str(invoice), str(codes[p]), "SYNTHETIC ITEM", int(rng.integers(1, 13)),
                             ts, float(prices[p]), cid, country))
            t += rng.exponential(mean_gap)
    df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "UnitPrice", "CustomerID", "Country"])
    df["LineTotal"] = df["Quantity"] * df["UnitPrice"]
    return df.sort_values(["InvoiceDate", "InvoiceNo"]).reset_index(drop=True)

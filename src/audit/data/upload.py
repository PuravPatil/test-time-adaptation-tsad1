"""Turn an uploaded table into a Dataset, so any numeric time series can be run.

    df = read_table("readings.csv")          # .csv .tsv .txt .xlsx .parquet .json .npy
    info = inspect(df)                       # columns, guessed time/label columns, preview
    data, report = prepare(df, time_col="auto", label_col=info["label_col"], train_frac=0.5)

What "any dataset" means here: a table with one row per timestep and numeric
columns (one per sensor). Univariate and multivariate both work. Free text,
images or tables with no time order are not time series and are rejected or
ignored column by column.

``prepare`` follows the SMD loader's conventions: the first ``train_frac`` of the
rows (in time order) is the training part and is ASSUMED NORMAL; the rest is the
test stream; values are standardised with statistics from the training part only.
Without a label column the test labels are all zero and ``report["has_labels"]``
is False, so callers must not report accuracy metrics.

Files are read as data only (``allow_pickle=False`` for .npy; no pickle anywhere).
Every error raised for bad input is an UploadError whose message is safe to show.
"""
from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import Dataset
from .smd import standardize_fit

FORMATS = (".csv", ".tsv", ".txt", ".xlsx", ".parquet", ".json", ".npy")
TIME_NAMES = {"time", "timestamp", "date", "datetime", "ts", "time_stamp"}
LABEL_NAMES = {"label", "labels", "anomaly", "is_anomaly", "anomalous", "attack", "outlier", "target"}
WHITESPACE = r"\s+"


class UploadError(ValueError):
    """Bad or unusable input. The message is meant for the person who uploaded the file."""


# -- reading ---------------------------------------------------------------
def read_table(path) -> pd.DataFrame:
    """Read a supported file into a DataFrame with unique, non-empty string column names."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in FORMATS:
        raise UploadError(f"unsupported file type '{ext}'. Supported: {', '.join(FORMATS)}")
    try:
        if ext in (".csv", ".tsv", ".txt"):
            df = _read_delimited(path, ext)
        elif ext == ".xlsx":
            df = pd.read_excel(path)
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        elif ext == ".json":
            df = _read_json(path)
        else:
            df = _read_npy(path)
    except UploadError:
        raise
    except ImportError as e:
        raise UploadError(f"reading {ext} files needs an extra package on the server ({e.name or e})") from e
    except Exception as e:  # malformed file: report it, never crash the caller
        raise UploadError(f"could not read the {ext} file: {e}") from e
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise UploadError("the file has no rows or no columns")
    df.columns = _unique_names(df.columns)
    return df.reset_index(drop=True)


def _unique_names(columns) -> list[str]:
    names, seen = [], {}
    for i, c in enumerate(columns):
        name = str(c).strip() or f"col_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        seen.setdefault(name, 0)
        names.append(name)
    return names


def _read_delimited(path: Path, ext: str) -> pd.DataFrame:
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        head = f.read(65536)
    lines = [line for line in head.splitlines() if line.strip()]
    if not lines:
        raise UploadError("the file is empty")
    sep = {".csv": ",", ".tsv": "\t"}.get(ext) or _sniff(head)
    first = _split(lines[0], sep)
    header = None if _all_numbers(first) else 0
    df = pd.read_csv(path, sep=sep, header=header, encoding="utf-8-sig", skip_blank_lines=True)
    if header is None:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    return df


def _sniff(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample[:8192], delimiters=",\t;|").delimiter
    except csv.Error:
        return WHITESPACE


def _split(line: str, sep: str) -> list[str]:
    tokens = line.split() if sep == WHITESPACE else line.split(sep)
    return [t.strip().strip("'\"") for t in tokens]


def _all_numbers(tokens: list[str]) -> bool:
    filled = [t for t in tokens if t]
    if not filled:
        return False
    try:
        for t in filled:
            float(t)
    except ValueError:
        return False
    return True


def _read_json(path: Path) -> pd.DataFrame:
    try:
        return pd.read_json(path)
    except ValueError:
        return pd.read_json(path, lines=True)   # JSON Lines: one record per line


def _read_npy(path: Path) -> pd.DataFrame:
    arr = np.load(path, allow_pickle=False)
    if arr.dtype.names:                          # structured array: named fields
        return pd.DataFrame(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise UploadError(f"expected a 1-D or 2-D array (timesteps x features), got {arr.ndim}-D")
    return pd.DataFrame(arr, columns=[f"col_{i}" for i in range(arr.shape[1])])


# -- inspecting ------------------------------------------------------------
def _as_numeric(s: pd.Series) -> pd.Series | None:
    """The column as floats, or None if it is not (mostly) numeric."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    if pd.api.types.is_datetime64_any_dtype(s):
        return None
    conv = pd.to_numeric(s, errors="coerce")
    present = s.notna().sum()
    if present and conv.notna().sum() >= 0.95 * present:
        return conv.astype(float)
    return None


def _as_time(s: pd.Series) -> pd.Series | None:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conv = pd.to_datetime(s, errors="coerce")
    present = s.notna().sum()
    return conv if present and conv.notna().sum() >= 0.95 * present else None


def guess_time_col(df: pd.DataFrame) -> str | None:
    """A column named like a timestamp, else the first column that parses as dates."""
    for c in df.columns:
        if c.lower() in TIME_NAMES:
            return c
    for c in df.columns:
        if _as_numeric(df[c]) is None and _as_time(df[c].head(200)) is not None:
            return c
    return None


def guess_label_col(df: pd.DataFrame) -> str | None:
    """A 0/1 column whose name says it is a label (never guessed from values alone)."""
    for c in df.columns:
        if c.lower() in LABEL_NAMES:
            v = _as_numeric(df[c])
            if v is not None and set(np.unique(v.dropna())) <= {0.0, 1.0}:
                return c
    return None


def inspect(df: pd.DataFrame, max_rows: int | None = None, preview_rows: int = 5) -> dict:
    """What the upload screen shows: size, columns, guesses and the first rows."""
    if max_rows and len(df) > max_rows:
        raise UploadError(f"the file has {len(df):,} rows; the limit is {max_rows:,}")
    columns = [{"name": c, "dtype": str(df[c].dtype), "numeric": _as_numeric(df[c]) is not None,
                "missing": int(df[c].isna().sum())} for c in df.columns]
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_numeric": sum(c["numeric"] for c in columns),
        "columns": columns,
        "time_col": guess_time_col(df),
        "label_col": guess_label_col(df),
        "preview": df.head(preview_rows).astype(str).values.tolist(),
    }


# -- preparing -------------------------------------------------------------
def prepare(df: pd.DataFrame, time_col: str | None = "auto", label_col: str | None = None,
            feature_cols: list[str] | None = None, train_frac: float = 0.5,
            name: str = "upload", min_train: int = 50, min_test: int = 20) -> tuple[Dataset, dict]:
    """Sort by time, pick numeric features, fill gaps, split train/test, scale.

    time_col: a column name, "auto" (guess), or None (row order is time order).
    label_col: a 0/1 (or >0 = anomaly) column, "auto" (guess), or None.
    feature_cols: columns to use, or None for every numeric column except time/label.
    """
    if not 0 < train_frac < 1:
        raise UploadError("train_frac must be between 0 and 1")
    for col in [c for c in (time_col, label_col) if c not in (None, "auto")] + list(feature_cols or []):
        if col not in df.columns:
            raise UploadError(f"column '{col}' is not in the file")
    tc = guess_time_col(df) if time_col == "auto" else time_col
    lc = guess_label_col(df) if label_col == "auto" else label_col
    warn = []

    if tc:
        t = _as_time(df[tc])
        if t is None:
            t = _as_numeric(df[tc])
        if t is None:
            raise UploadError(f"time column '{tc}' is neither dates nor numbers")
        order = np.argsort(t.to_numpy(), kind="mergesort")   # stable; NaT/NaN go last
        if not np.array_equal(order, np.arange(len(df))):
            warn.append(f"rows were sorted by '{tc}'")
        df = df.iloc[order].reset_index(drop=True)

    candidates = list(feature_cols) if feature_cols else [c for c in df.columns if c not in (tc, lc)]
    features, ignored = {}, []
    for c in candidates:
        v = _as_numeric(df[c])
        if v is None:
            if feature_cols:
                raise UploadError(f"feature column '{c}' is not numeric")
            ignored.append(c)
        else:
            features[c] = v
    if not features:
        raise UploadError("no numeric columns to use as features")
    if ignored:
        warn.append(f"ignored non-numeric columns: {', '.join(ignored)}")

    X = pd.DataFrame(features)
    n_missing = int(X.isna().sum().sum())
    if n_missing:
        X = X.ffill().bfill().fillna(0.0)
        warn.append(f"filled {n_missing:,} missing values from the previous timestep")

    y = None
    if lc:
        yv = _as_numeric(df[lc])
        if yv is None:
            raise UploadError(f"label column '{lc}' is not numeric")
        y = (yv.fillna(0.0).to_numpy() > 0).astype(int)

    n = len(X)
    n_train = int(round(n * train_frac))
    if n_train < min_train or n - n_train < min_test:
        raise UploadError(f"{n:,} rows is too few: need at least {min_train} for training "
                          f"and {min_test} to test with this split")
    values = X.to_numpy(dtype=np.float32)
    train, test = values[:n_train], values[n_train:]
    lo, rng = standardize_fit(train)
    labels = y[n_train:] if y is not None else np.zeros(n - n_train, dtype=int)
    if y is not None:
        if y[:n_train].any():
            warn.append(f"the training part has {int(y[:n_train].sum()):,} labelled anomalies; "
                        "the model assumes it is normal")
        if not labels.any():
            warn.append("the test part has no labelled anomalies, so AUROC/AUPRC are undefined")

    report = {
        "name": name, "n_rows": n, "n_train": n_train, "n_test": n - n_train,
        "features": list(features), "n_features": len(features), "ignored": ignored,
        "time_col": tc, "label_col": lc, "has_labels": y is not None,
        "test_anomaly_rate": float(labels.mean()) if y is not None else None,
        "missing_filled": n_missing, "warnings": warn,
    }
    return Dataset(name, (train - lo) / rng, (test - lo) / rng, labels), report

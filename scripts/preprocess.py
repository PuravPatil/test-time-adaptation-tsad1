"""Convert data/raw/smd/ into the common format in data/processed/smd/<machine>/.

Per machine: train.npy (T_train, D), test.npy (T_test, D), test_labels.npy (T_test,)
with 0 = normal, 1 = anomaly.

    python scripts/preprocess.py
"""
import argparse
from pathlib import Path

import numpy as np


def preprocess_smd(raw: Path, out: Path) -> None:
    files = sorted((raw / "train").glob("machine-*.txt"))
    if not files:
        raise SystemExit(f"No SMD files in {raw}/train. Run scripts/download_data.py first.")
    for f in files:
        m = f.stem
        train = np.loadtxt(raw / "train" / f"{m}.txt", delimiter=",", dtype=np.float32)
        test = np.loadtxt(raw / "test" / f"{m}.txt", delimiter=",", dtype=np.float32)
        labels = np.loadtxt(raw / "test_label" / f"{m}.txt", delimiter=",").astype(np.int8)
        d = out / m
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "train.npy", train)
        np.save(d / "test.npy", test)
        np.save(d / "test_labels.npy", labels)
        print(f"{m}: train {train.shape} test {test.shape} anomaly rate {labels.mean():.3%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()
    preprocess_smd(Path(args.raw) / "smd", Path(args.out) / "smd")


if __name__ == "__main__":
    main()

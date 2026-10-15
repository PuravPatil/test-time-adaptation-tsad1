"""Download SMD (Server Machine Dataset) from the OmniAnomaly repo.

    python scripts/download_data.py                 # the machines used in the papers
    python scripts/download_data.py --machines 1-7 2-4
    python scripts/download_data.py --all           # all 28 machines
"""
import argparse
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
# M2N2 uses 1-4 and 2-1; CANDI uses 1-7, 1-8, 2-1, 2-4, 3-2.
DEFAULT = ["1-4", "1-7", "1-8", "2-1", "2-4", "3-2"]
ALL = [f"1-{i}" for i in range(1, 9)] + [f"2-{i}" for i in range(1, 10)] + [f"3-{i}" for i in range(1, 12)]


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--machines", nargs="*", default=DEFAULT)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="data/raw/smd")
    args = ap.parse_args()
    for m in (ALL if args.all else args.machines):
        name = f"machine-{m}.txt"
        for split in ("train", "test", "test_label"):
            fetch(f"{BASE}/{split}/{name}", Path(args.out) / split / name)
    print("done ->", args.out)


if __name__ == "__main__":
    main()

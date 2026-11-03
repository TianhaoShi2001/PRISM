"""Build the Criteo train/valid/test parquet files used by PRISM.

The public file is Criteo Uplift Prediction v2.1 (13,979,592 rows):

    f0..f11, treatment, conversion, visit, exposure

This script keeps the public file's row order and applies data/criteo_split_order.npy.
That array is a permutation of raw row numbers: train, then valid, then test.
The resulting sizes are

    11,183,673 / 1,397,959 / 1,397,960

which is the split behind the released checkpoints. Treatment, visit, and
conversion in the model are the file's treatment, visit, and conversion columns.
"""

import argparse
import gzip
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "data" / "criteo_split_order.npy"
OUT_DIR = ROOT / "data" / "criteo"
COLUMNS = [f"f{i}" for i in range(12)] + ["treatment", "conversion", "visit", "exposure"]
SPLIT_ROWS = (("train", 11183673), ("valid", 1397959), ("test", 1397960))
SOURCE_URL = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"


def resolve_raw(path):
    path = Path(path)
    if path.suffix == ".gz":
        unpacked = path.with_suffix("")
        if not unpacked.exists():
            print(f"unpacking {path}")
            with gzip.open(path, "rb") as src, open(unpacked, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return unpacked
    return path


def download(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {SOURCE_URL}")
    urllib.request.urlretrieve(SOURCE_URL, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description="Prepare the Criteo split used by the released PRISM run.")
    parser.add_argument("--raw", default="", help="Path to criteo-uplift-v2.1.csv or .csv.gz. Downloaded when omitted.")
    args = parser.parse_args()

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Missing split order: {SPLIT_PATH}")
    order = np.load(SPLIT_PATH)
    expected = sum(size for _, size in SPLIT_ROWS)
    if order.shape != (expected,):
        raise ValueError(f"Unexpected split order shape {order.shape}")

    if args.raw:
        raw_path = resolve_raw(args.raw)
    else:
        packed = ROOT / "data" / "criteo-uplift-v2.1.csv.gz"
        if not packed.exists():
            download(packed)
        raw_path = resolve_raw(packed)

    print(f"reading {raw_path}")
    frame = pd.read_csv(raw_path, usecols=COLUMNS)
    if len(frame) != len(order):
        raise ValueError(f"{raw_path} has {len(frame)} rows, split order has {len(order)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = 0
    for name, size in SPLIT_ROWS:
        part = frame.iloc[order[start:start + size]].reset_index(drop=True)
        start += size
        if len(part) != size:
            raise ValueError(f"{name} has {len(part)} rows, expected {size}")
        destination = OUT_DIR / f"{name}.parquet"
        part.to_parquet(destination, index=False)
        print(
            f"{name}: {len(part)} rows, "
            f"P(T=1)={part.treatment.mean():.6f}, "
            f"P(visit=1)={part.visit.mean():.6f}, "
            f"P(conversion=1)={part.conversion.mean():.6f}"
        )

    # Global outcome ATE on every row. Lift@10 / Lift@30 divide by this value,
    # not by the ATE of the split under evaluation.
    t = frame["treatment"].to_numpy()
    y = frame["conversion"].to_numpy()
    population_ate = float(y[t == 1].mean() - y[t == 0].mean())
    ate_path = OUT_DIR / "population_ate.json"
    with open(ate_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "population_ate": population_ate,
                "n": int(len(frame)),
                "definition": "E[conversion|treatment=1] - E[conversion|treatment=0] on all rows",
            },
            f,
            indent=2,
        )
        f.write("\n")
    print(f"population_ate={population_ate:.12f} -> {ate_path}")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()

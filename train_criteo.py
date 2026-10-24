"""Train the released Criteo outcome model from its best_config. No search."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Train the released PRISM checkpoint config on Criteo.")
    parser.add_argument("--seed", type=int, default=42, help="42 is the released checkpoint. 1042 and 2042 are the other reported seeds.")
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--config", default="checkpoints/criteo_seed42/best_config.json")
    args = parser.parse_args()

    if not (ROOT / "data" / "criteo" / "train.parquet").exists():
        raise SystemExit("Missing data/criteo/train.parquet. Run: python data/prepare_criteo.py --raw <criteo-uplift-v2.1.csv>")
    if not (ROOT / args.config).exists():
        raise SystemExit(f"Missing config: {args.config}")

    subprocess.check_call(
        [
            sys.executable, "main.py",
            "--mode", "reproduce",
            "--config_path", args.config,
            "--seed", str(args.seed),
            "--cuda", args.cuda,
        ],
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()

import os
import argparse
from pathlib import Path
import sys
import json

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data_pipeline import get_base_config, get_data_spec
from trainer import train_trial


def set_all_seeds(seed):
    """Seed Python, NumPy, and PyTorch, including every visible GPU."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    print(f"Random seed fixed at {seed} for Python, NumPy, PyTorch, and cuDNN.")


def build_parser():
    parser = argparse.ArgumentParser(description="PRISM training and evaluation entry point.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["eval", "reproduce", "reproduce_eval"],
        default="reproduce",
        help="eval scores a checkpoint; reproduce trains from a released config",
    )
    parser.add_argument("--task", type=str, choices=["train_c", "train_y"], default="train_y")
    parser.add_argument("--model", type=str, default="TARNET", help="TARNET for mediator and PRISM outcome")
    parser.add_argument("--data_name", type=str, default="criteo")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="prism")
    parser.add_argument("--version", type=str, default="prism")
    parser.add_argument("--c_ckpt_path", type=str, default=None, help="Checkpoint of the Stage I mediator")
    parser.add_argument("--eval_ckpt", type=str, default=None, help="Checkpoint to evaluate; eval mode only")
    parser.add_argument("--cuda", type=str, default="0", help="Visible GPU ids")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--num_per_gpu", type=float, default=1, help="GPUs reserved for one run")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_parser().parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    set_all_seeds(args.seed)

    run_hierarchy = f"{args.data_name}/{args.task}/{args.model}/{args.version}/{args.exp_name}"
    print(f"Starting {args.exp_name} | mode {args.mode.upper()} | model {args.model}")
    print(f"Output hierarchy: {run_hierarchy}")
    print("=" * 60)

    root = os.path.abspath(os.path.dirname(__file__))
    base_cfg = get_base_config()
    base_cfg.update({
        "task": args.task,
        "data_name": args.data_name,
        "model": args.model,
        "exp_name": args.exp_name,
        "version": args.version,
        "c_ckpt_path": args.c_ckpt_path,
        "mode": args.mode,
        "run_hierarchy": run_hierarchy,
        "project_root": root,
        "num_workers": args.num_workers,
        "num_per_gpu": args.num_per_gpu,
        "seed": args.seed,
    })

    data_spec = get_data_spec(args.data_name)
    base_cfg["data"] = data_spec
    train_path = os.path.join(data_spec["data_dir"], "train.parquet")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Missing {train_path}. Run: python data/prepare_criteo.py --raw /path/to/criteo-uplift-v2.1.csv"
        )

    if args.mode == "eval":
        ckpt_to_load = args.eval_ckpt or f"./ckpts/{run_hierarchy}/best_model.pth"
        config_to_load = ckpt_to_load.replace("best_model.pth", "best_config.json")
        print(f"Eval mode: loading {ckpt_to_load}")
        if not os.path.exists(config_to_load):
            raise FileNotFoundError(f"Missing paired config: {config_to_load}")
        print(f"Using the paired config {config_to_load}.")
        with open(config_to_load, "r", encoding="utf-8") as f:
            saved_cfg = json.load(f)
        trial_cfg = {**base_cfg, **saved_cfg}
        trial_cfg["eval_ckpt_path"] = ckpt_to_load
        trial_cfg["mode"] = "eval"
        train_trial(trial_cfg)

    elif args.mode in ["reproduce", "reproduce_eval"]:
        if not args.config_path or not os.path.exists(args.config_path):
            raise FileNotFoundError(f"Missing config file: {args.config_path}")

        print(f"\nStarting {args.mode.upper()} from {args.config_path} | seed {args.seed}")
        with open(args.config_path, "r", encoding="utf-8") as f:
            best_cfg = json.load(f)

        trial_cfg = {**base_cfg, **best_cfg}
        trial_cfg["seed"] = args.seed
        trial_cfg["mode"] = args.mode
        train_trial(trial_cfg)
        print(f"Seed {args.seed} {args.mode} finished.")


if __name__ == "__main__":
    main()

import os
# Keep gRPC from forking inside DataLoader workers.
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
# Do not let Ray refuse to start when the disk is nearly full.
os.environ["RAY_DEBUG_DISABLE_USING_FREE_DISK_SPACE_CHECK"] = "1"
os.environ["RAY_local_fs_capacity_threshold"] = "0.99"
os.environ["RAY_RUNTIME_ENV_CREATE_TIMEOUT_PER_RESOURCE_KIB"] = "600"
os.environ["RAY_GCS_MAX_FINISHED_JOBS"] = "100"
os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

# Raise Ray's memory-usage threshold. The monitor itself stays enabled.
os.environ["RAY_memory_usage_threshold"] = "0.995"

import time
import argparse
from pathlib import Path
import sys
import uuid
import shutil
import pandas as pd
import numpy as np
import json

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data_pipeline import get_base_config, get_data_spec
from trainer import train_trial

def set_all_seeds(seed):
    """Seed Python, NumPy, and PyTorch, including every visible GPU."""
    import random
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    # Deterministic cuDNN is left off. Enabling it can reject some operators.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"Random seed fixed at {seed} for Python, NumPy, PyTorch, and cuDNN.")

def get_safe_run_config(exp_name, save_dir):
    """Ray RunConfig, accepting either storage_path or local_dir."""
    try:
        from ray.train import RunConfig
    except ImportError:
        from ray.air.config import RunConfig
    try:
        return RunConfig(name=exp_name, storage_path=save_dir)
    except TypeError:
        return RunConfig(name=exp_name, local_dir=save_dir)

def generate_mock_data(data_dir):
    """Refuse to invent a dataset. The released split is built by data/prepare_criteo.py."""
    train_path = os.path.join(data_dir, "train.parquet")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Missing {train_path}. Run: python data/prepare_criteo.py --raw /path/to/criteo-uplift-v2.1.csv"
        )

def build_parser():
    parser = argparse.ArgumentParser(description="Uplift training and evaluation entry point.")

    parser.add_argument("--mode", type=str, choices=["debug", "tune", "eval", 'reproduce', 'reproduce_eval', 'single_grid'], default="tune",
                        help="debug, tune, eval, reproduce, reproduce_eval, or single_grid")
    parser.add_argument("--task", type=str, choices=["train_c", "train_y"], default="train_y")
    parser.add_argument("--model", type=str, default="TARNET", help="Model name, for example TARNET")
    parser.add_argument("--data_name", type=str, default="criteo")
    parser.add_argument("--config_path", type=str, default=None)

    parser.add_argument("--exp_name", type=str, default="v1_uplift_baseline")
    parser.add_argument("--version", type=str, default="v1_baseline")

    parser.add_argument("--c_ckpt_path", type=str, default=None, help="Checkpoint of the Stage I mediator")
    parser.add_argument("--eval_ckpt", type=str, default=None, help="Checkpoint to evaluate; eval mode only")

    parser.add_argument("--cuda", type=str, default="0", help="Visible GPU ids")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--num_per_gpu", type=float, default=1, help="GPUs reserved for one trial")
    parser.add_argument("--seed", type=int, default=42)

    return parser

def main():
    args = build_parser().parse_args()

    # Isolate the requested GPUs before any CUDA context is created.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    set_all_seeds(args.seed)
    if args.mode in ["debug", "tune"]:
        import ray
        from ray import tune
        print("\n" + "="*60)
        print("Initializing a local Ray instance...")
        init_start_time = time.time()

        tmp_name = f"ray_{os.getuid()}_{uuid.uuid4().hex[:8]}"
        RAY_TMP = os.path.join('/data/shith/ray_tmp', tmp_name)
        os.makedirs(RAY_TMP, exist_ok=True)

        os.environ["RAY_TMPDIR"] = RAY_TMP
        os.environ["TMPDIR"] = RAY_TMP
        os.environ["TEMP"] = RAY_TMP
        os.environ["TMP"] = RAY_TMP



        if not ray.is_initialized():
            ray.init(
                address="local", 
                namespace=tmp_name,
                include_dashboard=False,
                ignore_reinit_error=True,
                _temp_dir=RAY_TMP,
                object_store_memory=20 * 1024 * 1024 * 1024,
            )
            # ray.init('auto')
        init_end_time = time.time()
        print(f"Ray is ready. Temporary directory: {RAY_TMP}")
        print(f"Startup took {init_end_time - init_start_time:.2f} seconds.")
        print("=" * 60 + "\n")

    # data / task / model / version / experiment
    run_hierarchy = f"{args.data_name}/{args.task}/{args.model}/{args.version}/{args.exp_name}"

    print(f"Starting {args.exp_name} | mode {args.mode.upper()} | model {args.model}")
    print(f"Output hierarchy: {run_hierarchy}")
    print("=" * 60)

    project_root = os.path.abspath(os.path.dirname(__file__))

    base_cfg = get_base_config()
    base_cfg.update({
        "task": args.task, "data_name": args.data_name,
        "model": args.model, 
        "exp_name": args.exp_name, "version": args.version,
        "c_ckpt_path": args.c_ckpt_path, "mode": args.mode,
        "run_hierarchy": run_hierarchy,
        "project_root": project_root, 
        "num_workers": args.num_workers,
        "num_per_gpu": args.num_per_gpu,
        "seed": args.seed
    })
    
    data_spec = get_data_spec(args.data_name)
    base_cfg["data"] = data_spec
    generate_mock_data(data_spec["data_dir"])

    if args.mode == "debug":
        print("Debug mode: two Ray Tune epochs, two steps each.")
        from search_space import get_default_hyperparams
        trial_space = {**base_cfg, **get_default_hyperparams(args.task, args.version)}
        trial_space["num_epochs"] = 2
        trial_space["max_steps_per_epoch"] = 2  
        
        save_dir = os.path.abspath(f"./ray_results/debug/{run_hierarchy}")
        tuner = tune.Tuner(
            tune.with_resources(train_trial, resources={"cpu": 1, "gpu": args.num_per_gpu}),
            param_space=trial_space,
            run_config=get_safe_run_config(args.exp_name, save_dir),
            runtime_env={"working_dir": project_root,
                "excludes": ["results", "logs", "ckpts", 'old', 'ablation_logs', '__pycache__'],
                "env_vars": {"RAY_CHDIR_TO_TRIAL_DIR": "0"}
        }
        )
        tuner.fit()
        print("Debug run finished.")
        
    elif args.mode == "tune":
        print("Tune mode: launching the search.")

        # Drop a leftover global-best marker so trials from another experiment cannot win.
        ckpt_dir = os.path.join(project_root, f"ckpts/{run_hierarchy}")
        os.makedirs(ckpt_dir, exist_ok=True)
        global_metric_path = os.path.join(ckpt_dir, "global_best_metric.txt")
        if os.path.exists(global_metric_path):
            os.remove(global_metric_path)
            
        from search_space import get_ray_search_space
        ray_space = get_ray_search_space(args.task, args.version)
        trial_space = {**base_cfg, **ray_space}
        
        save_dir = os.path.abspath(f"./ray_results/tune/{run_hierarchy}")
        tuner = tune.Tuner(
            tune.with_resources(train_trial, resources={"cpu": max(1, args.num_workers), "gpu": args.num_per_gpu}),
            param_space=trial_space,
            tune_config=tune.TuneConfig(
                metric="Target_C_AUUC" if args.task=="train_c" else "Target_Y_AUUC", 
                mode="max", 
                num_samples=1 
            ),
            run_config=get_safe_run_config(args.exp_name, save_dir)
        )
        tuner.fit()
        
        print("\nLoading the best checkpoint for a full evaluation...")
        best_config_path = os.path.join(ckpt_dir, "best_config.json")
        best_ckpt_save_path = os.path.join(ckpt_dir, "best_model.pth")
        
        if os.path.exists(best_config_path) and os.path.exists(best_ckpt_save_path):
            with open(best_config_path, 'r') as f:
                best_cfg = json.load(f)
            eval_cfg = {**base_cfg, **best_cfg, "mode": "eval", "eval_ckpt_path": best_ckpt_save_path}
            train_trial(eval_cfg)
        else:
            print(f"Missing {best_config_path}. No trial wrote a global best checkpoint.")

    elif args.mode == "eval":
        ckpt_to_load = args.eval_ckpt or f"./ckpts/{run_hierarchy}/best_model.pth"
        config_to_load = ckpt_to_load.replace("best_model.pth", "best_config.json")
        print(f"Eval mode: loading {ckpt_to_load}")
        if os.path.exists(config_to_load):
            print(f"Using the paired config {config_to_load}.")
            with open(config_to_load, 'r') as f:
                saved_cfg = json.load(f)
        else:
            print(f"Missing {config_to_load}. Falling back to the default hyperparameters.")
            from search_space import get_default_hyperparams
            saved_cfg = get_default_hyperparams(args.task, args.version)
        trial_cfg = {**base_cfg, **saved_cfg}
        trial_cfg["eval_ckpt_path"] = ckpt_to_load  
        trial_cfg["mode"] = "eval"
        train_trial(trial_cfg)

        
    elif args.mode in ["reproduce", "reproduce_eval"]:
        if not os.path.exists(args.config_path):
            raise FileNotFoundError(f"Missing config file: {args.config_path}")
            
        print(f"\nStarting {args.mode.upper()} from {args.config_path} | seed {args.seed}")
        
        with open(args.config_path, 'r', encoding='utf-8') as f:
            best_cfg = json.load(f)
            
        trial_cfg = {**base_cfg, **best_cfg}
        trial_cfg["seed"] = args.seed
        trial_cfg["mode"] = args.mode
        
        train_trial(trial_cfg)
        
        print(f"Seed {args.seed} {args.mode} finished.")
    elif args.mode == "single_grid":
        from custom_grid_space import ALL_CUSTOM_SPACES

        if args.version not in ALL_CUSTOM_SPACES:
            raise ValueError(f"Unknown single_grid version: {args.version}")
            
        current_config = ALL_CUSTOM_SPACES[args.version]
        
        print("=" * 80)
        print(f"single_grid started with seed {args.seed}.")
        print(f"Experiment directory: {args.exp_name}")
        print(f"Fixed hyperparameters: {current_config}")
        print("=" * 80)
        
        trial_cfg = {**base_cfg, **current_config}
        
        trial_cfg["seed"] = args.seed
        
        # reproduce mode skips Ray reporting and appends /seed_<seed> itself.
        trial_cfg["mode"] = "reproduce"
        
        if f"_seed_{args.seed}" in trial_cfg["run_hierarchy"]:
            trial_cfg["run_hierarchy"] = trial_cfg["run_hierarchy"].replace(f"_seed_{args.seed}", "")
                
        train_trial(trial_cfg)
        
        print(f"single_grid version {args.version}, seed {args.seed}, finished.")

    if args.mode in ["debug", "tune"]:
        try:
            ray.shutdown()
            shutil.rmtree(RAY_TMP, ignore_errors=True)
            print(f"Removed the Ray temporary directory {RAY_TMP}.")
        except Exception as e:
            print(f"Could not remove the Ray temporary directory: {e}")

if __name__ == "__main__":
    main()

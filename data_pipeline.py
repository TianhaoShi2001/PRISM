import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

def get_base_config():
    return {
        "exp_name": "criteo_baseline_run",
        "task": "train_y",               
        "data_name": "criteo",
        "model_type": "tarnet_proposed", 
        "loss_type": "strata_weighted",  
        "c_fusion_mode": "joint_emb",    
        "ray_dir": "./ray_results",
        "best_c_model_path": "./checkpoints/best_c_model.pth", 
        "best_y_model_path": "./checkpoints/best_y_model.pth",
    }

def get_data_spec(data_name: str) -> dict:
    if data_name == "criteo":
        return {
            "name": "criteo",
            "data_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "criteo"), 
            "feature_cols": [f"f{i}" for i in range(12)], 
            "categorical_cols": [],                       
            "categorical_cardinalities": {},
            "treatment_col": "treatment",
            "outcome_col": "conversion",
            "mediator_col": "visit",
            "population_ate": 0.0012,
        }
    elif data_name == "hillstrom":
        return {
            "name": "hillstrom",
            "data_dir": "/NAS/shith/datasets2024/uplift/hillstrom",
            "feature_cols": ["recency", "history_segment_id", "mens", "womens", "zip_code_id", "newbie", "channel_id"],
            "categorical_cols": ["history_segment_id", "zip_code_id", "channel_id"],
            "categorical_cardinalities": {"history_segment_id": 3, "zip_code_id": 3, "channel_id": 3},
            "treatment_col": "treatment",
            "outcome_col": "conversion",
            "mediator_col": "visit",
        }
    else:
        raise ValueError(f"Unknown data_name: {data_name}")

class UpliftDataset(Dataset):
    """Parquet split preloaded as contiguous arrays."""
    def __init__(self, trial_cfg: dict, split: str):
        self.data_spec = trial_cfg["data"]
        self.split = split
        
        data_dir = self.data_spec["data_dir"]
        path_parquet = os.path.join(data_dir, f"{split}.parquet")
        
        if not os.path.exists(path_parquet):
            raise FileNotFoundError(f"Missing data file: {path_parquet}. Run data/prepare_criteo.py first.")
            
        df = pd.read_parquet(path_parquet)

        self.cat_cols = self.data_spec.get("categorical_cols", [])
        self.cont_cols = [c for c in self.data_spec["feature_cols"] if c not in self.cat_cols]

        # Store each column as one contiguous array.
        self.x_cont = df[self.cont_cols].values.astype(np.float32) if len(self.cont_cols) > 0 else None
        self.x_cat = df[self.cat_cols].values.astype(np.int64) if len(self.cat_cols) > 0 else None
        self.t = df[self.data_spec["treatment_col"]].values.astype(np.float32)
        self.y = df[self.data_spec["outcome_col"]].values.astype(np.float32)
        self.c = df[self.data_spec["mediator_col"]].values.astype(np.float32) if self.data_spec.get("mediator_col") else np.zeros(len(df), dtype=np.float32)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        # Slice the preloaded arrays.
        x_cont = torch.from_numpy(self.x_cont[idx]) if self.x_cont is not None else None
        
        if self.x_cat is not None:
            x_cat = {c: torch.tensor(self.x_cat[idx, i]) for i, c in enumerate(self.cat_cols)}
        else:
            x_cat = None

        t = torch.tensor(self.t[idx])
        y = torch.tensor(self.y[idx])
        c = torch.tensor(self.c[idx])

        return x_cont, x_cat, t, y, c

def uplift_collate_fn(batch):
    x_cont_list, x_cat_list, t_list, y_list, c_list = zip(*batch)

    x_cont = torch.stack(x_cont_list) if x_cont_list[0] is not None else None

    if x_cat_list[0] is None:
        x_cat = None
    else:
        x_cat = {key: torch.stack([d[key] for d in x_cat_list]) for key in x_cat_list[0].keys()}

    t = torch.stack(t_list)
    y = torch.stack(y_list)
    c = torch.stack(c_list)

    return x_cont, x_cat, t, y, c
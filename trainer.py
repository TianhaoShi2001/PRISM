import os
import time
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm 

# Automatic mixed precision.
from torch.cuda.amp import autocast, GradScaler

from data_pipeline import UpliftDataset, uplift_collate_fn
from models import *
from losses import *
from evaluator import evaluate_and_dump
import numpy as np
def set_all_seeds(seed):
    """
    Seed Python, NumPy, and PyTorch, including every visible GPU.
    """
    import random
    import torch
    
    # Python and NumPy.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # PyTorch, including every visible GPU.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.benchmark = True
    # Deterministic cuDNN is left off. Enabling it can reject some operators.
    # torch.backends.cudnn.deterministic = True 
    # torch.backends.cudnn.benchmark = False
    
    # torch.use_deterministic_algorithms is left off for the same reason.
    # torch.use_deterministic_algorithms(True, warn_only=True)
    
    print(f"Random seed fixed at {seed} for Python, NumPy, PyTorch, and cuDNN.")

# ==========================================
# Report metrics to Ray during search, and to the console during reproduction.
# ==========================================
def safe_report(metrics, mode="tune"):
    # Reproduction prints metrics and does not call Ray.
    if mode in ["reproduce", "reproduce_eval"]:
        ep = metrics.get("epoch", "?")
        loss = metrics.get("loss", 0.0)
        auuc = metrics.get("Target_Y_AUUC", metrics.get("Target_C_AUUC", 0.0))
        auc = metrics.get("Target_Y_AUC", metrics.get("Target_C_AUC", 0.0))
        print(f"Epoch {ep} | loss {loss:.4f} | AUUC {auuc:.4f} | AUC {auc:.4f}")
        return

    # Search mode reports to Ray.
    try:
        from ray import train
        if train.get_context().get_trial_name(): 
            train.report(metrics)
            return
    except Exception as e:
        print(f"Ray Train report failed: {e}")
        
    try:
        from ray import tune
        tune.report(**metrics)
    except Exception as e:
        print(f"Ray Tune report failed: {e}")

# ==========================================
# Build the network named by config["model"].
# ==========================================
def build_model(config, data_spec, device):
    cont_dim = len(data_spec["feature_cols"]) - len(data_spec.get("categorical_cols", []))
    cat_cards = data_spec.get("categorical_cardinalities", {})
    model_type = config.get("model", "TARNET")
    
    # None keeps each outcome head as one linear layer.
    head_hidden_dims = config.get("head_hidden_dims", None)

    # Outcome TARNet variants load the frozen mediator before the branch below.
    model_c = None
    if config["task"] == "train_y" and model_type in ["TARNET", "TARNET_Baseline_PureV10"]:
        c_path = config.get("c_ckpt_path")
        
        # Fall back to the outcome width if the mediator config is absent.
        c_hidden_dims = config["hidden_dims"]
        c_dropout_rate = config["dropout_rate"]
        
        # Read the mediator width from the config paired with its checkpoint.
        if c_path and os.path.exists(c_path):
            c_config_path = c_path.replace("best_model.pth", "best_config.json")
            if os.path.exists(c_config_path):
                try:
                    with open(c_config_path, 'r') as f:
                        c_cfg = json.load(f)
                        c_hidden_dims = c_cfg.get("hidden_dims", c_hidden_dims)
                        c_dropout_rate = c_cfg.get("dropout_rate", c_dropout_rate)
                except Exception:
                    pass
        
        # Build the mediator on the training device.
        model_c = TARNET_Baseline(
            continuous_dim=cont_dim, 
            categorical_cardinalities=cat_cards, 
            hidden_dims=c_hidden_dims, 
            dropout_rate=c_dropout_rate
        ).to(device)
        
        # Load the mediator weights when the checkpoint exists.
        if c_path and os.path.exists(c_path):
            model_c.load_state_dict(torch.load(c_path, map_location=device))

    # TARNet outcome and mediator variants.
    if model_type == "TARNET":
        if config["task"] == "train_c":
            return TARNET_Baseline(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config["hidden_dims"], dropout_rate=config["dropout_rate"]).to(device)
            
        elif config["task"] == "train_y":
            if config.get("c_fusion_mode") == "moe":
                return TARNET_MoE(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config["hidden_dims"], dropout_rate=config["dropout_rate"], c_model=model_c, head_hidden_dims=head_hidden_dims).to(device)
                
            elif config.get("c_fusion_mode") == "res_moe": 
                return TARNET_Residual_MoE(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config["hidden_dims"], dropout_rate=config["dropout_rate"], c_model=model_c, head_hidden_dims=head_hidden_dims).to(device)
                
            elif config.get("c_fusion_mode") == "v7_truncated_moe":
                return TARNET_V7_Truncated_MoE(
                    continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config["hidden_dims"], dropout_rate=config["dropout_rate"], c_model=model_c,
                    truncation_mode=config.get("truncation_mode", "hard"), truncation_pct=config.get("truncation_pct", 0.05), truncation_temp=config.get("truncation_temp", 10.0), ema_momentum=config.get("ema_momentum", 0.9)
                ).to(device)
                
            elif config.get("c_fusion_mode") == "v8_evolution_moe":
                return TARNET_V8_Evolution_MoE(
                    continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config["hidden_dims"], dropout_rate=config["dropout_rate"], c_model=model_c,
                    v8_scheme=config.get("v8_scheme", 3), shared_emb_dim=config["hidden_dims"][-1], truncation_pct=config.get("truncation_pct", 0.05), truncation_temp=config.get("truncation_temp", 10.0), ema_momentum=config.get("ema_momentum", 0.9),
                    head_hidden_dims=head_hidden_dims,
                    align_temp = config.get('align_temp', 1)
                ).to(device)
                
            elif config.get("c_fusion_mode") == "v11_aligned_moe":
                return TARNET_V11_Aligned_MoE(
                    continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config["hidden_dims"], dropout_rate=config["dropout_rate"], c_model=model_c,
                    v11_scheme=config.get("v11_scheme", 4), align_type=config.get("align_type", "lift"), align_temp=config.get("align_temp", 1.0)
                ).to(device)
                
            elif config.get("c_fusion_mode") == "ours_s4_conflict":
                return TARNET_Ours_S4_Conflict(
                    continuous_dim=cont_dim, categorical_cardinalities=cat_cards, 
                    hidden_dims=config["hidden_dims"], dropout_rate=config["dropout_rate"], 
                    c_model=model_c, embedding_dim=8,
                    head_hidden_dims=head_hidden_dims, 
                    ours_s4_use_stop_grad=config.get("ours_s4_use_stop_grad", False) 
                ).to(device)

                
            elif config.get("c_fusion_mode") == "ours_s6_conflict":
                return TARNET_Ours_S6_Conflict(
                    continuous_dim=cont_dim, categorical_cardinalities=cat_cards, 
                    hidden_dims=config["hidden_dims"], dropout_rate=config["dropout_rate"], 
                    c_model=model_c, embedding_dim=8,
                    ours_s6_temp=config.get("ours_s6_temp", 1.0),
                    head_hidden_dims=head_hidden_dims, 
                    ours_s6_use_stop_grad=config.get("ours_s6_use_stop_grad", False), 
                    ours_s6_use_logit_clamp=config.get("ours_s6_use_logit_clamp", False), 
                    ours_s6_clamp_val=config.get("ours_s6_clamp_val", 2.0)
                ).to(device)
                
            else:
                return TARNET_Proposed(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config["hidden_dims"], dropout_rate=config["dropout_rate"], c_fusion_mode=config["c_fusion_mode"], c_embedding_dim=config.get("c_embedding_dim", 4), c_model=model_c, head_hidden_dims=head_hidden_dims).to(device)

    # Residual mixture that reuses the mediator loaded above.
    elif model_type == "TARNET_Baseline_PureV10":
        return TARNET_Residual_MoE(
            continuous_dim=cont_dim, 
            categorical_cardinalities=cat_cards, 
            hidden_dims=config["hidden_dims"], 
            dropout_rate=config["dropout_rate"], 
            c_model=model_c,
            head_hidden_dims=head_hidden_dims 
        ).to(device)

    # Other published baselines.
    elif model_type == "MTMT":
        return MTMT_STMT(
            continuous_dim=cont_dim, categorical_cardinalities=cat_cards, num_experts=config.get("num_experts", 4), expert_type=config.get("expert_type", "mlp"), expert_hidden_dims=config.get("expert_hidden_dims", [64]), dropout_rate=config.get("dropout_rate", 0.1), t_emb_dim=config.get("t_emb_dim", 16) 
        ).to(device)
    elif model_type == "ECUP":
        return ECUP_Model(
            continuous_dim=cont_dim, categorical_cardinalities=cat_cards, d_dim=config.get("d_dim", 16), gamma=config.get("gamma", 1.0), tower_h=config.get("tower_h", 128), tae_h=config.get("tae_h", 64), num_heads=config.get("num_heads", 2)
        ).to(device)
    elif model_type == "MOTTO":
        return MOTTO_Model(
            continuous_dim=cont_dim, categorical_cardinalities=cat_cards, d_dim=config.get("d_dim", 16), bottom_dim=config.get("bottom_dim", 128), expert_hidden_dims=config.get("expert_hidden_dims", [64, 64]), tower_dim=config.get("tower_dim", 64), use_specific_experts=config.get("use_specific_experts", True)
        ).to(device)
    elif model_type == "TARNET_MT":
        return TARNET_Naive_MT(
            continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config.get("hidden_dims", [128, 64, 32]), dropout_rate=config.get("dropout_rate", 0.0)
        ).to(device)
    elif model_type == "DragonNet":
        return DragonNet(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config.get("hidden_dims", [128, 64])).to(device)
    elif model_type == "EUEN":
        return EUEN(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config.get("hidden_dims", [128, 64])).to(device)
    elif model_type == "EUEN_Academic":
        hidden_dims = config.get("hidden_dims", [128, 64])
        dropout_rate = config.get("dropout_rate", 0.0)
        return EUEN_Academic(
            continuous_dim=cont_dim, 
            categorical_cardinalities=cat_cards, 
            hidden_dims=hidden_dims, 
            dropout_rate=dropout_rate
        ).to(device)
    elif model_type == "EFIN_ours":
        # EFIN embedding width is separate from hidden_dims.
        embed_dim = config.get("efin_embed_dim", 16)
        hidden_dims = config.get("hidden_dims", [128, 64])
        return EFIN_ours(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, 
                             embed_dim=embed_dim, hidden_dims=hidden_dims).to(device)
    elif model_type == "CanniUplift":
        return CanniUplift(
            continuous_dim=cont_dim,
            categorical_cardinalities=cat_cards,
            hidden_dims=config.get("hidden_dims", [128, 64]),
            use_treat_attn=config.get("canniuplift_use_treat_attn", True),
            attn_d_dim=config.get("canniuplift_attn_d_dim", 16),
            attn_num_heads=config.get("canniuplift_attn_num_heads", 2),
        ).to(device)
    elif model_type == "EFIN_0717new":
        return EFIN_0717new(
            continuous_dim=cont_dim,
            categorical_cardinalities=cat_cards,
            hu_dim=config.get("efin_hu_dim", 128),
            hc_dim=config.get("efin_hc_dim", 64),
            is_self=config.get("efin_is_self", False),
            act_type=config.get("efin_act_type", "elu"),
            dropout_rate=config.get("efin_dropout", 0.0),
        ).to(device)

    elif model_type == "ECUP_0717new":
        return ECUP_0717new(
            continuous_dim=cont_dim,
            categorical_cardinalities=cat_cards,
            d_dim=config.get("d_dim", 16),
            gamma=config.get("gamma", 1.0),
            tower_h=config.get("tower_h", 128),
            tae_h=config.get("tae_h", 64),
            num_heads=config.get("num_heads", 2)
        ).to(device)

    elif model_type == "MTMT_0717new":
        return MTMT_0717new(
            continuous_dim=cont_dim,
            categorical_cardinalities=cat_cards,
            num_experts=config.get("num_experts", 4),
            expert_type=config.get("expert_type", "mlp"),
            expert_hidden_dims=config.get("expert_hidden_dims", [64]),
            dropout_rate=config.get("dropout_rate", 0.1),
            t_emb_dim=config.get("t_emb_dim", 16)
        ).to(device)
    elif model_type == "EFIN":
        # This EFIN variant uses efin_embed_dim as its only hidden width.
        efin_rank = config.get("efin_embed_dim", 64)
        
        return EFIN(
            continuous_dim=cont_dim, 
            categorical_cardinalities=cat_cards, 
            efin_rank=efin_rank,
            dropout_rate=config.get("dropout_rate", 0.0)
        ).to(device)
    elif model_type == "DESCN":
        return DESCN(
            continuous_dim=cont_dim, categorical_cardinalities=cat_cards, shared_dims=config.get("hidden_dims", [128]), tower_dims=config.get("hidden_dims", [128])
        ).to(device)
    elif model_type == "S_Learner":
        return S_Learner(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config.get("hidden_dims", [128, 64])).to(device)
    elif model_type == "T_Learner":
        return T_Learner(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config.get("hidden_dims", [128, 64])).to(device)
    elif model_type == "CFRNet":
        return CFRNet(continuous_dim=cont_dim, categorical_cardinalities=cat_cards, hidden_dims=config.get("hidden_dims", [128, 64])).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_type}")

def run_final_evaluation(model, loaders_dict, device, config):
    run_hier = config.get("run_hierarchy", "default_run")
    seed = config.get("seed", 42)
    mode = config.get("mode", "tune")
    
    # Reproduction writes under seed_<seed>.
    if mode in ["reproduce", "reproduce_eval"]:
        run_hier = os.path.join(run_hier, f"seed_{seed}")
        
    project_root = config.get("project_root", ".") 
    res_dir = os.path.join(project_root, f"results/{run_hier}") 
    os.makedirs(res_dir, exist_ok=True)

    eval_max_steps = config.get("max_steps_per_epoch", float('inf')) if mode == "debug" else float('inf')
    
    print("\n" + "="*80)
    print(f"Evaluating the loaded checkpoint. Metrics go to {res_dir}.")
    print("="*80)
    
    final_metrics = {}
    for split_name, loader in loaders_dict.items():
        if loader is None: continue
            
        if split_name in ['train', 'valid']:
            csv_path = None
        else:
            csv_path = os.path.join(res_dir, f"{split_name}_dist.csv")
        metrics, _ = evaluate_and_dump(model, loader, device, config["task"], save_path=csv_path, max_steps=eval_max_steps, show_pbar=True)
        
        for k, v in metrics.items():
            final_metrics[f"{split_name.capitalize()}_{k}"] = v
            
        print(f"[{split_name.upper():<5}] Y_AUUC: {metrics.get('Target_Y_AUUC',0):.4f} | Y_AUC: {metrics.get('Target_Y_AUC',0):.4f} | Y_MAE: {metrics.get('Target_Y_MAE',0):.4f}")
        print(f"        C_AUUC: {metrics.get('Target_C_AUUC',0):.4f} | C_AUC: {metrics.get('Target_C_AUC',0):.4f} | C_MAE: {metrics.get('Target_C_MAE',0):.4f}")

    metrics_save_path = os.path.join(res_dir, "final_metrics.json")
    try:
        with open(metrics_save_path, 'w') as f:
            json.dump(final_metrics, f, ensure_ascii=False, indent=4)
        print(f"\nWrote metrics to {metrics_save_path}.")
    except Exception as e:
        print(f"Could not write metrics: {e}")

    return final_metrics

def train_trial(trial_cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_spec = trial_cfg["data"]
    accumulate_steps = trial_cfg.get("accumulate_steps", 1)
    
    mode = trial_cfg.get("mode", "tune")
    seed = trial_cfg.get("seed", 42)
    set_all_seeds(seed)
    max_steps = trial_cfg.get("max_steps_per_epoch", float('inf'))
    run_hier = trial_cfg.get("run_hierarchy", "default_run")

    if mode in ["reproduce", "reproduce_eval"]:
        run_hier = os.path.join(run_hier, f"seed_{seed}")
        
    eval_max_steps = max_steps if mode == "debug" else float('inf')
    show_pbar = (mode in ["debug", "eval"])
    
    project_root = trial_cfg.get("project_root", ".")
    ckpt_dir = os.path.join(project_root, f"ckpts/{run_hier}")
    res_dir = os.path.join(project_root, f"results/{run_hier}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    best_ckpt_save_path = os.path.join(ckpt_dir, "best_model.pth")
    best_config_save_path = os.path.join(ckpt_dir, "best_config.json")
    global_metric_path = os.path.join(ckpt_dir, "global_best_metric.txt")
    metrics_save_path = os.path.join(res_dir, "final_metrics.json")

    # A finished reproduction seed is not trained again.
    if mode == "reproduce" and os.path.exists(metrics_save_path):
        print(f"Skipping seed {seed}: final_metrics.json already exists.")
        return

    num_workers = trial_cfg.get("num_workers", 0)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        import random
        random.seed(worker_seed)
        
    g = torch.Generator()
    g.manual_seed(seed)
    try:
        train_loader = DataLoader(
            UpliftDataset(trial_cfg, split="train"), 
            batch_size=trial_cfg["batch_size"], 
            shuffle=True, 
            collate_fn=uplift_collate_fn, 
            pin_memory=True, 
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g
        )
        valid_loader = DataLoader(
            UpliftDataset(trial_cfg, split="valid"), 
            batch_size=trial_cfg["batch_size"], 
            shuffle=False, 
            collate_fn=uplift_collate_fn, 
            pin_memory=True, 
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g
        )
        test_loader = DataLoader(
            UpliftDataset(trial_cfg, split="test"), 
            batch_size=trial_cfg["batch_size"], 
            shuffle=False, 
            collate_fn=uplift_collate_fn, 
            pin_memory=True, 
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g
        )
        # train_loader = DataLoader(UpliftDataset(trial_cfg, split="train"), batch_size=trial_cfg["batch_size"], shuffle=True, collate_fn=uplift_collate_fn, pin_memory=True, num_workers=num_workers)
        # valid_loader = DataLoader(UpliftDataset(trial_cfg, split="valid"), batch_size=trial_cfg["batch_size"], shuffle=False, collate_fn=uplift_collate_fn, pin_memory=True, num_workers=num_workers)
        # test_loader = DataLoader(UpliftDataset(trial_cfg, split="test"), batch_size=trial_cfg["batch_size"], shuffle=False, collate_fn=uplift_collate_fn, pin_memory=True, num_workers=num_workers)
    except Exception as e:
        print(f"Could not load the dataset: {e}")
        return
        
    loaders_dict = {"train": train_loader, "valid": valid_loader, "test": test_loader}
    model = build_model(trial_cfg, data_spec, device)
    
    # Evaluation loads a checkpoint and returns.
    if mode == "eval":
        ckpt_path = trial_cfg.get("eval_ckpt_path")
        if not os.path.exists(ckpt_path): return
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
        run_final_evaluation(model, loaders_dict, device, trial_cfg)
        return
        
    if mode == "reproduce_eval":
        if os.path.exists(best_ckpt_save_path):
            print(f"Seed {seed}: loading {best_ckpt_save_path} for evaluation.")
            state = torch.load(best_ckpt_save_path, map_location=device)
            model.load_state_dict(state.get("model_state_dict", state))
            run_final_evaluation(model, loaders_dict, device, trial_cfg)
        else:
            print(f"Seed {seed} has no checkpoint at {best_ckpt_save_path}.")
        return

    dro_criterion = None
    if "group_dro" in trial_cfg.get("loss_types", []):
        dro_criterion = UpliftGroupDROLoss(
            grouping_mode=trial_cfg.get("dro_grouping_mode", "2d_coarse"),
            ng=trial_cfg.get("dro_ng", 0.01),
            ema_gamma=trial_cfg.get("dro_ema", 0.1),
            max_clip=trial_cfg.get("dro_clip", 0.5)
        ).to(device)

    # Training.
    optim_mode = trial_cfg.get('optim_mode', 'adam')
    if optim_mode == 'adam':
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=trial_cfg["learning_rate"], 
                           weight_decay=trial_cfg["weight_decay"])
    elif optim_mode == 'adamw':
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=trial_cfg["learning_rate"], 
                           weight_decay=trial_cfg["weight_decay"])
    focal_criterion = FocalLoss(alpha=trial_cfg.get("focal_alpha", 0.5), gamma=trial_cfg.get("focal_gamma", 0.0))
    scaler = GradScaler()
    

    num_epochs = trial_cfg.get("num_epochs", 100)
    patience = trial_cfg.get("patience", 10) 
    epochs_no_improve = 0
    local_best_metric = -float('inf')

    # Two-stage reweighting is off unless the config turns it on.
    two_stage_mode = trial_cfg.get("conflict_two_stage_mode", False)
    stage1_epochs = trial_cfg.get("conflict_stage1_epochs", 0)
    freeze_base_in_stage2 = trial_cfg.get("conflict_freeze_base_in_stage2", False)

    for epoch in range(1, num_epochs + 1):
        # Stage 1 zeros the three reweighting strengths.
        current_cfg = trial_cfg.copy()
        if two_stage_mode and epoch <= stage1_epochs:
            current_cfg["conflict_alpha_wool"] = 0.0
            current_cfg["conflict_alpha_gold"] = 0.0
            current_cfg["conflict_alpha_walkin"] = 0.0

        # The first stage-2 epoch can freeze the encoder and shared base.
        if two_stage_mode and epoch == (stage1_epochs + 1) and freeze_base_in_stage2:
            print(f"\nEpoch {epoch}: freezing the encoder and shared base.")
            for name, param in model.named_parameters():
                if "encoder" in name or "shared_base" in name:
                    param.requires_grad = False
                    
            # Rebuild the optimizer so frozen parameters drop out of Adam state.
            if optim_mode == 'adam':
                optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], 
                                       lr=trial_cfg["learning_rate"], weight_decay=trial_cfg["weight_decay"])
            elif optim_mode == 'adamw':
                optimizer = optim.AdamW([p for p in model.parameters() if p.parameters() if p.requires_grad], 
                                        lr=trial_cfg["learning_rate"], weight_decay=trial_cfg["weight_decay"])
            print("Rebuilt the optimizer on the parameters that still require gradients.\n")

        train_start = time.time()
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False) if show_pbar else train_loader
        for batch_idx, (x_cont, x_cat, t, y, c) in enumerate(pbar):
            if batch_idx >= max_steps: break 
            x_cont = x_cont.to(device, non_blocking=True) if x_cont is not None else None
            x_cat = {k: v.to(device, non_blocking=True) for k, v in x_cat.items()} if x_cat is not None else None
            t, y, c = t.to(device, non_blocking=True), y.to(device, non_blocking=True), c.to(device, non_blocking=True)
            
            with autocast():
                if current_cfg["task"] == "train_c":
                    c0_pred, c1_pred, z_c = model(x_cont, x_cat)
                    cls_loss = 2 * focal_criterion(torch.where(t == 1, c1_pred, c0_pred), c.float())
                    align_weight = current_cfg.get("align_weight", current_cfg.get("mmd_alpha", 0.1))
                    
                    if align_weight > 0:
                        align_method = current_cfg.get("align_method", "mmd")
                        z_t1, z_t0 = z_c[t == 1], z_c[t == 0]
                        if align_method == "mmd": align_loss = mmd_loss(z_t1, z_t0, sigma=current_cfg.get("mmd_sigma", 1.0))
                        elif align_method == "swd": align_loss = sliced_wasserstein_distance(z_t1, z_t0, num_projections=128)
                        elif align_method == "moment": align_loss = moment_matching_loss(z_t1, z_t0)
                        else: align_loss = torch.tensor(0.0, device=device)
                    else:
                        align_loss = torch.tensor(0.0, device=device)
                    loss = cls_loss + align_weight * align_loss
                           
                elif current_cfg["task"] == "train_y":
                    if current_cfg.get("model") == "ECUP":
                        preds_dict = model(x_cont, x_cat)
                        loss, loss_comp = compute_ecup_loss(preds_dict, y, c, t, current_cfg)
                    elif current_cfg.get("model") == "MTMT":
                        preds_dict = model(x_cont, x_cat)
                        loss, loss_comp = compute_mtmt_loss(
                            preds_dict, y, c, t, preds_dict["pi_dict"], current_cfg, dro_criterion
                        )
                    elif current_cfg.get("model") == "EFIN_0717new":
                        y0_pred, y1_pred, pi_dict = model(x_cont, x_cat)
                        loss, loss_comp = compute_efin_0717new_loss(y0_pred, y1_pred, y, t, pi_dict, current_cfg)
                    elif trial_cfg.get("model") == "CanniUplift":
                        # Visit is the mediator label passed into the CanniUplift loss.
                        y0_pred, y1_pred, pi_dict = model(x_cont, x_cat)
                        loss, loss_comp = compute_canniuplift_loss(y0_pred, y1_pred, y, t, c, pi_dict, trial_cfg)
                    elif current_cfg.get("model") == "ECUP_0717new":
                        preds_dict = model(x_cont, x_cat)
                        loss, loss_comp = compute_ecup_0717new_loss(preds_dict, y, c, t, current_cfg)

                    elif current_cfg.get("model") == "MTMT_0717new":
                        # Same output layout as MTMT, so the MTMT loss is reused.
                        preds_dict = model(x_cont, x_cat)
                        loss, loss_comp = compute_mtmt_loss(
                            preds_dict, y, c, t, preds_dict["pi_dict"], current_cfg , dro_criterion
                        )
                    elif current_cfg.get("model") == "MOTTO":
                        preds_dict = model(x_cont, x_cat)
                        loss, loss_comp = compute_motto_loss(preds_dict, y, c, t, current_cfg)
                    elif current_cfg.get("model") == "DragonNet":
                        y0_pred, y1_pred, pi_dict = model(x_cont, x_cat)
                        loss, loss_comp = compute_dragonnet_loss(y0_pred, y1_pred, y, t, pi_dict, current_cfg)
                    elif current_cfg.get("model") == "EFIN":
                        y0_pred, y1_pred, pi_dict = model(x_cont, x_cat)
                        loss, loss_comp = compute_efin_loss(y0_pred, y1_pred, y, t, pi_dict, current_cfg)
                    elif current_cfg.get("model") == "EFIN_ours":
                        y0_pred, y1_pred, pi_dict = model(x_cont, x_cat)
                        loss, loss_comp = compute_efin_ours_loss(y0_pred, y1_pred, y, t, pi_dict, current_cfg)
                    elif current_cfg.get("model") == "S_Learner":
                        y0_pred, y1_pred, pi_dict = model(x_cont, x_cat, t=t) 
                        loss, _ = compute_stage3_loss(y0_pred, y1_pred, y, t, pi_dict, current_cfg, dro_criterion=dro_criterion)
                    elif current_cfg.get("model") == "TARNET_MT":
                        preds_dict = model(x_cont, x_cat)
                        loss, loss_comp = compute_naive_mt_loss(
                            preds_dict, y, c, t, preds_dict["pi_dict"], current_cfg, dro_criterion
                        )
                    elif current_cfg.get("model") == "CFRNet":
                        y0_pred, y1_pred, pi_dict = model(x_cont, x_cat)
                        loss, loss_comp = compute_stage3_loss(y0_pred, y1_pred, y, t, pi_dict, current_cfg, dro_criterion=dro_criterion)
                        z = pi_dict["z"]
                        z_t1, z_t0 = z[t == 1], z[t == 0]
                        cfr_weight = current_cfg.get("cfr_weight", 0.1)
                        if cfr_weight > 0 and z_t1.size(0) > 0 and z_t0.size(0) > 0:
                            align_loss = sliced_wasserstein_distance(z_t1, z_t0)
                            loss += cfr_weight * align_loss
                            loss_comp["cfr_align_loss"] = (cfr_weight * align_loss).item()
                    elif current_cfg.get("model") == "DESCN":
                        y0_pred, y1_pred, pi_dict = model(x_cont, x_cat)
                        loss, loss_comp = compute_descn_loss(y0_pred, y1_pred, y, t, pi_dict, current_cfg)
                    else:
                        y0_pred, y1_pred, pi_dict = model(x_cont, x_cat)
                        loss, _ = compute_stage3_loss(y0_pred, y1_pred, y, t, pi_dict, current_cfg, dro_criterion=dro_criterion)
            
            scaler.scale(loss / accumulate_steps).backward()
            
            if (batch_idx + 1) % accumulate_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            epoch_loss += loss.item()
        train_time = time.time() - train_start

        val_start = time.time()
        val_metrics, _ = evaluate_and_dump(model, valid_loader, device, current_cfg["task"], max_steps=eval_max_steps, show_pbar=show_pbar)
        test_metrics, _ = evaluate_and_dump(model, test_loader, device, current_cfg["task"], max_steps=eval_max_steps, show_pbar=False)
        val_time = time.time() - val_start
        
        prefix = "Target_C" if current_cfg["task"] == "train_c" else "Target_Y"
        target_metric = val_metrics.get(f"{prefix}_AUUC", 0)

        is_local_best = False
        if target_metric > local_best_metric:
            local_best_metric = target_metric
            local_best_val_metrics = val_metrics
            local_best_test_metrics = test_metrics
            best_epoch = epoch
            is_local_best = True
            epochs_no_improve = 0  
        else:
            epochs_no_improve += 1

        is_best = ""
        global_best = -float('inf')
        if os.path.exists(global_metric_path):
            try:
                with open(global_metric_path, 'r') as f:
                    content = f.read().strip()
                    if content: global_best = float(content)
            except Exception:
                pass 
            
        if target_metric > global_best:
            try:
                with open(global_metric_path, 'w') as f:
                    f.write(str(target_metric))
                torch.save(model.state_dict(), best_ckpt_save_path)
                safe_cfg = {k: v for k, v in current_cfg.items() if isinstance(v, (int, float, str, list, bool)) or v is None}
                with open(best_config_save_path, 'w') as f:
                    json.dump(safe_cfg, f, ensure_ascii=False, indent=4)
                is_best = "GLOBAL"
            except Exception:
                pass
            
        report_dict = {"epoch": epoch, "loss": epoch_loss / max(1, min(max_steps, len(train_loader))), **val_metrics}
        report_dict.update({f"local_best_valid_{k}": v for k, v in local_best_val_metrics.items()})
        report_dict.update({f"test_{k}": v for k, v in test_metrics.items()})
        report_dict.update({f"local_best_test_{k}": v for k, v in local_best_test_metrics.items()})
        report_dict.update({"best_epoch": best_epoch})

        safe_report(report_dict, mode=mode)
        
        mv, mt, p = val_metrics, test_metrics, prefix
        print(f"Ep {epoch:<2} {is_best:<10} | T:{train_time:.1f}s V+Test:{val_time:.1f}s")
        print(f"   [VAL ] AUUC:{mv.get(f'{p}_AUUC',0):.4f} | Qini:{mv.get(f'{p}_Qini',0):.4f} | L10:{mv.get(f'{p}_Lift@10',0):.4f} | AUC:{mv.get(f'{p}_AUC',0):.4f}")
        print(f"   [TEST] AUUC:{mt.get(f'{p}_AUUC',0):.4f} | Qini:{mt.get(f'{p}_Qini',0):.4f} | L10:{mt.get(f'{p}_Lift@10',0):.4f} | AUC:{mt.get(f'{p}_AUC',0):.4f}")
        print("-" * 60)

        if epochs_no_improve >= patience:
            print(f"Early stopping: validation AUUC did not improve for {patience} epochs.")
            break
    
    if mode == "reproduce":
        print(f"Training finished. Loading the best checkpoint for seed {seed}.")
        if os.path.exists(best_ckpt_save_path):
            state = torch.load(best_ckpt_save_path, map_location=device)
            model.load_state_dict(state.get("model_state_dict", state))
            print("Loaded the best checkpoint for the final evaluation.")
        run_final_evaluation(model, loaders_dict, device, trial_cfg)


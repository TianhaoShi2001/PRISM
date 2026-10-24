import torch
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from data_pipeline import get_data_spec
from tqdm import tqdm

def get_perfect_uplift_order(y_true, t):
    """Rank that puts treated responders first and control responders last.

    Ties are broken by a small uniform perturbation so the resulting order is
    not an artifact of stable sorting. This helper is not used by AUUC.
    """
    score = np.zeros_like(y_true, dtype=float)
    score[(t == 1) & (y_true == 1)] = 4
    score[(t == 0) & (y_true == 0)] = 3
    score[(t == 1) & (y_true == 0)] = 2
    score[(t == 0) & (y_true == 1)] = 1
    score += np.random.uniform(0, 0.1, size=len(score))
    return np.argsort(score)[::-1]

def calculate_uplift_metrics(target_true, uplift_pred, t, prefix="Target", ate=None):
    """AUUC, Qini, and Lift as reported in the paper.

    Sort by predicted uplift. On each prefix, the uplift curve is
    (mean treated outcome - mean control outcome) times the prefix size,
    and the Qini curve is Y_t - Y_c * n_t / n_c. Both curves are scaled
    by population fraction on the horizontal axis and by their full-sample
    total on the vertical axis, so each curve ends at 1.

    AUUC is the area under the scaled uplift curve. Qini is the area under
    the scaled Qini curve minus 1/2. Lift@10 and Lift@30 are the treated-minus-control
    outcome rate in the top 10% and 30%, divided by ``ate``. For Criteo outcomes
    that value is the paper's full-population ATE, +0.0012, not the ATE of the
    split being scored.
    """
    target_true, uplift_pred, t = np.array(target_true), np.array(uplift_pred), np.array(t)
    if len(target_true) == 0: return {}

    order = np.argsort(uplift_pred)[::-1]
    y_true_sorted = target_true[order]
    t_sorted = t[order]

    n_t_cum = np.cumsum(t_sorted == 1)
    n_c_cum = np.cumsum(t_sorted == 0)
    y_t_cum = np.cumsum(y_true_sorted * (t_sorted == 1))
    y_c_cum = np.cumsum(y_true_sorted * (t_sorted == 0))

    n_t_safe = np.where(n_t_cum == 0, 1e-6, n_t_cum)
    n_c_safe = np.where(n_c_cum == 0, 1e-6, n_c_cum)

    x_axis = np.arange(1, len(target_true) + 1) / len(target_true)

    with np.errstate(invalid='ignore', divide='ignore'):
        uplift_curve = (y_t_cum / n_t_safe - y_c_cum / n_c_safe) * (n_t_cum + n_c_cum)
    uplift_curve = np.nan_to_num(uplift_curve)

    if abs(uplift_curve[-1]) > 1e-10:
        uplift_curve = uplift_curve / uplift_curve[-1]

    trapz = np.trapz if hasattr(np, "trapz") else np.trapezoid
    auuc = trapz(uplift_curve, x=x_axis)

    with np.errstate(invalid='ignore', divide='ignore'):
        qini_curve = y_t_cum - (y_c_cum * n_t_safe / n_c_safe)
    qini_curve = np.nan_to_num(qini_curve)

    if abs(qini_curve[-1]) > 1e-10:
        qini_curve = qini_curve / qini_curve[-1]

    auqc = trapz(qini_curve, x=x_axis)

    if ate is None:
        n_treated = np.sum(t == 1)
        n_control = np.sum(t == 0)
        if n_treated > 0 and n_control > 0:
            ate = float(np.sum(target_true * (t == 1)) / n_treated - np.sum(target_true * (t == 0)) / n_control)
        else:
            ate = 0.0
    else:
        ate = float(ate)

    def get_lift_at_k(k):
        idx = int(len(target_true) * k / 100.0)
        if idx == 0: return 0.0
        cr_t = np.sum(y_true_sorted[:idx] * (t_sorted[:idx] == 1)) / (np.sum(t_sorted[:idx] == 1) + 1e-6)
        cr_c = np.sum(y_true_sorted[:idx] * (t_sorted[:idx] == 0)) / (np.sum(t_sorted[:idx] == 0) + 1e-6)
        if abs(ate) < 1e-12:
            return 0.0
        return (cr_t - cr_c) / ate

    return {
        f"{prefix}_AUUC": float(auuc),
        f"{prefix}_Qini": float(auqc - 0.5),
        f"{prefix}_Lift@10": float(get_lift_at_k(10)),
        f"{prefix}_Lift@30": float(get_lift_at_k(30))
    }

def calc_class_and_calib(target_true, y_prob, prefix="Target"):
    """Classification and calibration summaries for an observed outcome."""
    if len(target_true) == 0: return {}
    target_true, y_prob = np.array(target_true), np.array(y_prob)
    preds = (y_prob > 0.5).astype(int)
    
    metrics = {
        f"{prefix}_PCOC": float(y_prob.mean() / (target_true.mean() + 1e-6)),
        f"{prefix}_MAE": float(np.abs(y_prob - target_true).mean()),
        f"{prefix}_Recall": float(recall_score(target_true, preds, zero_division=0)),
        f"{prefix}_Precision": float(precision_score(target_true, preds, zero_division=0)),
        f"{prefix}_AUC": float(roc_auc_score(target_true, y_prob) if len(np.unique(target_true)) > 1 else 0.5)
    }
    return metrics



@torch.no_grad()
def evaluate_and_dump(model, loader, device, task_name, save_path=None, max_steps=float('inf'), show_pbar=False):
    model.eval()
    
    t_all, y_all, c_all = [], [], []
    prob0_all, prob1_all, pi_01_all = [], [], []
    c0_prob_all, c1_prob_all = [], [] 
    
    white_box_collections = {}

    pbar_loader = tqdm(loader, desc="Evaluating", leave=False) if show_pbar else loader
    
    for i, batch in enumerate(pbar_loader):
        if i >= max_steps: break
            
        # Batches have four tensors, or five when the mediator label is included.
        x_cont, x_cat, t, y = batch[0], batch[1], batch[2], batch[3]
        c = batch[4] if len(batch) == 5 else torch.zeros_like(y)
            
        x_cont = x_cont.to(device) if x_cont is not None else None
        x_cat = {k: v.to(device) for k, v in x_cat.items()} if x_cat is not None else None
        
        out = model(x_cont, x_cat)
        pi_dict = {}

        if task_name == "train_y":
            pi_dict = out.get("pi_dict", {}) if isinstance(out, dict) else (out[2] if len(out) > 2 else {})

            if isinstance(out, dict) and out.get("ecup_0717new"):
                # Apply sigmoid once. The probabilities are already pCTR * pCVR.
                c0_logit, c1_logit = out["c_logits"]
                cvr0_logit, cvr1_logit = out["cvr_logits"]
                
                c_out_0 = torch.sigmoid(c0_logit)
                c_out_1 = torch.sigmoid(c1_logit)
                
                out_0 = c_out_0 * torch.sigmoid(cvr0_logit)
                out_1 = c_out_1 * torch.sigmoid(cvr1_logit)
                
                c0_prob_all.append(c_out_0.cpu().numpy())
                c1_prob_all.append(c_out_1.cpu().numpy())

            elif isinstance(out, dict) and "c_logits" in out:
                c0_logit, c1_logit = out["c_logits"]
                c_out_0, c_out_1 = torch.sigmoid(c0_logit), torch.sigmoid(c1_logit)
                out_0 = c_out_0 * torch.sigmoid(out["cvr_logits"][0])
                out_1 = c_out_1 * torch.sigmoid(out["cvr_logits"][1])
                c0_prob_all.append(c_out_0.cpu().numpy())
                c1_prob_all.append(c_out_1.cpu().numpy())
                
            elif isinstance(out, dict) and "main_task" in out:
                out_0, out_1 = out["main_task"]
                if "aux_task" in out:
                    c0_logit, c1_logit = out["aux_task"]
                    c0_prob_all.append(torch.sigmoid(c0_logit).cpu().numpy())
                    c1_prob_all.append(torch.sigmoid(c1_logit).cpu().numpy())
                    
            else:
                out_0, out_1 = out[0], out[1]

            pi_01 = pi_dict.get("p_complier", torch.zeros_like(out_0)).detach().cpu().numpy()

        else:
            out_0, out_1 = out[0], out[1]
            pi_01 = np.zeros(len(t))

        # Keep auxiliary tensors whose names start with wb_.
        for k, v in pi_dict.items():
            if str(k).startswith("wb_"):
                if k not in white_box_collections:
                    white_box_collections[k] = []
                white_box_collections[k].append(v.detach().cpu().numpy())

        prob0_all.append(torch.sigmoid(out_0).cpu().numpy())
        prob1_all.append(torch.sigmoid(out_1).cpu().numpy())
        pi_01_all.append(pi_01)
        t_all.append(t.cpu().numpy())
        y_all.append(y.cpu().numpy())
        c_all.append(c.cpu().numpy())
        
    if len(t_all) == 0: return {}, pd.DataFrame()

    t_np = np.concatenate(t_all)
    y_np = np.concatenate(y_all)
    c_np = np.concatenate(c_all)
    prob0_np = np.concatenate(prob0_all)
    prob1_np = np.concatenate(prob1_all)
    pi_01_np = np.concatenate(pi_01_all)
    uplift_pred = prob1_np - prob0_np
    observed_prob = np.where(t_np == 1, prob1_np, prob0_np)
    
    metrics = {}
    outcome_ate = get_data_spec("criteo")["population_ate"]
    metrics.update(calculate_uplift_metrics(y_np, uplift_pred, t_np, prefix="Target_Y", ate=outcome_ate))
    metrics.update(calc_class_and_calib(y_np, observed_prob, prefix="Target_Y"))
    metrics.update(calculate_uplift_metrics(c_np, uplift_pred, t_np, prefix="Target_C"))
    metrics.update(calc_class_and_calib(c_np, observed_prob, prefix="Target_C"))

    df = pd.DataFrame({
        "t": t_np, "y_true": y_np, "c_true": c_np,
        "y0_prob": prob0_np, "y1_prob": prob1_np,
        "uplift_pred": uplift_pred, "pi_01_prior": pi_01_np
    })
    
    if len(c0_prob_all) > 0:
        c0_np = np.concatenate(c0_prob_all)
        c1_np = np.concatenate(c1_prob_all)
        df["c0_prob"] = c0_np
        df["c1_prob"] = c1_np
        df["c_uplift_pred"] = c1_np - c0_np

    embeddings_dict = {}
    
    for k, v_list in white_box_collections.items():
        stacked_v = np.concatenate(v_list, axis=0)
        
        # Scalars and short vectors go in the table. Wider arrays go in the npz.
        if stacked_v.ndim == 1:
            df[k] = stacked_v
        elif stacked_v.ndim == 2 and stacked_v.shape[1] <= 10:
            for dim in range(stacked_v.shape[1]):
                df[f"{k}_dim{dim}"] = stacked_v[:, dim]
        else:
            embeddings_dict[k] = stacked_v

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

        if embeddings_dict:
            npy_path = save_path.replace(".csv", "_embeddings.npz")
            np.savez_compressed(npy_path, **embeddings_dict)
            print(f"Saved auxiliary arrays to {npy_path}")
        
    return metrics, df

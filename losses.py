import torch
import torch.nn as nn
import torch.nn.functional as F

from prism import prior_guided_weights

# ==========================================
# Stage 1: mediator loss.
# ==========================================

class FocalLoss(nn.Module):
    """Down-weights easy examples. With gamma = 0 this is weighted binary cross-entropy."""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        targets = targets.type_as(logits)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = focal_weight * (1 - p_t) ** self.gamma * bce_loss
        
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss

def mmd_loss(x, y, kernel="rbf", sigma=1.0):
    """
    Maximum mean discrepancy between two representation sets.
    The two sets may have different numbers of rows.
    """
    if x.size(0) == 0 or y.size(0) == 0:
        return torch.tensor(0.0, device=x.device, requires_grad=True)
    
    # xx: (N1, N1), yy: (N2, N2), zz: (N1, N2)
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    
    # Squared row norms, shaped for broadcasting.
    rx = xx.diag().unsqueeze(1)  # Shape (N1, 1).
    ry = yy.diag().unsqueeze(0)  # Shape (1, N2).
    
    if kernel == "rbf":
        # Pairwise squared distances.
        # rx + rx.t() has shape (N1, N1).
        dxx = rx + rx.t() - 2. * xx
        # ry.t() + ry has shape (N2, N2).
        dyy = ry.t() + ry - 2. * yy
        # rx + ry has shape (N1, N2).
        dxy = rx + ry - 2. * zz
        
        K_xx = torch.exp(-dxx / (2.0 * sigma ** 2))
        K_yy = torch.exp(-dyy / (2.0 * sigma ** 2))
        K_xy = torch.exp(-dxy / (2.0 * sigma ** 2))
    else: 
        K_xx, K_yy, K_xy = xx, yy, zz
        
    return K_xx.mean() + K_yy.mean() - 2. * K_xy.mean()

def sliced_wasserstein_distance(z1, z2, num_projections=128, p=2):
    """
    Sliced Wasserstein distance. The two batches may have different lengths.
    z1: treated representations, shape (N1, dim).
    z2: control representations, shape (N2, dim).
    """
    # An empty group contributes zero instead of NaN.
    if z1.size(0) == 0 or z2.size(0) == 0:
        return torch.tensor(0.0, device=z1.device if z1.device.type != 'cpu' else z2.device)

    dim = z1.size(1)
    device = z1.device

    # Random directions on the unit sphere. Normalize over the feature axis.
    projections = torch.randn(dim, num_projections, device=device)
    projections = projections / torch.norm(projections, p=2, dim=0, keepdim=True)

    # Project onto each direction.
    # proj1 shape: (N1, num_projections)
    # proj2 shape: (N2, num_projections)
    proj1 = torch.matmul(z1, projections)
    proj2 = torch.matmul(z2, projections)

    # Sort each projected line independently.
    proj1_sorted, _ = torch.sort(proj1, dim=0)
    proj2_sorted, _ = torch.sort(proj2, dim=0)

    # Match unequal sample sizes before the distance.
    N1, N2 = proj1.size(0), proj2.size(0)
    if N1 != N2:
        # interpolate expects (batch, channels, length).
        proj1_sorted = proj1_sorted.t().unsqueeze(0)
        proj2_sorted = proj2_sorted.t().unsqueeze(0)

        # Stretch the shorter empirical quantile function to the longer one.
        target_size = max(N1, N2)
        
        # align_corners=True requires length > 1.
        align_corners = True if (N1 > 1 and N2 > 1) else False

        proj1_sorted = F.interpolate(proj1_sorted, size=target_size, mode='linear', align_corners=align_corners)
        proj2_sorted = F.interpolate(proj2_sorted, size=target_size, mode='linear', align_corners=align_corners)

        # Back to (target_size, num_projections).
        proj1_sorted = proj1_sorted.squeeze(0).t()
        proj2_sorted = proj2_sorted.squeeze(0).t()

    # Mean p-power. For p = 2 this is W_2 squared, which is more stable than W_2.
    wasserstein_distance = torch.pow(torch.abs(proj1_sorted - proj2_sorted), p).mean()

    return wasserstein_distance

def moment_matching_loss(source_features, target_features):
    """
    Match the mean and the covariance. No extra scale hyperparameter.
    """
    if source_features.size(0) <= 1 or target_features.size(0) <= 1:
        return torch.tensor(0.0, device=source_features.device, requires_grad=True)
        
    # First moment.
    mean_source = source_features.mean(dim=0)
    mean_target = target_features.mean(dim=0)
    mean_loss = torch.mean(torch.pow(mean_source - mean_target, 2))
    
    # Second moment.
    def compute_covariance(x):
        batch_size = x.size(0)
        x_centered = x - x.mean(dim=0, keepdim=True)
        return torch.matmul(x_centered.t(), x_centered) / (batch_size - 1 + 1e-5)
        
    cov_source = compute_covariance(source_features)
    cov_target = compute_covariance(target_features)
    cov_loss = torch.mean(torch.pow(cov_source - cov_target, 2))
    
    return mean_loss + cov_loss


class UpliftGroupDROLoss(nn.Module):
    def __init__(self, grouping_mode="2d_coarse", ng=0.01, ema_gamma=0.1, max_clip=0.5):
        """
        :param grouping_mode:
            "1d_coarse" -> four pi_01 bins: 0-5%, 5-10%, 10-20%, 20-100%
            "2d_coarse" -> those four bins crossed with four (T, Y) cells, 16 groups
            "1d_fine"   -> ten pi_01 cuts
            "2d_fine"   -> the fine bins crossed with four (T, Y) cells
        """
        super().__init__()
        self.grouping_mode = grouping_mode
        self.ng = ng
        self.ema_gamma = ema_gamma
        self.max_clip = max_clip
        
        if grouping_mode == "1d_coarse": self.num_groups = 4
        elif grouping_mode == "2d_coarse": self.num_groups = 16
        elif grouping_mode == "1d_fine": self.num_groups = 11
        elif grouping_mode == "2d_fine": self.num_groups = 44
        else: raise ValueError(f"Unknown grouping mode: {grouping_mode}")
        
        # Weights persist across batches.
        self.register_buffer("group_weights", torch.ones(self.num_groups) / self.num_groups)
        self.register_buffer("ema_loss", torch.zeros(self.num_groups))
        self.register_buffer("ema_initialized", torch.zeros(self.num_groups, dtype=torch.bool))

    def forward(self, bce_losses, pi_01, t, y):
        pi_01 = pi_01.detach().float()
        
        # Quantile bins.
        if "coarse" in self.grouping_mode:
            # Four bins: 0-5%, 5-10%, 10-20%, 20-100%.
            boundaries = torch.quantile(pi_01, torch.tensor([0.80, 0.90, 0.95], device=pi_01.device))
            # bucketize gives 0 below 80%, then 1, 2, and 3. Subtract from 3 so the top 5% is bin 0.
            pi_bins = 3 - torch.bucketize(pi_01, boundaries)
        else:
            # Eleven bins. Cuts: 10, 20, 30, 40, 50, 60, 70, 80, 90, 95 percent.
            fine_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
            boundaries = torch.quantile(pi_01, torch.tensor(fine_quantiles, device=pi_01.device))
            
            # bucketize returns 0..10. Subtract from 10 so values above 95% land in bin 0.
            pi_bins = 10 - torch.bucketize(pi_01, boundaries)
            # Bin 0: 0-5%. Bin 1: 5-10%. Bin 2: 10-20%. Then 10-point bins through 90-100%.
            
        # Optional cross with the observed (T, Y) cell.
        if "2d" in self.grouping_mode:
            causal_state = (t.long() * 2 + y.long())  # Cells 0, 1, 2, 3.
            num_pi_bins = 4 if "coarse" in self.grouping_mode else 10
            group_idx = pi_bins * 4 + causal_state
        else:
            group_idx = pi_bins
            
        group_idx = group_idx.long()

        # Per-group mean loss in this batch.
        group_counts = torch.bincount(group_idx, minlength=self.num_groups).float()
        group_loss_sum = torch.zeros(self.num_groups, device=bce_losses.device)
        group_loss_sum.scatter_add_(0, group_idx, bce_losses)
        
        valid_groups = group_counts > 0
        current_group_loss = torch.zeros_like(self.ema_loss)
        current_group_loss[valid_groups] = group_loss_sum[valid_groups] / group_counts[valid_groups]

        # Group-weight update. No gradient flows through this block.
        with torch.no_grad():
            # Exponential moving average.
            first_time = (~self.ema_initialized) & valid_groups
            if first_time.any():
                self.ema_loss[first_time] = current_group_loss[first_time]
                self.ema_initialized[first_time] = True
            
            self.ema_loss[valid_groups] = (1 - self.ema_gamma) * self.ema_loss[valid_groups] + self.ema_gamma * current_group_loss[valid_groups]
            
            # Exponential reweighting.
            new_weights = self.group_weights * torch.exp(self.ng * self.ema_loss)
            
            # Cap one group so it cannot take the whole weight mass.
            new_weights = torch.clamp(new_weights, max=self.max_clip)
            self.group_weights = new_weights / new_weights.sum()

        # Sample weight = group weight / group count.
        sample_weights = self.group_weights[group_idx] / torch.clamp(group_counts[group_idx], min=1.0)
        final_dro_loss = torch.sum(sample_weights * bce_losses)
        
        return final_dro_loss


# ==========================================
# Stage 3: outcome loss.
# ==========================================
def compute_stage3_loss(y0_pred, y1_pred, targets, treatment, pi_dict, config, dro_criterion=None):
    loss_types = config.get("loss_types", ["bce"])
    y_pred_observed = torch.where(treatment == 1, y1_pred, y0_pred)
    bce_raw = F.binary_cross_entropy_with_logits(y_pred_observed, targets.float(), reduction='none')
    
    total_loss = 0.0
    loss_components = {} 
    
    # ---------------------------------------------------------
    # Stratum-weighted binary cross-entropy.
    # A larger complier probability increases the weight on the observed outcome.
    # ---------------------------------------------------------
    if "strata_weighted" in loss_types:
        # Floor the complier weight so a batch of zeros stays defined.
        weight_c = pi_dict["p_complier"].detach() + 1e-4
        
        # Match the scale of an unweighted batch mean.
        weight_c = weight_c / weight_c.mean() 
        sw_loss = (bce_raw * weight_c).mean()
        
        total_loss += sw_loss
        loss_components["sw_loss"] = sw_loss.item()
    
    elif "group_dro" in loss_types and dro_criterion is not None:
        dro_loss = dro_criterion(bce_raw, pi_dict["p_complier"], treatment, targets)
        total_loss += dro_loss
        loss_components["dro_loss"] = dro_loss.item()
    
    elif "prior_conflict" in loss_types:
        weights = prior_guided_weights(
            targets, treatment, pi_dict["p_complier"], y_pred_observed, bce_raw, config,
        )
        conflict_loss = (bce_raw * weights).mean()
        total_loss += conflict_loss
        loss_components["conflict_loss"] = conflict_loss.item()
    else:
        bce_loss = bce_raw.mean()
        total_loss += bce_loss
        loss_components["bce_loss"] = bce_loss.item()

    # ---------------------------------------------------------
    # Variance penalty.
    # Never-takers and always-takers are pushed toward zero uplift.
    # ---------------------------------------------------------
    if "variance_reg" in loss_types:
        # Penalty weight comes from the stratum probabilities.
        p_never = pi_dict["p_never"].detach()
        p_always = pi_dict["p_always"].detach()
        
        # Combined probability of the two strata whose outcome should not move with treatment.
        weight_v = (p_never + p_always) 
        
        # Squared predicted uplift.
        uplift_variance = torch.pow(y1_pred - y0_pred, 2)
        
        # Larger stratum probability, larger penalty.
        var_penalty = (uplift_variance * weight_v).mean()
        
        var_lambda = config.get("var_lambda", 1.0)
        total_loss += var_lambda * var_penalty
        loss_components["var_loss"] = (var_lambda * var_penalty).item()

    # ---------------------------------------------------------
    # Pairwise ranking constraint.
    # ---------------------------------------------------------
    if "pairwise" in loss_types:
        uplift_pred = y1_pred - y0_pred
        pi_01 = pi_dict["p_complier"].detach()
        q_high = torch.quantile(pi_01, 0.75) 
        q_low = torch.quantile(pi_01, 0.25)  
        
        high_mask = pi_01 >= q_high
        low_mask = pi_01 <= q_low
        
        if high_mask.any() and low_mask.any():
            u_high = uplift_pred[high_mask].mean()
            u_low = uplift_pred[low_mask].mean()
            margin = config.get("rank_margin", 0.05)
            rank_loss = F.relu(-(u_high - u_low) + margin)
            rank_alpha = config.get("rank_alpha", 1.0)
            total_loss += rank_alpha * rank_loss
            loss_components["rank_loss"] = (rank_alpha * rank_loss).item()
            


    return total_loss, loss_components


def compute_mtmt_loss(preds_dict, y, c, t, pi_dict, config, dro_criterion=None):
    """
    Multi-task loss for MTMT.
    The main head predicts the outcome and the auxiliary head predicts the mediator.
    """
    # Main head: outcome.
    y0_main, y1_main = preds_dict["main_task"]
    # Auxiliary head: mediator.
    c0_aux, c1_aux = preds_dict["aux_task"]
    
    # Same outcome loss on each head.
    loss_main, comp_main = compute_stage3_loss(
        y0_main, y1_main, y, t, pi_dict, config, dro_criterion
    )
    
    loss_aux, comp_aux = compute_stage3_loss(
        c0_aux, c1_aux, c, t, pi_dict, config, dro_criterion
    )
    
    # aux_weight is read from the config.
    aux_weight = config.get("aux_weight", 0.5)
    total_loss = loss_main + aux_weight * loss_aux
    
    # Prefix the two heads so their terms stay distinct in the log.
    loss_components = {f"main_{k}": v for k, v in comp_main.items()}
    loss_components.update({f"aux_{k}": v for k, v in comp_aux.items()})
    
    return total_loss, loss_components


def compute_ecup_loss(preds_dict, y, c, t, config):
    """
    Entire-chain uplift loss.
    The product term is computed in a form that stays finite under mixed precision.
    """
    # Logits.
    c0_logit, c1_logit = preds_dict["c_logits"]
    cvr0_logit, cvr1_logit = preds_dict["cvr_logits"]
    
    # Logit on the observed treatment arm.
    c_logit_obs = torch.where(t == 1, c1_logit, c0_logit)
    cvr_logit_obs = torch.where(t == 1, cvr1_logit, cvr0_logit)
    
    # ==========================================
    # Click loss. binary_cross_entropy_with_logits stays finite under autocast.
    # ==========================================
    loss_c = F.binary_cross_entropy_with_logits(c_logit_obs, c.float(), reduction='mean')
    
    # ==========================================
    # Conversion loss. The click-conversion product has no logits form,
    # so the cross-entropy is written out and the probability is clamped.
    # ==========================================
    p_c = torch.sigmoid(c_logit_obs)
    p_cvr = torch.sigmoid(cvr_logit_obs)
    
    # p(click and conversion) = p(click) * p(conversion | click).
    p_y = p_c * p_cvr 
    
    # Keep the probability away from 0 and 1 before the logarithm.
    p_y = torch.clamp(p_y, min=1e-7, max=1.0 - 1e-7)
    
    # Binary cross-entropy, written without the fused kernel.
    loss_y = -(y.float() * torch.log(p_y) + (1.0 - y.float()) * torch.log(1.0 - p_y)).mean()
    
    # Combined objective.
    ctcvr_weight = config.get("ctcvr_weight", 1.0)
    total_loss = loss_c + ctcvr_weight * loss_y
    
    loss_components = {
        "ecup_ctr_loss": loss_c.item(),
        "ecup_ctcvr_loss": loss_y.item()
    }
    
    return total_loss, loss_components


# ==========================================
# Naive multi-task loss.
# ==========================================
def compute_naive_mt_loss(preds_dict, y, c, t, pi_dict, config, dro_criterion=None):
    """
    Outcome head uses the configured stage-3 loss.
    Mediator head uses binary cross-entropy only.
    """
    # Outcome loss.
    y0_main, y1_main = preds_dict["main_task"]
    loss_y, comp_y = compute_stage3_loss(y0_main, y1_main, y, t, pi_dict, config, dro_criterion)
    
    # Mediator loss. This head predicts the observed mediator, not the complier label.
    c0_aux, c1_aux = preds_dict["aux_task"]
    # Plain binary cross-entropy, without the outcome ranking or variance terms.
    c_pred_observed = torch.where(t == 1, c1_aux, c0_aux)
    loss_c = F.binary_cross_entropy_with_logits(c_pred_observed, c.float(), reduction='mean')
    
    # Fixed combination. The default mediator weight is 0.5.
    mt_c_weight = config.get("mt_c_weight", 0.5)  # Default mediator weight is one half.
    total_loss = loss_y + mt_c_weight * loss_c
    
    # Component log.
    loss_components = {f"y_{k}": v for k, v in comp_y.items()}
    loss_components["c_bce_loss"] = loss_c.item()
    
    return total_loss, loss_components

def compute_motto_loss(preds_dict, y, c, t, config):
    """
    MOTTO loss: factual risk plus selective distribution alignment.
    """
    # Predicted logits.
    y0_main, y1_main = preds_dict["main_task"]
    y0_aux, y1_aux = preds_dict["aux_task"]
    
    # Shared representations used by the alignment term.
    sda_reps = preds_dict.get("sda_reps", {})
    trt_sh_main = sda_reps.get("main")
    trt_sh_aux = sda_reps.get("aux")
    
    # ---------------------------------------------------------
    # Factual risk.
    # ---------------------------------------------------------
    # Logit on the observed treatment arm.
    obs_main = torch.where(t == 1, y1_main, y0_main)
    obs_aux = torch.where(t == 1, y1_aux, y0_aux)
    
    # Binary cross-entropy. Safe under mixed precision.
    loss_main = F.binary_cross_entropy_with_logits(obs_main, y.float(), reduction='mean')
    loss_aux = F.binary_cross_entropy_with_logits(obs_aux, c.float(), reduction='mean')
    
    # ---------------------------------------------------------
    # Selective distribution alignment.
    # Pull the treated and control shared representations toward the batch mean.
    # ---------------------------------------------------------
    sda_loss = torch.tensor(0.0, device=y.device)
    alpha_sda = config.get("alpha_sda", 0.1)  # Alignment weight.
    
    if alpha_sda > 0 and trt_sh_main is not None:
        mask_t0 = (t == 0)
        mask_t1 = (t == 1)
        
        # Both arms must be present in the batch.
        if mask_t0.sum() > 0 and mask_t1.sum() > 0:
            # Main-head alignment, first-order.
            mean_main_all = trt_sh_main.mean(dim=0)
            sda_main_t0 = F.mse_loss(trt_sh_main[mask_t0].mean(dim=0), mean_main_all)
            sda_main_t1 = F.mse_loss(trt_sh_main[mask_t1].mean(dim=0), mean_main_all)
            
            # Auxiliary-head alignment.
            mean_aux_all = trt_sh_aux.mean(dim=0)
            sda_aux_t0 = F.mse_loss(trt_sh_aux[mask_t0].mean(dim=0), mean_aux_all)
            sda_aux_t1 = F.mse_loss(trt_sh_aux[mask_t1].mean(dim=0), mean_aux_all)
            
            sda_loss = sda_main_t0 + sda_main_t1 + sda_aux_t0 + sda_aux_t1

    # Weighted sum.
    aux_weight = config.get("aux_weight", 1.0)
    total_loss = loss_main + aux_weight * loss_aux + alpha_sda * sda_loss
    
    loss_components = {
        "motto_main_loss": loss_main.item(),
        "motto_aux_loss": loss_aux.item(),
        "motto_sda_loss": sda_loss.item() if isinstance(sda_loss, torch.Tensor) else 0.0
    }
    
    return total_loss, loss_components


def compute_dragonnet_loss(y0_pred, y1_pred, targets, treatment, pi_dict, config):
    """
    DragonNet loss: outcome, treatment, and the targeted-regularization term.
    """
    # Factual binary cross-entropy.
    y_pred_obs = torch.where(treatment == 1, y1_pred, y0_pred)
    loss_y = F.binary_cross_entropy_with_logits(y_pred_obs, targets.float())
    
    # Propensity binary cross-entropy.
    t_logit = pi_dict["t_logit"]
    loss_t = F.binary_cross_entropy_with_logits(t_logit, treatment.float())
    
    # Targeted regularization.
    propensity = torch.sigmoid(t_logit).detach()  # Stop the gradient into the propensity head.
    propensity = torch.clamp(propensity, 1e-4, 1.0 - 1e-4)  # Keep the denominator away from zero.
    
    eps = pi_dict["epsilon"]
    
    # Perturbation H(X).
    perturb = (treatment.float() / propensity) - ((1.0 - treatment.float()) / (1.0 - propensity))
    
    # Loss after adding the perturbation.
    y_pred_perturbed = y_pred_obs + eps * perturb
    loss_reg = F.binary_cross_entropy_with_logits(y_pred_perturbed, targets.float())
    
    alpha = config.get("dragon_alpha", 1.0)
    beta = config.get("dragon_beta", 1.0)
    
    total_loss = loss_y + alpha * loss_t + beta * loss_reg
    
    loss_components = {
        "dragon_loss_y": loss_y.item(), 
        "dragon_loss_t": (alpha * loss_t).item(), 
        "dragon_loss_reg": (beta * loss_reg).item()
    }
    
    return total_loss, loss_components

def compute_efin_ours_loss(y0_pred, y1_pred, targets, treatment, pi_dict, config):
    """
    EFIN loss: factual outcome loss plus the intervention-constraint loss.
    """
    # Factual binary cross-entropy on the observed arm.
    y_pred_obs = torch.where(treatment == 1, y1_pred, y0_pred)
    loss_y = F.binary_cross_entropy_with_logits(y_pred_obs, targets.float())
    
    # Intervention constraint, equation 14 of the EFIN paper.
    # The constraint head is trained against the flipped treatment label.
    t_constraint_logit = pi_dict["efin_constraint_logit"]
    inverse_treatment = 1.0 - treatment.float()
    loss_c = F.binary_cross_entropy_with_logits(t_constraint_logit, inverse_treatment)
    
    # Combined loss. The default constraint weight is 0.01.
    lam = config.get("efin_lambda", 0.01)  # Default constraint weight.
    total_loss = loss_y + lam * loss_c
    
    loss_components = {
        "efin_loss_y": loss_y.item(), 
        "efin_loss_c": (lam * loss_c).item()
    }
    
    return total_loss, loss_components

def compute_efin_loss(y0_pred, y1_pred, targets, treatment, pi_dict, config):
    """
    EFIN loss: factual outcome loss plus the intervention-constraint loss.
    """
    # Factual binary cross-entropy on the observed arm.
    y_pred_obs = torch.where(treatment == 1, y1_pred, y0_pred)
    loss_y = F.binary_cross_entropy_with_logits(y_pred_obs, targets.float())
    
    # Intervention constraint, equation 14 of the EFIN paper.
    # The constraint head is trained against the flipped treatment label.
    t_constraint_logit = pi_dict["efin_t_logit"] # ["efin_constraint_logit"]
    inverse_treatment = 1.0 - treatment.float()
    loss_c = F.binary_cross_entropy_with_logits(t_constraint_logit, inverse_treatment)
    
    # The two terms are added with equal weight.
    # lam = config.get("efin_lambda", 0.01)  # Unused scale. The released sum is unweighted.
    total_loss = loss_y + loss_c
    
    loss_components = {
        "efin_loss_y": loss_y.item(), 
        "efin_loss_c": (loss_c).item()
    }
    
    return total_loss, loss_components
def compute_descn_loss(mu0_logit, mu1_logit, y, t, pi_dict, trial_cfg):
    """
    DESCN loss. The product terms are computed in float32, outside autocast.
    """
    device = mu0_logit.device
    pi_logit = pi_dict["descn_pi_logit"]
    tau_logit = pi_dict["descn_tau_logit"]
    
    # Propensity classification loss. Safe under autocast.
    loss_pi = F.binary_cross_entropy_with_logits(pi_logit, t.float())
    
    # ----------------------------------------------------
    # Product terms leave autocast and run in float32.
    # ----------------------------------------------------
    with torch.cuda.amp.autocast(enabled=False):
        # float32 avoids overflow in the product.
        p_pi = torch.clamp(torch.sigmoid(pi_logit.float()), min=0.001, max=0.999)
        p_mu1 = torch.sigmoid(mu1_logit.float())
        p_mu0 = torch.sigmoid(mu0_logit.float())
        
        # Joint event probabilities.
        p_estr = p_pi * p_mu1
        p_escr = (1.0 - p_pi) * p_mu0
        
        label_estr = (y == 1) & (t == 1)
        label_escr = (y == 1) & (t == 0)
        
        # binary_cross_entropy on probabilities, in float32.
        loss_estr = F.binary_cross_entropy(p_estr, label_estr.float())
        loss_escr = F.binary_cross_entropy(p_escr, label_escr.float())
    
    # Cross-arm constraints. Safe under autocast.
    treat_mask = (t == 1)
    control_mask = (t == 0)
    
    # CrossTR: mean over W=1 of BCE(mu0_logit + tau_logit, Y).
    if treat_mask.sum() > 0:
        cross_tr_pred = mu0_logit[treat_mask] + tau_logit[treat_mask]
        loss_crosstr = F.binary_cross_entropy_with_logits(cross_tr_pred, y[treat_mask].float())
    else:
        loss_crosstr = torch.tensor(0.0, device=device)
        
    # CrossCR: mean over W=0 of BCE(mu1_logit - tau_logit, Y).
    if control_mask.sum() > 0:
        cross_cr_pred = mu1_logit[control_mask] - tau_logit[control_mask]
        loss_crosscr = F.binary_cross_entropy_with_logits(cross_cr_pred, y[control_mask].float())
    else:
        loss_crosscr = torch.tensor(0.0, device=device)

    # Configured combination of the five terms.
    w_alpha = trial_cfg.get("descn_alpha", 0.5)
    w_beta1 = trial_cfg.get("descn_beta1", 1.0)
    w_beta0 = trial_cfg.get("descn_beta0", 1.0)
    w_gamma1 = trial_cfg.get("descn_gamma1", 0.5)  
    w_gamma0 = trial_cfg.get("descn_gamma0", 0.1)
    
    total_loss = (w_alpha * loss_pi + 
                  w_beta1 * loss_estr + 
                  w_beta0 * loss_escr + 
                  w_gamma1 * loss_crosstr + 
                  w_gamma0 * loss_crosscr)
    
    loss_comp = {
        "loss_pi": loss_pi.item(),
        "loss_estr": loss_estr.item(),
        "loss_escr": loss_escr.item(),
        "loss_crosstr": loss_crosstr.item(),
        "loss_crosscr": loss_crosscr.item()
    }
    
    return total_loss, loss_comp

def compute_efin_0717new_loss(y0_pred, y1_pred, targets, treatment, pi_dict, config):
    """
    EFIN loss matching the official calculate_loss:
    squared error on the observed raw logit, plus an unweighted
    binary cross-entropy of the treatment head against 1 - t.
    The two terms are added with equal weight.
    """
    y0_pred, y1_pred = y0_pred.float(), y1_pred.float()
    y_true = targets.float()
    t_true = treatment.float()
    
    y_pred_obs = (1.0 - t_true) * y0_pred + t_true * y1_pred
    loss_y = torch.mean(torch.square(y_pred_obs - y_true))
    
    t_logit = pi_dict["efin_t_logit"]
    loss_t = F.binary_cross_entropy_with_logits(t_logit, 1.0 - t_true)
    
    total_loss = loss_y + loss_t
    loss_components = {
        "efin_0717new_loss_y_mse": loss_y.item(),
        "efin_0717new_loss_t_bce": loss_t.item(),
    }
    return total_loss, loss_components


def compute_ecup_0717new_loss(preds_dict, y, c, t, config):
    """
    Entire-chain uplift loss. The conversion term is written so it stays finite under mixed precision.
    """
    c0_logit, c1_logit = preds_dict["c_logits"]
    cvr0_logit, cvr1_logit = preds_dict["cvr_logits"]
    
    # Float32 labels for the arithmetic below.
    y = y.float()
    c = c.float()
    
    c_logit_obs = torch.where(t == 1, c1_logit, c0_logit)
    cvr_logit_obs = torch.where(t == 1, cvr1_logit, cvr0_logit)
    
    # Click loss.
    loss_c = F.binary_cross_entropy_with_logits(c_logit_obs, c, reduction='mean')
    
    # Conversion loss. p(y) = p(click) * p(conversion | click).
    # log p(y) = logsigmoid(click logit) + logsigmoid(conversion logit).
    log_p_c = F.logsigmoid(c_logit_obs)
    log_p_cvr = F.logsigmoid(cvr_logit_obs)
    log_p_y = log_p_c + log_p_cvr
    
    # 1 - p(y) = 1 - p(click) * p(conversion | click).
    p_c = torch.sigmoid(c_logit_obs)
    p_cvr = torch.sigmoid(cvr_logit_obs)
    p_y = p_c * p_cvr
    log_one_minus_p_y = torch.log(torch.clamp(1.0 - p_y, min=1e-7, max=1.0))
    
    # Conversion binary cross-entropy.
    loss_y = -(y * log_p_y + (1.0 - y) * log_one_minus_p_y).mean()
    
    # Weighted sum.
    ctcvr_weight = config.get("ctcvr_weight", 1.0)
    total_loss = loss_c + ctcvr_weight * loss_y
    
    loss_components = {
        "ecup_0717new_ctr_loss": loss_c.item(),
        "ecup_0717new_ctcvr_loss": loss_y.item(),
    }
    return total_loss, loss_components  

def compute_canniuplift_loss(y0_pred, y1_pred, targets, treatment, mediator, pi_dict, config):
    """CanniUplift loss. Three terms.

    L_seller is the inverse-propensity-weighted factual loss on the seller towers.
    L_rdd is binary cross-entropy on the RDD composite. The control arm uses mu_c,
    which is a separate head from y0.
    L_redem supervises the mediator head on treated rows only. In this repository
    the mediator label is visit.
    The total is the configured weighted sum of the three terms.

    y0_pred and y1_pred are the seller-tower logits.
    mediator is the 0/1 visit label passed in from training.
    pi_dict carries canni_p_r_logit, canni_mu_c_logit, canni_delta_r_logit,
    and canni_delta_1mr_logit.
    If every row is untreated, the treatment rate is clamped to at least 1e-3 and
    the treated weight is zero. The treated count has a 1e-8 floor.
    """
    t = treatment.float()
    y = targets.float()
    r = mediator.float()
    p_r_logit = pi_dict["canni_p_r_logit"]
    mu_c_logit = pi_dict["canni_mu_c_logit"]
    delta_r_logit = pi_dict["canni_delta_r_logit"]
    delta_1mr_logit = pi_dict["canni_delta_1mr_logit"]
    # L_seller: inverse-propensity-weighted factual loss on the seller towers.
    y_obs_seller = torch.where(t > 0.5, y1_pred, y0_pred)
    p_t = torch.clamp(t.mean(), min=1e-3)
    p_c = torch.clamp(1.0 - t.mean(), min=1e-3)
    iptw_w = config.get("canniuplift_iptw_w", 1.0)
    w_t = t / (2.0 * p_t)
    w_c = (1.0 - t) / (2.0 * p_c)
    iptw = iptw_w * (w_t + w_c)
    l_seller_per = F.binary_cross_entropy_with_logits(y_obs_seller, y, reduction="none")
    loss_seller = (iptw * l_seller_per).mean()
    # L_rdd: control arm uses mu_c, a separate head from y0. The treated arm uses the RDD composite.
    p_r = torch.sigmoid(p_r_logit)
    p_delta_r = torch.sigmoid(delta_r_logit)
    p_delta_1mr = torch.sigmoid(delta_1mr_logit)
    p_ctrl = torch.sigmoid(mu_c_logit)
    p_composite_t = p_r * p_delta_r + (1.0 - p_r) * p_delta_1mr
    gmv_pred_prob = torch.where(t > 0.5, p_composite_t, p_ctrl)
    gmv_pred_prob = gmv_pred_prob.float()
    gmv_pred_prob = torch.clamp(gmv_pred_prob, 1e-4, 1.0 - 1e-4)
    
    # Invert the clamped probability so the loss can use the logits kernel.
    gmv_pred_logit = torch.logit(gmv_pred_prob) 
    
    # Logits form passes the autocast check.
    loss_rdd = F.binary_cross_entropy_with_logits(gmv_pred_logit, y, reduction="mean")
    # Mediator cross-entropy, treated rows only.
    n_treat = t.sum() + 1e-8
    l_redem_per = F.binary_cross_entropy_with_logits(p_r_logit, r, reduction="none")
    loss_redem = (t * l_redem_per).sum() / n_treat
    seller_w = config.get("canniuplift_seller_w", 1.0)
    rdd_w = config.get("canniuplift_rdd_w", 1.0)
    redem_w = config.get("canniuplift_redem_w", 1.0)
    total_loss = seller_w * loss_seller + rdd_w * loss_rdd + redem_w * loss_redem
    return total_loss, {
        "canni_loss_seller": loss_seller.item(),
        "canni_loss_rdd": loss_rdd.item(),
        "canni_loss_redem": loss_redem.item(),
        "canni_p_r_mean": p_r.mean().item(),
    }



# ==========================================
# Local checks. Run this file directly.
# ==========================================
if __name__ == "__main__":
    print("Checking the loss operators.\n")
    torch.manual_seed(42)
    
    batch_size = 8
    logits = torch.randn(batch_size, requires_grad=True)
    targets = torch.randint(0, 2, (batch_size,)).float()
    t = torch.randint(0, 2, (batch_size,)).float()       
    z_c = torch.randn(batch_size, 16, requires_grad=True)
    
    print("-" * 50)
    print("Test 1: focal loss")
    focal_fn = FocalLoss(alpha=0.25, gamma=2.0)
    print(f"BCE:   {F.binary_cross_entropy_with_logits(logits, targets).item():.4f}")
    print(f"Focal: {focal_fn(logits, targets).item():.4f}")
    
    print("-" * 50)
    print("Test 2: MMD")
    # Confirm the two arms can have different sizes.
    print(f"Treatment counts: T=1 {(t==1).sum().item()}, T=0 {(t==0).sum().item()}")
    z_t1, z_t0 = z_c[t == 1], z_c[t == 0]
    loss_mmd = mmd_loss(z_t1, z_t0, kernel="rbf", sigma=1.0)
    print(f"MMD: {loss_mmd.item():.4f}")
    
    print("-" * 50)
    print("Test 3: stage-3 loss")
    y0_pred = torch.randn(batch_size, requires_grad=True)
    y1_pred = torch.randn(batch_size, requires_grad=True)
    pi_dict = {
        "p_never": torch.tensor([0.95, 0.99, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5]),
        "p_complier": torch.tensor([0.01, 0.01, 0.8, 0.9, 0.2, 0.3, 0.1, 0.2]),
        "p_always": torch.tensor([0.04, 0.0, 0.1, 0.0, 0.3, 0.2, 0.4, 0.3])
    }
    stage3_cfg = {
        "loss_types": ["strata_weighted", "pairwise", "variance_reg"],
        "rank_margin": 0.05, "rank_alpha": 1.0,
        "never_threshold": 0.9, "var_lambda": 2.0
    }
    total_loss, loss_details = compute_stage3_loss(y0_pred, y1_pred, targets, t, pi_dict, stage3_cfg)
    total_loss.backward()
    
    print(f"Stage-3 loss: {total_loss.item():.4f}")
    for k, v in loss_details.items():
        print(f"  - {k:<10}: {v:.4f}")
    print(f"\nGradient on y1_pred is present: {y1_pred.grad is not None}")

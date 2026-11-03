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


# Alias used by older call sites.
compute_outcome_loss = compute_stage3_loss

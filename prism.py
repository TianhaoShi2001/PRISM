"""PRISM outcome model: stratum-indexed residual experts with feature-adaptive gating.

A frozen Stage-I mediator supplies principal-stratum priors (never-taker,
complier, always-taker). The outcome trunk predicts a shared representation,
base outcomes, and three residual experts. A gating network adapts how the
priors weight those residuals. Prior-guided sample reweighting is implemented
in ``prior_guided_weights``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEncoder(nn.Module):
    """Continuous features plus one embedding table per categorical column."""

    def __init__(self, continuous_dim, categorical_cardinalities=None, embedding_dim=8):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.embedding_dim = embedding_dim
        categorical_cardinalities = categorical_cardinalities or {}
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(cardinality, embedding_dim)
            for col, cardinality in categorical_cardinalities.items()
        })
        self.num_cat = len(self.embeddings)
        continuous = continuous_dim if continuous_dim is not None else 0
        self.output_dim = continuous + embedding_dim * self.num_cat

    def forward(self, x_cont=None, x_cat=None):
        parts = []
        if x_cont is not None:
            parts.append(x_cont)
        if x_cat is not None and self.num_cat > 0:
            parts.append(torch.cat([self.embeddings[col](x_cat[col]) for col in self.embeddings], dim=1))
        if not parts:
            return torch.empty(0)
        return torch.cat(parts, dim=1)


def _hidden_stack(in_dim, hidden_dims, dropout_rate):
    layers = []
    width = in_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(width, hidden))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        width = hidden
    return nn.Sequential(*layers), width


def _outcome_head(in_dim, head_hidden_dims):
    if not head_hidden_dims:
        return nn.Linear(in_dim, 1)
    layers = []
    width = in_dim
    for hidden in head_hidden_dims:
        layers.append(nn.Linear(width, hidden))
        layers.append(nn.ReLU())
        width = hidden
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def _residual_expert(in_dim):
    """One stratum expert: control residual and treatment residual."""

    def tower():
        return nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, 1),
        )

    return nn.ModuleDict({"res_0": tower(), "res_1": tower()})


def _match_released_init_order(in_dim, hidden_dims, dropout_rate, head_hidden_dims):
    """Consume RNG draws in the order used by the released checkpoint."""
    shared, width = _hidden_stack(in_dim, hidden_dims, dropout_rate)
    _outcome_head(width, head_hidden_dims)
    _outcome_head(width, head_hidden_dims)
    return shared


class _ResidualOutcome(nn.Module):
    """Shared outcome trunk: encoder, base heads, and three stratum experts."""

    def __init__(self, continuous_dim, categorical_cardinalities, hidden_dims, dropout_rate,
                 c_model=None, embedding_dim=8, head_hidden_dims=None):
        super().__init__()
        self.c_fusion_mode = "prism"
        self.c_model = c_model
        if self.c_model is not None:
            self.c_model.eval()
            self.c_model.requires_grad_(False)

        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities, embedding_dim)
        in_dim = self.encoder.output_dim
        _match_released_init_order(in_dim, hidden_dims, dropout_rate, head_hidden_dims)

        self.shared_base, width = _hidden_stack(in_dim, hidden_dims, dropout_rate)
        self.base_head_0 = _outcome_head(width, head_hidden_dims)
        self.base_head_1 = _outcome_head(width, head_hidden_dims)
        self.res_never = _residual_expert(width)
        self.res_comp = _residual_expert(width)
        self.res_always = _residual_expert(width)

    def extract_pi_prior(self, x_cont, x_cat):
        """Never-taker, complier, and always-taker priors from the frozen mediator.

        With mediator heads ``c0 = P(M=1 | T=0)`` and ``c1 = P(M=1 | T=1)``:

            pi_00 = clip(1 - c1, 0)
            pi_01 = clip(c1 - c0, 0)
            pi_11 = c0
        """
        if self.c_model is None:
            return None, None, None
        with torch.no_grad():
            c0_logit, c1_logit, _ = self.c_model(x_cont, x_cat)
            c0 = torch.sigmoid(c0_logit)
            c1 = torch.sigmoid(c1_logit)
            pi_never = (1.0 - c1).clamp(min=0.0)
            pi_complier = (c1 - c0).clamp(min=0.0)
            pi_always = c0
        return pi_never, pi_complier, pi_always

    def _responses(self, hidden, weight_never, weight_complier, weight_always):
        """Base outcome plus the weighted residual of each stratum."""
        base_y0 = self.base_head_0(hidden).squeeze(-1)
        base_y1 = self.base_head_1(hidden).squeeze(-1)

        res0_never = self.res_never["res_0"](hidden).squeeze(-1)
        res1_never = self.res_never["res_1"](hidden).squeeze(-1)
        res0_complier = self.res_comp["res_0"](hidden).squeeze(-1)
        res1_complier = self.res_comp["res_1"](hidden).squeeze(-1)
        res0_always = self.res_always["res_0"](hidden).squeeze(-1)
        res1_always = self.res_always["res_1"](hidden).squeeze(-1)

        y0 = base_y0 + (
            weight_never * res0_never
            + weight_complier * res0_complier
            + weight_always * res0_always
        )
        y1 = base_y1 + (
            weight_never * res1_never
            + weight_complier * res1_complier
            + weight_always * res1_always
        )
        return y0, y1, base_y0, base_y1


def _stratum_gate(hidden_dim):
    """Maps the outcome hidden state to three stratum gate scores."""
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, 3),
    )


class PRISM(_ResidualOutcome):
    """Released PRISM outcome model (feature-adaptive gating over stratum priors)."""

    def __init__(self, continuous_dim, categorical_cardinalities, hidden_dims, dropout_rate,
                 c_model=None, embedding_dim=8, head_hidden_dims=None,
                 gate_temperature=1.0, gate_stop_grad=False,
                 gate_logit_clamp=False, gate_clamp_val=2.0):
        super().__init__(
            continuous_dim, categorical_cardinalities, hidden_dims, dropout_rate,
            c_model=c_model, embedding_dim=embedding_dim, head_hidden_dims=head_hidden_dims,
        )
        self.gate_temperature = gate_temperature
        self.gate_stop_grad = gate_stop_grad
        self.gate_logit_clamp = gate_logit_clamp
        self.gate_clamp_val = gate_clamp_val
        self.dynamic_net = _stratum_gate(hidden_dims[-1])
        nn.init.zeros_(self.dynamic_net[-1].weight)
        nn.init.zeros_(self.dynamic_net[-1].bias)

    def forward(self, x_cont, x_cat):
        hidden = self.shared_base(self.encoder(x_cont, x_cat))
        pi_never, pi_complier, pi_always = self.extract_pi_prior(x_cont, x_cat)

        gate_input = hidden.detach() if self.training and self.gate_stop_grad else hidden
        offsets = self.dynamic_net(gate_input)
        if self.gate_logit_clamp:
            offsets = torch.clamp(offsets, min=-self.gate_clamp_val, max=self.gate_clamp_val)

        prior = torch.stack([pi_never, pi_complier, pi_always], dim=1)
        mixed = F.softmax((torch.log(prior + 1e-7) + offsets) / self.gate_temperature, dim=-1)
        weight_never, weight_complier, weight_always = mixed[:, 0], mixed[:, 1], mixed[:, 2]
        y0, y1, base_y0, base_y1 = self._responses(hidden, weight_never, weight_complier, weight_always)

        info = {
            "p_never": pi_never,
            "p_complier": pi_complier,
            "p_always": pi_always,
            "wb_shared_emb": hidden,
            "wb_final_pi_never": weight_never,
            "wb_final_pi_comp": weight_complier,
            "wb_final_pi_always": weight_always,
            "wb_base_y0": base_y0,
            "wb_base_y1": base_y1,
        }
        return y0, y1, info


def _cfg_float(config, *keys, default=0.0):
    for key in keys:
        if key in config and config[key] is not None:
            return float(config[key])
    return float(default)


def prior_guided_weights(targets, treatment, pi_complier, y_pred_observed, bce_raw, config):
    """Per-sample outcome-loss weights from the complier prior (paper Eq. sample-weight).

    Extra weight before renormalization:

        lambda * pi_01          if t = 1 and y = 0
        lambda * (1 - pi_01)    if t = 1 and y = 1
        lambda * pi_01          if t = 0 and y = 1
        0                       otherwise

    The released config leaves ``conflict_renorm`` unset, so weights are divided
    by their batch mean (``mean``).
    """
    pi_complier = pi_complier.detach()
    mode = config.get("conflict_mode", "all")
    alpha_treated_nonconvert = _cfg_float(config, "conflict_alpha_treated_nonconvert", default=0.0)
    alpha_treated_convert = _cfg_float(config, "conflict_alpha_treated_convert", default=0.0)
    alpha_control_convert = _cfg_float(config, "conflict_alpha_control_convert", default=0.0)
    focal_type = config.get("conflict_focal_type", "none")
    gamma = config.get("conflict_gamma", 2.0)
    global_margin = config.get("conflict_global_margin", 1.0)
    use_ohem = config.get("conflict_use_ohem", False)
    ohem_pct = config.get("conflict_ohem_pct", 0.10)
    use_weight_clip = config.get("conflict_use_weight_clip", False)
    max_weight = config.get("conflict_max_weight_thres", 3.0)

    weights = torch.ones_like(targets, dtype=torch.float)
    extra = torch.zeros_like(targets, dtype=torch.float)
    selected = torch.zeros_like(targets, dtype=torch.bool)

    treated_nonconvert = (treatment == 1) & (targets == 0)
    treated_convert = (treatment == 1) & (targets == 1)
    control_convert = (treatment == 0) & (targets == 1)

    use_tn = mode in ("all", "treated_nonconvert")
    use_tc = mode in ("all", "treated_convert")
    use_cc = mode in ("all", "control_convert")

    if use_tn and treated_nonconvert.any():
        extra[treated_nonconvert] = (alpha_treated_nonconvert * pi_complier[treated_nonconvert]).float()
        selected[treated_nonconvert] = True
    if use_tc and treated_convert.any():
        extra[treated_convert] = (alpha_treated_convert * (1.0 - pi_complier[treated_convert])).float()
        selected[treated_convert] = True
    if use_cc and control_convert.any():
        extra[control_convert] = (alpha_control_convert * pi_complier[control_convert]).float()
        selected[control_convert] = True

    if focal_type == "none":
        weights += extra
    elif focal_type == "global_bounded":
        confidence = torch.sigmoid(y_pred_observed).detach().float()
        focal = torch.clamp(1.0 + extra * torch.pow(1.0 - confidence, gamma), min=global_margin)
        weights = torch.where(selected, focal, torch.ones_like(weights))

    if use_ohem:
        with torch.no_grad():
            threshold = torch.quantile(bce_raw.detach(), 1.0 - ohem_pct)
            weights[bce_raw < threshold] = 1.0

    if use_weight_clip:
        weights = torch.clamp(weights, max=max_weight)

    min_weight = config.get("conflict_min_weight_thres", None)
    if min_weight is not None:
        weights = torch.clamp(weights, min=float(min_weight))

    renorm = config.get("conflict_renorm", "mean")
    if renorm == "mean":
        weights = weights / (weights.mean() + 1e-8)
    elif renorm != "none":
        raise ValueError(f"conflict_renorm must be 'mean' or 'none', got {renorm}")
    return weights

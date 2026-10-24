import torch
import torch.nn as nn
from prism import PRISM, PRISMLogitGate

# ==========================================
# Feature encoder.
# ==========================================
class FeatureEncoder(nn.Module):
    """
    Continuous features, categorical features, or both.
    """
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict = None, embedding_dim: int = 8):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.embedding_dim = embedding_dim
        categorical_cardinalities = categorical_cardinalities or {}
        
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(card, embedding_dim)
            for col, card in categorical_cardinalities.items()
        })
        self.num_cat = len(self.embeddings)
        self.output_dim = (continuous_dim if continuous_dim is not None else 0) + embedding_dim * self.num_cat

    def forward(self, x_cont=None, x_cat=None):
        feats = []
        if x_cont is not None:
            feats.append(x_cont)
        if x_cat is not None and self.num_cat > 0:
            emb_list = [self.embeddings[col](x_cat[col]) for col in self.embeddings]
            feats.append(torch.cat(emb_list, dim=1))
        return torch.cat(feats, dim=1) if feats else torch.empty(0)


# ==========================================
# 1D ResNet-18 for tabular inputs.
# ==========================================
class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        # 1D convolution.
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18_1D(nn.Module):
    def __init__(self, num_classes=64):
        super(ResNet18_1D, self).__init__()
        self.in_planes = 64
        
        # kernel 3 and stride 1 keep a short feature axis from collapsing.
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Four stages. Stride stays 1 so the length is preserved and the width carries the features.
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=1)
        self.layer3 = self._make_layer(256, 2, stride=1)
        self.layer4 = self._make_layer(512, 2, stride=1)
        
        # Global pool, then the final map.
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * BasicBlock1D.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(BasicBlock1D(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock1D.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # [Batch, Features] -> [Batch, 1, Features].
        x = x.unsqueeze(1) 
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.adaptive_pool(out)    # [Batch, 512, 1]
        out = out.view(out.size(0), -1)  # [Batch, 512]
        out = self.fc(out)               # [Batch, num_classes]
        return out


# ==========================================
# Expert factory.
# ==========================================
def build_expert(expert_type, input_dim, hidden_dims, dropout_rate=0.1):
    layers = []
    curr_dim = input_dim
    
    if expert_type == "mlp":
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim
        return nn.Sequential(*layers), curr_dim
        
    elif expert_type == "resnet18":
        # hidden_dims[0] is the expert output width.
        out_dim = hidden_dims[0] if isinstance(hidden_dims, list) else hidden_dims
        return ResNet18_1D(num_classes=out_dim), out_dim
        
    else:
        raise ValueError(f"Unsupported expert_type: {expert_type}")


# ==========================================
# Multi-task mixture-of-experts layer.
# ==========================================
class MMoE_Layer(nn.Module):
    def __init__(self, input_dim, num_experts, num_tasks, expert_type="mlp", expert_hidden_dims=[64], dropout_rate=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert_net, self.expert_out_dim = build_expert(expert_type, input_dim, expert_hidden_dims, dropout_rate)
            self.experts.append(expert_net)
        
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):
        # expert_outputs: [B, num_experts, expert_out_dim]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) 
        
        task_reps = []
        for i in range(self.num_tasks):
            gate_weights = self.gates[i](x).unsqueeze(-1) # [B, num_experts, 1]
            task_rep = torch.sum(expert_outputs * gate_weights, dim=1) # [B, expert_out_dim]
            task_reps.append(task_rep)
            
        return task_reps


# ==========================================
# Treatment-feature attention. Equation 6.
# ==========================================
class UserTreatmentInteraction(nn.Module):
    def __init__(self, t_dim, u_dim, out_dim):
        super().__init__()
        self.W_t = nn.Linear(t_dim, out_dim)
        self.W_u = nn.Linear(u_dim, out_dim)
        self.W_v = nn.Linear(u_dim, out_dim)
        self.scale = out_dim ** 0.5

    def forward(self, epsilon, phi):
        Q = self.W_t(epsilon) # [B, out_dim]
        K = self.W_u(phi)     # [B, out_dim]
        V = self.W_v(phi)     # [B, out_dim]
        
        # Attention Score
        attn_score = torch.sigmoid((Q * K) / self.scale) 
        psi = attn_score * V # [B, out_dim]
        return psi


# ==========================================
# MTMT, single-treatment form.
# ==========================================
class MTMT_STMT(nn.Module):
    """
    Single-treatment MTMT:
    1. The task representation phi predicts the control response y(0).
    2. Attention between the treatment embedding and phi predicts the increment tau.
    3. The treated response is y(1) = y(0) + tau.
    """
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 num_experts: int = 4, expert_type: str = "mlp", expert_hidden_dims: list = [64], 
                 dropout_rate: float = 0.1, t_emb_dim: int = 16):
        super().__init__()
        
        # Feature encoder.
        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities)
        
        # MMoE separates the task representations.
        self.mmoe = MMoE_Layer(
            input_dim=self.encoder.output_dim, 
            num_experts=num_experts, 
            num_tasks=2, # Outcome head and mediator head.
            expert_type=expert_type, 
            expert_hidden_dims=expert_hidden_dims, 
            dropout_rate=dropout_rate
        )
        expert_out_dim = self.mmoe.expert_out_dim
        
        # Treatment embedding.
        self.t_emb = nn.Parameter(torch.randn(1, t_emb_dim)) 
        
        # Outcome head.
        self.y0_head_main = nn.Linear(expert_out_dim, 1)
        self.interaction_main = UserTreatmentInteraction(t_emb_dim, expert_out_dim, expert_out_dim)
        self.enhancer_main = nn.Sequential(
            nn.Linear(expert_out_dim, 32), 
            nn.ReLU(), 
            nn.Linear(32, 1)
        )
        
        # Mediator head.
        self.y0_head_aux = nn.Linear(expert_out_dim, 1)
        self.interaction_aux = UserTreatmentInteraction(t_emb_dim, expert_out_dim, expert_out_dim)
        self.enhancer_aux = nn.Sequential(
            nn.Linear(expert_out_dim, 32), 
            nn.ReLU(), 
            nn.Linear(32, 1)
        )

    def forward(self, x_cont, x_cat):
        # Encode features.
        x = self.encoder(x_cont, x_cat)
        
        # One representation per task.
        phi_main, phi_aux = self.mmoe(x)
        
        # Control response y(0).
        y0_main = self.y0_head_main(phi_main).squeeze(-1)
        y0_aux = self.y0_head_aux(phi_aux).squeeze(-1)
        
        # Broadcast the treatment embedding from [1, t_emb_dim] to [B, t_emb_dim].
        eps = self.t_emb.expand(x.size(0), -1) 
        
        # Attention between the treatment embedding and the task representation.
        psi_main = self.interaction_main(eps, phi_main)
        psi_aux = self.interaction_aux(eps, phi_aux)
        
        # Increment tau.
        tau_main = self.enhancer_main(psi_main).squeeze(-1)
        tau_aux = self.enhancer_aux(psi_aux).squeeze(-1)
        
        # Treated response y(1).
        y1_main = y0_main + tau_main
        y1_aux = y0_aux + tau_aux
        
        # Same dictionary layout as the trainer.
        return {
            "main_task": (y0_main, y1_main), 
            "aux_task": (y0_aux, y1_aux),    
            "pi_dict": {} # MTMT does not emit a complier probability.
        }


# ==========================================
# ECUP, WWW 2024.
# Section 5.1.2 and Table 2.
# ==========================================
class TENet(nn.Module):
    """Treatment-enhanced network. Figure 5."""
    def __init__(self, f_num, d_dim, num_heads=2):
        super().__init__()
        self.f_num = f_num
        self.d_dim = d_dim
        
        # Self-attention over crossed features.
        self.self_attn = nn.MultiheadAttention(embed_dim=d_dim, num_heads=num_heads, batch_first=True)
        
        # The treatment-information extractor has one layer.
        self.tie_mlp_g = nn.Linear(d_dim, d_dim)
        self.tie_mlp_w = nn.Linear(d_dim, d_dim)
        
    def forward(self, E_x, E_tr):
        # E_x: [B, f_num, d_dim], E_tr: [B, 1, d_dim]
        E_att, _ = self.self_attn(E_x, E_x, E_x) 
        
        E_bit_g = self.tie_mlp_g(E_tr) 
        E_bit_w = self.tie_mlp_w(E_tr) 
        
        E_TAU_g = E_att * E_bit_g 
        W_b = E_att * E_bit_w     
        
        gate = torch.sigmoid(W_b)
        E_r = E_x * gate + E_TAU_g * (1 - gate) 
        
        E_r_final = torch.cat([E_r, E_tr], dim=1) # [B, f_num + 1, d_dim]
        return E_r_final

class TAENet(nn.Module):
    """Task-enhanced network. Injects the task prior."""
    def __init__(self, d_dim, tae_h, tower_h, num_tasks=2, num_heads=2, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        self.num_tasks = num_tasks
        self.E_ta = nn.Parameter(torch.randn(num_tasks, d_dim)) 
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_dim, num_heads=num_heads, batch_first=True)
        
        # Two gate layers. Hidden width tae_h, output width tower_h, so the gate can scale the tower.
        self.tae_mlp = nn.Sequential(
            nn.Linear(d_dim, tae_h),
            nn.ReLU(),
            nn.Linear(tae_h, tower_h) # Output width matches the tower.
        )
        
    def forward(self, E_r_final):
        B = E_r_final.size(0)
        query = self.E_ta.unsqueeze(0).expand(B, -1, -1) # [B, 2, d_dim]
        
        E_pri, _ = self.cross_attn(query, E_r_final, E_r_final) 
        
        # [B, 2, tower_h]
        delta = self.gamma * torch.sigmoid(self.tae_mlp(E_pri))
        return delta

class ECUP_Model(nn.Module):
    """Entire-chain uplift model."""
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict, 
                 tower_h: int = 128, tae_h: int = 64, d_dim: int = 16, 
                 num_heads: int = 2, gamma: float = 1.0):
        super().__init__()
        
        # Field metadata.
        self.d_dim = d_dim
        self.cont_dim = continuous_dim
        self.cat_cards = categorical_cardinalities or {}
        self.f_num = self.cont_dim + len(self.cat_cards)

        # One projection per field. Equation 3.
        self.cont_projections = nn.ModuleList([
            nn.Linear(1, d_dim) for _ in range(self.cont_dim)
        ])
        self.cat_embeddings = nn.ModuleDict({
            col: nn.Embedding(card, d_dim)
            for col, card in self.cat_cards.items()
        })

        self.t_emb_0 = nn.Parameter(torch.randn(1, 1, d_dim))
        self.t_emb_1 = nn.Parameter(torch.randn(1, 1, d_dim))
        
        self.tenet = TENet(f_num=self.f_num, d_dim=d_dim, num_heads=num_heads)
        
        # tower_h is passed in so each gate unit matches one tower unit.
        self.taenet = TAENet(d_dim=d_dim, tae_h=tae_h, tower_h=tower_h, 
                             num_tasks=2, num_heads=num_heads, gamma=gamma)
        
        # Three tower layers. The first two use width tower_h.
        # Layer 1: Shared Layer
        self.shared_layer = nn.Sequential(nn.Linear((self.f_num + 1) * d_dim, tower_h), nn.ReLU())
        
        # Layer 2: Task-specific Layer
        self.ctr_layer2 = nn.Sequential(nn.Linear(tower_h, tower_h), nn.ReLU())
        self.cvr_layer2 = nn.Sequential(nn.Linear(tower_h, tower_h), nn.ReLU())
        
        # Layer 3: Task-specific Output Layer (The Last Layer)
        self.ctr_out = nn.Linear(tower_h, 1)
        self.cvr_out = nn.Linear(tower_h, 1)

    def _get_initial_embeddings(self, x_cont, x_cat):
        B = x_cont.shape[0] if x_cont is not None else next(iter(x_cat.values())).shape[0]
        all_field_embeds = []

        if x_cont is not None:
            for i in range(self.cont_dim):
                feat = x_cont[:, i:i+1]
                all_field_embeds.append(self.cont_projections[i](feat).unsqueeze(1)) 

        if x_cat is not None:
            for col, val in x_cat.items():
                all_field_embeds.append(self.cat_embeddings[col](val).unsqueeze(1)) 

        return torch.cat(all_field_embeds, dim=1)

    def _forward_once(self, x_cont, x_cat, T):
        E_x = self._get_initial_embeddings(x_cont, x_cat) # [B, f_num, d_dim]
        B = E_x.size(0)
        
        E_tr = self.t_emb_1.expand(B, -1, -1) if T == 1 else self.t_emb_0.expand(B, -1, -1)
        
        E_r_final = self.tenet(E_x, E_tr) 
        delta = self.taenet(E_r_final) # [B, 2, tower_h]
        
        # Scale every layer except the last: "scale with each layer ... except the last layer".
        # --- Layer 1 ---
        l1_out = self.shared_layer(E_r_final.view(B, -1)) # [B, tower_h]
        
        # --- Layer 2 & Scale ---
        # Click branch.
        ctr_h1 = l1_out * delta[:, 0, :]               # Scale the first layer.
        ctr_h2 = self.ctr_layer2(ctr_h1)               # Second layer.
        ctr_h2 = ctr_h2 * delta[:, 0, :]               # Scale the second layer.
        ctr_logit = self.ctr_out(ctr_h2).squeeze(-1)   # Last layer is not scaled.
        
        # Conversion branch.
        cvr_h1 = l1_out * delta[:, 1, :]               # Scale the first layer.
        cvr_h2 = self.cvr_layer2(cvr_h1)               # Second layer.
        cvr_h2 = cvr_h2 * delta[:, 1, :]               # Scale the second layer.
        cvr_logit = self.cvr_out(cvr_h2).squeeze(-1)   # Last layer is not scaled.
        
        return ctr_logit, cvr_logit

    def forward(self, x_cont, x_cat):
        c0_logit, cvr0_logit = self._forward_once(x_cont, x_cat, T=0)
        c1_logit, cvr1_logit = self._forward_once(x_cont, x_cat, T=1)
        return {
            "c_logits": (c0_logit, c1_logit),
            "cvr_logits": (cvr0_logit, cvr1_logit),
            "pi_dict": {}
        }




# ==========================================
# MOTTO, KDD 2025. 
# Per-field projection, width-matched experts, and selective distribution alignment.
# ==========================================

class ExpertNetwork(nn.Module):
    """Expert network with a configurable depth."""
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            curr_dim = h_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = curr_dim # Width consumed by the towers.

    def forward(self, x):
        return self.net(x)

class MOTTO_Model(nn.Module):
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict, 
                 d_dim: int = 16, bottom_dim: int = 128, 
                 expert_hidden_dims: list = [64, 64], tower_dim: int = 64,
                 use_specific_experts: bool = True):
        super().__init__()
        
        self.use_specific_experts = use_specific_experts
        
        # ==========================================
        # One projection per field, continuous or categorical.
        # ==========================================
        self.cont_dim = continuous_dim
        self.cat_cards = categorical_cardinalities or {}
        # Continuous field: Linear(1, d_dim).
        self.cont_projections = nn.ModuleList([
            nn.Linear(1, d_dim) for _ in range(self.cont_dim)
        ])
        # Categorical field: embedding table.
        self.cat_embeddings = nn.ModuleDict({
            col: nn.Embedding(card, d_dim)
            for col, card in self.cat_cards.items()
        })
        
        # Flattened width = (continuous fields + categorical fields) * d_dim.
        self.total_feature_dim = (self.cont_dim + len(self.cat_cards)) * d_dim
        
        # ==========================================
        # Shared bottom.
        # ==========================================
        self.shared_bottom = nn.Sequential(
            nn.Linear(self.total_feature_dim, bottom_dim),
            nn.ReLU()
        )
        
        # ==========================================
        # Experts share one layout. Their widths follow the config.
        # ==========================================
        # Global expert.
        self.exp_global = ExpertNetwork(bottom_dim, expert_hidden_dims)
        expert_out_dim = self.exp_global.output_dim # Expert output width.
        
        # Outcome experts shared across T=0 and T=1.
        self.exp_out_sh_t0 = ExpertNetwork(bottom_dim, expert_hidden_dims)
        self.exp_out_sh_t1 = ExpertNetwork(bottom_dim, expert_hidden_dims)
        
        # Treatment-shared experts for the main and auxiliary heads. Used by the alignment loss.
        self.exp_trt_sh_main = ExpertNetwork(bottom_dim, expert_hidden_dims)
        self.exp_trt_sh_aux = ExpertNetwork(bottom_dim, expert_hidden_dims)
        
        # Experts specific to (T, task).
        self.num_experts_per_gate = 3
        if use_specific_experts:
            self.exp_spec_t0_main = ExpertNetwork(bottom_dim, expert_hidden_dims)
            self.exp_spec_t1_main = ExpertNetwork(bottom_dim, expert_hidden_dims)
            self.exp_spec_t0_aux = ExpertNetwork(bottom_dim, expert_hidden_dims)
            self.exp_spec_t1_aux = ExpertNetwork(bottom_dim, expert_hidden_dims)
            self.num_experts_per_gate = 4
            
        # ==========================================
        # Gates and towers.
        # ==========================================
        # Gates return weights of shape [B, num_experts].
        self.gate_t0_main = nn.Sequential(nn.Linear(bottom_dim, self.num_experts_per_gate), nn.Softmax(dim=-1))
        self.gate_t1_main = nn.Sequential(nn.Linear(bottom_dim, self.num_experts_per_gate), nn.Softmax(dim=-1))
        self.gate_t0_aux = nn.Sequential(nn.Linear(bottom_dim, self.num_experts_per_gate), nn.Softmax(dim=-1))
        self.gate_t1_aux = nn.Sequential(nn.Linear(bottom_dim, self.num_experts_per_gate), nn.Softmax(dim=-1))
        
        # Towers take the aggregated expert output.
        def build_tower():
            return nn.Sequential(nn.Linear(expert_out_dim, tower_dim), nn.ReLU(), nn.Linear(tower_dim, 1))
            
        self.tower_t0_main = build_tower()
        self.tower_t1_main = build_tower()
        self.tower_t0_aux = build_tower()
        self.tower_t1_aux = build_tower()

    def _get_fair_embeddings(self, x_cont, x_cat):
        """Project continuous and categorical fields to a common width."""
        all_field_embeds = []
        if x_cont is not None:
            for i in range(self.cont_dim):
                feat = x_cont[:, i:i+1]
                all_field_embeds.append(self.cont_projections[i](feat)) # [B, d_dim]
        if x_cat is not None:
            for col, val in x_cat.items():
                all_field_embeds.append(self.cat_embeddings[col](val)) # [B, d_dim]
        # Flatten for the shared bottom.
        return torch.cat(all_field_embeds, dim=1) # [B, f_num * d_dim]

    def forward(self, x_cont, x_cat):
        # Shared representation.
        x_flat = self._get_fair_embeddings(x_cont, x_cat)
        h_bottom = self.shared_bottom(x_flat)
        
        # Every expert.
        e_glob = self.exp_global(h_bottom)
        e_out_t0 = self.exp_out_sh_t0(h_bottom)
        e_out_t1 = self.exp_out_sh_t1(h_bottom)
        e_trt_main = self.exp_trt_sh_main(h_bottom)
        e_trt_aux = self.exp_trt_sh_aux(h_bottom)
        
        if self.use_specific_experts:
            e_sp_t0_main = self.exp_spec_t0_main(h_bottom)
            e_sp_t1_main = self.exp_spec_t1_main(h_bottom)
            e_sp_t0_aux = self.exp_spec_t0_aux(h_bottom)
            e_sp_t1_aux = self.exp_spec_t1_aux(h_bottom)
            
        # Route and aggregate.
        # --- (T=0, Main) ---
        exps_t0_main = [e_glob, e_out_t0, e_trt_main] + ([e_sp_t0_main] if self.use_specific_experts else [])
        agg_t0_main = torch.sum(torch.stack(exps_t0_main, dim=1) * self.gate_t0_main(h_bottom).unsqueeze(-1), dim=1)
        y0_main = self.tower_t0_main(agg_t0_main).squeeze(-1)
        
        # --- (T=1, Main) ---
        exps_t1_main = [e_glob, e_out_t1, e_trt_main] + ([e_sp_t1_main] if self.use_specific_experts else [])
        agg_t1_main = torch.sum(torch.stack(exps_t1_main, dim=1) * self.gate_t1_main(h_bottom).unsqueeze(-1), dim=1)
        y1_main = self.tower_t1_main(agg_t1_main).squeeze(-1)
        
        # --- (T=0, Aux) ---
        exps_t0_aux = [e_glob, e_out_t0, e_trt_aux] + ([e_sp_t0_aux] if self.use_specific_experts else [])
        agg_t0_aux = torch.sum(torch.stack(exps_t0_aux, dim=1) * self.gate_t0_aux(h_bottom).unsqueeze(-1), dim=1)
        y0_aux = self.tower_t0_aux(agg_t0_aux).squeeze(-1)
        
        # --- (T=1, Aux) ---
        exps_t1_aux = [e_glob, e_out_t1, e_trt_aux] + ([e_sp_t1_aux] if self.use_specific_experts else [])
        agg_t1_aux = torch.sum(torch.stack(exps_t1_aux, dim=1) * self.gate_t1_aux(h_bottom).unsqueeze(-1), dim=1)
        y1_aux = self.tower_t1_aux(agg_t1_aux).squeeze(-1)

        # Predictions and the representations used by selective distribution alignment.
        return {
            "main_task": (y0_main, y1_main),
            "aux_task": (y0_aux, y1_aux),
            "sda_reps": {"main": e_trt_main, "aux": e_trt_aux},
            "pi_dict": {}
        }



# ==========================================
# Naive multi-task TARNet.
# Shared bottom and four heads: Y0, Y1, C0, C1.
# ==========================================
class TARNET_Naive_MT(nn.Module):
    def __init__(self, continuous_dim, categorical_cardinalities, hidden_dims=[128, 64, 32], dropout_rate=0.1):
        super().__init__()
        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities)
        input_dim = self.encoder.output_dim
        
        # Shared bottom.
        layers = []
        curr_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(curr_dim, dim), nn.ReLU(), nn.Dropout(dropout_rate)])
            curr_dim = dim
        self.shared_bottom = nn.Sequential(*layers)
        
        # Four heads.
        self.head_y0 = nn.Linear(curr_dim, 1)
        self.head_y1 = nn.Linear(curr_dim, 1)
        self.head_c0 = nn.Linear(curr_dim, 1)
        self.head_c1 = nn.Linear(curr_dim, 1)

    def forward(self, x_cont, x_cat):
        # Encode.
        feat = self.encoder(x_cont, x_cat)
        
        # Shared bottom.
        shared_rep = self.shared_bottom(feat)
        
        # Four logits. The last dimension is squeezed so the loss sees a vector.
        y0_logit = self.head_y0(shared_rep).squeeze(-1)
        y1_logit = self.head_y1(shared_rep).squeeze(-1)
        c0_logit = self.head_c0(shared_rep).squeeze(-1)
        c1_logit = self.head_c1(shared_rep).squeeze(-1)
        
        # Stratum probabilities from the two mediator heads.
        p_c0 = torch.sigmoid(c0_logit)
        p_c1 = torch.sigmoid(c1_logit)
        
        pi_dict = {
            "p_complier": torch.clamp(p_c1 - p_c0, min=0.0), # Complier probability, floored at 0.
            "p_never": torch.clamp(1.0 - p_c1, min=0.0),     # Never-taker probability, floored at 0.
            "p_always": torch.clamp(p_c0, min=0.0)           # Always-taker probability.
        }
        
        # Same output layout as the evaluator and the loss.
        return {
            "main_task": (y0_logit, y1_logit),
            "aux_task": (c0_logit, c1_logit),
            "pi_dict": pi_dict
        }


# ==========================================
# Mediator backbone.
# ==========================================
# ==========================================
# Mediator backbone, also used as an outcome baseline without mediator features.
# ==========================================
class TARNET_Baseline(nn.Module):
    """
    Mediator backbone. Also the outcome baseline that does not consume mediator features.
    The forward pass returns the shared representation for the MMD loss.
    """
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 hidden_dims: list, dropout_rate: float, embedding_dim: int = 8,
                 head_hidden_dims: list = None): # Optional hidden widths for the outcome heads.
        super().__init__()
        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities, embedding_dim)
        
        layers = []
        curr_dim = self.encoder.output_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0: layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # A non-empty head_hidden_dims turns each head into an MLP.
        if head_hidden_dims and len(head_hidden_dims) > 0:
            # Control head.
            h0_layers = []
            c_dim = curr_dim
            for dim in head_hidden_dims:
                h0_layers.extend([nn.Linear(c_dim, dim), nn.ReLU()])
                c_dim = dim
            h0_layers.append(nn.Linear(c_dim, 1))
            self.head_0 = nn.Sequential(*h0_layers)
            
            # Treated head.
            h1_layers = []
            c_dim = curr_dim
            for dim in head_hidden_dims:
                h1_layers.extend([nn.Linear(c_dim, dim), nn.ReLU()])
                c_dim = dim
            h1_layers.append(nn.Linear(c_dim, 1))
            self.head_1 = nn.Sequential(*h1_layers)
            
        else:
            # One linear layer per head.
            self.head_0 = nn.Linear(curr_dim, 1)
            self.head_1 = nn.Linear(curr_dim, 1)

    def forward(self, x_cont, x_cat):
        x = self.encoder(x_cont, x_cat)
        shared_emb = self.shared_layers(x) # Shared representation.
        out_0 = self.head_0(shared_emb).squeeze(-1)
        out_1 = self.head_1(shared_emb).squeeze(-1)
        # Shared representation, exposed for inspection.
        if not self.training:
            # Carried in pi_dict so the evaluator can read it.
            pi_dict = {"wb_shared_emb": shared_emb}
            return out_0, out_1, pi_dict

        return out_0, out_1, shared_emb


# ==========================================
# Outcome model. Level 1 uses raw probabilities. Level 2 uses a joint embedding.
# Level 1 (raw probability) and level 2 (joint embedding).
# ==========================================
class TARNET_Proposed(nn.Module):
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 hidden_dims: list, dropout_rate: float, 
                 c_fusion_mode: str = "joint_emb", c_embedding_dim: int = 4, 
                 c_model: nn.Module = None, embedding_dim: int = 8,
                 head_hidden_dims: list = None): # Optional hidden widths for the outcome heads.
        super().__init__()
        self.c_fusion_mode = c_fusion_mode
        self.c_model = c_model
        
        # The mediator is frozen.
        if self.c_model is not None:
            self.c_model.eval()
            self.c_model.requires_grad_(False)

        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities, embedding_dim)
        input_dim = self.encoder.output_dim

        # Width of the injected prior.
        if self.c_model is not None:
            if self.c_fusion_mode == "raw_prob":  
                input_dim += 3 # [pi_00, pi_01, pi_11]
            elif self.c_fusion_mode == "joint_emb": 
                input_dim += c_embedding_dim
                self.emb_never = nn.Parameter(torch.randn(c_embedding_dim))
                self.emb_comp = nn.Parameter(torch.randn(c_embedding_dim))
                self.emb_always = nn.Parameter(torch.randn(c_embedding_dim))

        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0: layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim
            
        self.shared_layers = nn.Sequential(*layers)
        
        # Optional MLP heads.
        if head_hidden_dims and len(head_hidden_dims) > 0:
            h0_layers = []
            c_dim = curr_dim
            for dim in head_hidden_dims:
                h0_layers.extend([nn.Linear(c_dim, dim), nn.ReLU()])
                c_dim = dim
            h0_layers.append(nn.Linear(c_dim, 1))
            self.head_0 = nn.Sequential(*h0_layers)
            
            h1_layers = []
            c_dim = curr_dim
            for dim in head_hidden_dims:
                h1_layers.extend([nn.Linear(c_dim, dim), nn.ReLU()])
                c_dim = dim
            h1_layers.append(nn.Linear(c_dim, 1))
            self.head_1 = nn.Sequential(*h1_layers)
        else:
            self.head_0 = nn.Linear(curr_dim, 1)
            self.head_1 = nn.Linear(curr_dim, 1)

    def extract_pi_prior(self, x_cont, x_cat):
        """Read stratum probabilities from the frozen mediator."""
        if self.c_model is None:
            return None, None, None
            
        with torch.no_grad(): 
            c0_logit, c1_logit, _ = self.c_model(x_cont, x_cat)
            c0_prob, c1_prob = torch.sigmoid(c0_logit), torch.sigmoid(c1_logit)
            
            # Principal-stratum probabilities.
            pi_00 = (1.0 - c1_prob).clamp(min=0.0)      # Never-taker
            pi_01 = (c1_prob - c0_prob).clamp(min=0.0)  # Complier
            pi_11 = c0_prob                             # Always-taker
            
        return pi_00, pi_01, pi_11

    def forward(self, x_cont, x_cat):
        x_main = self.encoder(x_cont, x_cat)
        pi_dict = {}
        c_feat = None
        
        if self.c_model is not None:
            pi_00, pi_01, pi_11 = self.extract_pi_prior(x_cont, x_cat)
            pi_dict = {"p_never": pi_00, "p_complier": pi_01, "p_always": pi_11}
            
            # Fuse the prior into the features.
            if self.c_fusion_mode == "raw_prob":
                c_feat = torch.stack([pi_00, pi_01, pi_11], dim=1)
            elif self.c_fusion_mode == "joint_emb":
                c_feat = (pi_00.unsqueeze(1) * self.emb_never.unsqueeze(0) +
                          pi_01.unsqueeze(1) * self.emb_comp.unsqueeze(0) +
                          pi_11.unsqueeze(1) * self.emb_always.unsqueeze(0))

        x_aug = torch.cat([x_main, c_feat], dim=1) if c_feat is not None else x_main
        
        shared_y = self.shared_layers(x_aug)
        y0 = self.head_0(shared_y).squeeze(-1)
        y1 = self.head_1(shared_y).squeeze(-1)
        if not self.training:
            pi_dict["wb_shared_emb"] = shared_y
        return y0, y1, pi_dict


# ==========================================
# Three expert towers, gated by the stratum probabilities.
# ==========================================
class TARNET_MoE(TARNET_Proposed):
    """
    Three expert towers. Features are not concatenated with the prior.
    The stratum probabilities gate which tower contributes.
    """
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 hidden_dims: list, dropout_rate: float, 
                 c_model: nn.Module = None, embedding_dim: int = 8,
                 head_hidden_dims: list = None): # Passed to the parent and used by each expert head.
        # Reuse extract_pi_prior from the parent.
        super().__init__(continuous_dim, categorical_cardinalities, hidden_dims, 
                         dropout_rate, c_fusion_mode="none", c_model=c_model, 
                         embedding_dim=embedding_dim, head_hidden_dims=head_hidden_dims)
        
        input_dim = self.encoder.output_dim
        
        # Build one expert tower.
        def build_expert():
            layers = []
            curr_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(curr_dim, h_dim))
                layers.append(nn.ReLU())
                if dropout_rate > 0: layers.append(nn.Dropout(dropout_rate))
                curr_dim = h_dim
            
            # Match the head depth of the parent so the capacity is the same.
            if head_hidden_dims and len(head_hidden_dims) > 0:
                h0_layers = []
                c_dim = curr_dim
                for dim in head_hidden_dims:
                    h0_layers.extend([nn.Linear(c_dim, dim), nn.ReLU()])
                    c_dim = dim
                h0_layers.append(nn.Linear(c_dim, 1))
                head_0_net = nn.Sequential(*h0_layers)
                
                h1_layers = []
                c_dim = curr_dim
                for dim in head_hidden_dims:
                    h1_layers.extend([nn.Linear(c_dim, dim), nn.ReLU()])
                    c_dim = dim
                h1_layers.append(nn.Linear(c_dim, 1))
                head_1_net = nn.Sequential(*h1_layers)
            else:
                head_0_net = nn.Linear(curr_dim, 1)
                head_1_net = nn.Linear(curr_dim, 1)

            return nn.ModuleDict({
                'shared': nn.Sequential(*layers),
                'head_0': head_0_net,
                'head_1': head_1_net
            })

        # One tower per stratum.
        self.expert_never = build_expert()
        self.expert_comp = build_expert()
        self.expert_always = build_expert()
        
        # Drop the unused parent layers.
        del self.shared_layers
        del self.head_0
        del self.head_1

    def forward(self, x_cont, x_cat):
        x_main = self.encoder(x_cont, x_cat)
        
        # Stratum probabilities from the mediator.
        pi_00, pi_01, pi_11 = self.extract_pi_prior(x_cont, x_cat)
        pi_dict = {"p_never": pi_00, "p_complier": pi_01, "p_always": pi_11}
        
        # One score per expert.
        sh_never = self.expert_never['shared'](x_main)
        y0_never = self.expert_never['head_0'](sh_never).squeeze(-1)
        y1_never = self.expert_never['head_1'](sh_never).squeeze(-1)
        
        sh_comp = self.expert_comp['shared'](x_main)
        y0_comp = self.expert_comp['head_0'](sh_comp).squeeze(-1)
        y1_comp = self.expert_comp['head_1'](sh_comp).squeeze(-1)
        
        sh_always = self.expert_always['shared'](x_main)
        y0_always = self.expert_always['head_0'](sh_always).squeeze(-1)
        y1_always = self.expert_always['head_1'](sh_always).squeeze(-1)
        
        # Soft mixture.
        y0 = pi_00 * y0_never + pi_01 * y0_comp + pi_11 * y0_always
        y1 = pi_00 * y1_never + pi_01 * y1_comp + pi_11 * y1_always
        
        if not self.training:
            # Per-expert representation.
            pi_dict["wb_sh_never"] = sh_never
            pi_dict["wb_sh_comp"] = sh_comp
            pi_dict["wb_sh_always"] = sh_always
            
            # Never-taker expert.
            pi_dict["wb_exp_y0_never"] = y0_never
            pi_dict["wb_exp_y1_never"] = y1_never
            pi_dict["wb_exp_u_never"] = y1_never - y0_never
            
            # Complier expert.
            pi_dict["wb_exp_y0_comp"] = y0_comp
            pi_dict["wb_exp_y1_comp"] = y1_comp
            pi_dict["wb_exp_u_comp"] = y1_comp - y0_comp
            
            # Always-taker expert.
            pi_dict["wb_exp_y0_always"] = y0_always
            pi_dict["wb_exp_y1_always"] = y1_always
            pi_dict["wb_exp_u_always"] = y1_always - y0_always
        return y0, y1, pi_dict


# ==========================================
# Residual mixture of experts.
# ==========================================
# ==========================================
# Residual mixture of experts. Expert heads match the base head.
# ==========================================
class TARNET_Residual_MoE(TARNET_Proposed):
    """
    Residual mixture of experts.
    The base fits the shared response. Each expert fits a stratum residual.
    Expert-head depth matches the base head.
    """
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 hidden_dims: list, dropout_rate: float, 
                 c_model: nn.Module = None, embedding_dim: int = 8,
                 head_hidden_dims: list = None): # Passed to the parent. Base heads and residual heads both use it.
        super().__init__(continuous_dim, categorical_cardinalities, hidden_dims, 
                         dropout_rate, c_fusion_mode="none", c_model=c_model, 
                         embedding_dim=embedding_dim, head_hidden_dims=head_hidden_dims)
        
        input_dim = self.encoder.output_dim
        
        # Shared base, fit on the full sample.
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0: layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim
        self.shared_base = nn.Sequential(*layers)
        
        # Base heads. Depth follows head_hidden_dims.
        if head_hidden_dims and len(head_hidden_dims) > 0:
            bh0_layers = []
            c_dim = curr_dim
            for dim in head_hidden_dims:
                bh0_layers.extend([nn.Linear(c_dim, dim), nn.ReLU()])
                c_dim = dim
            bh0_layers.append(nn.Linear(c_dim, 1))
            self.base_head_0 = nn.Sequential(*bh0_layers)
            
            bh1_layers = []
            c_dim = curr_dim
            for dim in head_hidden_dims:
                bh1_layers.extend([nn.Linear(c_dim, dim), nn.ReLU()])
                c_dim = dim
            bh1_layers.append(nn.Linear(c_dim, 1))
            self.base_head_1 = nn.Sequential(*bh1_layers)
        else:
            self.base_head_0 = nn.Linear(curr_dim, 1)
            self.base_head_1 = nn.Linear(curr_dim, 1)

        # Residual experts use the same nonlinear depth and width as the base heads.
        # Each residual tower mirrors the base head.
        def build_res_expert():
            return nn.ModuleDict({
                'res_0': nn.Sequential(
                    nn.Linear(curr_dim, curr_dim // 2), 
                    nn.ReLU(), 
                    nn.Linear(curr_dim // 2, 1)
                ),
                'res_1': nn.Sequential(
                    nn.Linear(curr_dim, curr_dim // 2), 
                    nn.ReLU(), 
                    nn.Linear(curr_dim // 2, 1)
                )
            })

        # One residual tower per stratum.
        self.res_never = build_res_expert()
        self.res_comp = build_res_expert()
        self.res_always = build_res_expert()
        
        # Drop the unused parent layers.
        del self.shared_layers
        del self.head_0
        del self.head_1

    def forward(self, x_cont, x_cat):
        x_main = self.encoder(x_cont, x_cat)
        
        # Stratum probabilities, used as gates.
        pi_00, pi_01, pi_11 = self.extract_pi_prior(x_cont, x_cat)
        pi_dict = {"p_never": pi_00, "p_complier": pi_01, "p_always": pi_11}
        
        # Base logits.
        shared_emb = self.shared_base(x_main)
        base_y0 = self.base_head_0(shared_emb).squeeze(-1)
        base_y1 = self.base_head_1(shared_emb).squeeze(-1)
        
        # Stratum residuals.
        r0_never = self.res_never['res_0'](shared_emb).squeeze(-1)
        r1_never = self.res_never['res_1'](shared_emb).squeeze(-1)
        
        r0_comp = self.res_comp['res_0'](shared_emb).squeeze(-1)
        r1_comp = self.res_comp['res_1'](shared_emb).squeeze(-1)
        
        r0_always = self.res_always['res_0'](shared_emb).squeeze(-1)
        r1_always = self.res_always['res_1'](shared_emb).squeeze(-1)
        
        # Prediction = base + probability-weighted residual.
        y0 = base_y0 + (pi_00 * r0_never + pi_01 * r0_comp + pi_11 * r0_always)
        y1 = base_y1 + (pi_00 * r1_never + pi_01 * r1_comp + pi_11 * r1_always)
        
        if not self.training:
            pi_dict["wb_shared_emb"] = shared_emb
            
            # Base score.
            pi_dict["wb_base_y0"] = base_y0
            pi_dict["wb_base_y1"] = base_y1
            pi_dict["wb_base_u"] = base_y1 - base_y0
            
            # Never-taker residual.
            pi_dict["wb_res_y0_never"] = r0_never
            pi_dict["wb_res_y1_never"] = r1_never
            pi_dict["wb_res_u_never"] = r1_never - r0_never
            
            # Complier residual.
            pi_dict["wb_res_y0_comp"] = r0_comp
            pi_dict["wb_res_y1_comp"] = r1_comp
            pi_dict["wb_res_u_comp"] = r1_comp - r0_comp
            
            # Always-taker residual.
            pi_dict["wb_res_y0_always"] = r0_always
            pi_dict["wb_res_y1_always"] = r1_always
            pi_dict["wb_res_u_always"] = r1_always - r0_always

        return y0, y1, pi_dict


import torch
import torch.nn as nn

# ==========================================
# Rank-truncated residual mixture.
# ==========================================
class TARNET_V7_Truncated_MoE(TARNET_Residual_MoE):
    """
    Truncates the mediator prior by a within-batch rank.
    The cutoff is a top-k percentile of the current batch.
    An exponential moving average of that cutoff is used at evaluation.
    """
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 hidden_dims: list, dropout_rate: float, 
                 c_model: nn.Module = None, embedding_dim: int = 8,
                 truncation_mode: str = "hard",  # "hard" or "soft".
                 truncation_pct: float = 0.05,   # 0.05 keeps the top 5 percent.
                 truncation_temp: float = 10.0,  # Temperature of the soft cutoff.
                 ema_momentum: float = 0.9):     # Momentum of the cutoff average.
        
        # Residual-mixture backbone.
        super().__init__(continuous_dim, categorical_cardinalities, hidden_dims, 
                         dropout_rate, c_model, embedding_dim)
        
        self.truncation_mode = truncation_mode
        self.truncation_pct = truncation_pct
        self.truncation_temp = truncation_temp
        self.ema_momentum = ema_momentum
        
        # Cutoff state. Saved in state_dict.
        self.register_buffer('threshold_ema', torch.tensor(0.0))
        self.register_buffer('ema_initialized', torch.tensor(False))

    def forward(self, x_cont, x_cat):
        x_main = self.encoder(x_cont, x_cat)
        
        # Stratum probabilities from the frozen mediator.
        pi_00, pi_01, pi_11 = self.extract_pi_prior(x_cont, x_cat)
        pi_dict = {"p_never": pi_00, "p_complier": pi_01, "p_always": pi_11}
        
        # ==========================================
        # Rank cutoff.
        # ==========================================
        if self.truncation_mode != "none" and pi_01 is not None:
            if self.training:
                # Training cutoff: top-k percentile of this batch.
                q = 1.0 - self.truncation_pct
                # detach: the percentile is a threshold, not a path for gradients.
                # batch_thresh = torch.quantile(pi_01.detach(), q)
                batch_thresh = torch.quantile(pi_01.detach().float(), q)
                # In-place update of the moving average.
                if not self.ema_initialized:
                    # First batch copies the cutoff. Later batches average it.
                    self.threshold_ema.copy_(batch_thresh)
                    self.ema_initialized.fill_(True)
                else:
                    # Later batches.
                    new_ema = self.ema_momentum * self.threshold_ema + (1.0 - self.ema_momentum) * batch_thresh
                    self.threshold_ema.copy_(new_ema)
                
                # Training uses the cutoff of the current batch.
                current_thresh = batch_thresh
            else:
                # Evaluation uses the averaged cutoff.
                current_thresh = self.threshold_ema

            # Gate.
            if self.truncation_mode == "hard":
                # Hard gate: 1 above the cutoff, 0 below.
                gate = (pi_01 >= current_thresh.float()).float()
            elif self.truncation_mode == "soft":
                # Soft gate: sigmoid around the cutoff.
                scale = self.truncation_temp / (pi_01.std().clamp(min=1e-5))
                gate = torch.sigmoid(scale * (pi_01 - current_thresh.float()))
            
            # Rows below the cutoff lose the expert residual and fall back to the base.
            pi_00 = pi_00 * gate
            pi_01 = pi_01 * gate
            pi_11 = pi_11 * gate
        # ==========================================
        
        # Base logits.
        shared_emb = self.shared_base(x_main)
        base_y0 = self.base_head_0(shared_emb).squeeze(-1)
        base_y1 = self.base_head_1(shared_emb).squeeze(-1)
        
        # Residuals. The gate zeros them outside the kept rank.
        r0_never, r1_never = self.res_never['res_0'](shared_emb).squeeze(-1), self.res_never['res_1'](shared_emb).squeeze(-1)
        r0_comp, r1_comp   = self.res_comp['res_0'](shared_emb).squeeze(-1), self.res_comp['res_1'](shared_emb).squeeze(-1)
        r0_always, r1_always = self.res_always['res_0'](shared_emb).squeeze(-1), self.res_always['res_1'](shared_emb).squeeze(-1)
        
        # Prediction = base + gated, probability-weighted residual.
        y0 = base_y0 + (pi_00 * r0_never + pi_01 * r0_comp + pi_11 * r0_always)
        y1 = base_y1 + (pi_00 * r1_never + pi_01 * r1_comp + pi_11 * r1_always)
        if not self.training:
            pi_dict["wb_shared_emb"] = shared_emb
            if 'gate_for_wb' in locals() and gate is not None:
                pi_dict["wb_v7_gate"] = gate
                
            # Probability after the rank gate.
            pi_dict["wb_final_pi_never"] = pi_00
            pi_dict["wb_final_pi_comp"] = pi_01
            pi_dict["wb_final_pi_always"] = pi_11
                
            pi_dict["wb_base_y0"] = base_y0
            pi_dict["wb_base_y1"] = base_y1
            pi_dict["wb_base_u"] = base_y1 - base_y0
            
            pi_dict["wb_res_y0_never"] = r0_never
            pi_dict["wb_res_y1_never"] = r1_never
            pi_dict["wb_res_u_never"] = r1_never - r0_never
            
            pi_dict["wb_res_y0_comp"] = r0_comp
            pi_dict["wb_res_y1_comp"] = r1_comp
            pi_dict["wb_res_u_comp"] = r1_comp - r0_comp
            
            pi_dict["wb_res_y0_always"] = r0_always
            pi_dict["wb_res_y1_always"] = r1_always
            pi_dict["wb_res_u_always"] = r1_always - r0_always
        return y0, y1, pi_dict




import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# Eight gate variants. Scheme 6 adds an offset in logit space.
# ==========================================
class TARNET_V8_Evolution_MoE(TARNET_Residual_MoE):
    """
    Eight gate variants. Schemes 1-3 can initialize from a top-k percentile.
    """
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 hidden_dims: list, dropout_rate: float, 
                 c_model: nn.Module = None, embedding_dim: int = 8,
                 v8_scheme: int = 3,             # 1-5 unchanged. 6: logit offset. 7: interpolation. 8: prior features, then a product.
                 shared_emb_dim: int = 128,      
                 truncation_pct: float = 0.05,   
                 truncation_temp: float = 10.0,
                 ema_momentum: float = 0.9,
                 align_temp: float = 1.0,
                 head_hidden_dims: list = None):
        
        super().__init__(continuous_dim, categorical_cardinalities, hidden_dims, 
                         dropout_rate, c_model, embedding_dim,head_hidden_dims=head_hidden_dims)
        
        self.v8_scheme = v8_scheme
        self.truncation_pct = truncation_pct
        self.truncation_temp = truncation_temp
        self.ema_momentum = ema_momentum
        self.temp = align_temp
        
        # Moving-average anchor. Schemes 1, 2, and 3.
        if self.v8_scheme in [1, 2, 3]:
            self.register_buffer('threshold_ema', torch.tensor(0.0))
            self.register_buffer('ema_initialized', torch.tensor(False))

        # Scheme-specific head.
        if self.v8_scheme == 1:
            self.dynamic_net = nn.Sequential(
                nn.Linear(shared_emb_dim, shared_emb_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_emb_dim // 2, 1) 
            )
            nn.init.zeros_(self.dynamic_net[-1].weight)
            nn.init.zeros_(self.dynamic_net[-1].bias)

        elif self.v8_scheme == 2:
            self.dynamic_net = nn.Linear(shared_emb_dim, 1)
            nn.init.zeros_(self.dynamic_net.weight)
            nn.init.zeros_(self.dynamic_net.bias)

        elif self.v8_scheme == 3:
            self.dynamic_net = nn.Linear(shared_emb_dim, 3) 
            nn.init.zeros_(self.dynamic_net.weight)
            nn.init.zeros_(self.dynamic_net.bias)

        elif self.v8_scheme == 4:
            self.dynamic_net = nn.Sequential(
                nn.Linear(shared_emb_dim, shared_emb_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_emb_dim // 2, 3) 
            )

        elif self.v8_scheme == 5:
            self.dynamic_net = nn.Linear(shared_emb_dim, 4)
            nn.init.xavier_normal_(self.dynamic_net.weight)
            nn.init.zeros_(self.dynamic_net.bias)
            nn.init.constant_(self.dynamic_net.bias[3], 2.83) 

        # Schemes 6, 7, and 8.
        elif self.v8_scheme == 6:
            # Scheme 6: convert the prior to logits, add a feature offset, then softmax.
            self.dynamic_net = nn.Sequential(
                nn.Linear(shared_emb_dim, shared_emb_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_emb_dim // 2, 3) 
            )
            nn.init.zeros_(self.dynamic_net[-1].weight)
            nn.init.zeros_(self.dynamic_net[-1].bias)

        elif self.v8_scheme == 7:
            # Scheme 7: a feature network predicts probabilities and one trust score.
            self.dynamic_net = nn.Sequential(
                nn.Linear(shared_emb_dim, shared_emb_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_emb_dim // 2, 4) 
            )
            nn.init.zeros_(self.dynamic_net[-1].weight)
            nn.init.zeros_(self.dynamic_net[-1].bias)
            nn.init.constant_(self.dynamic_net[-1].bias[3], 2.0) # Initial trust score favors the prior.

        elif self.v8_scheme == 8:
            # Scheme 8: concatenate the three prior probabilities, then a product gate.
            self.dynamic_net = nn.Sequential(
                nn.Linear(shared_emb_dim + 3, shared_emb_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_emb_dim // 2, 3) 
            )
        # ==========================================================

    def forward(self, x_cont, x_cat):
        x_main = self.encoder(x_cont, x_cat)
        shared_emb = self.shared_base(x_main)
        
        pi_00, pi_01, pi_11 = self.extract_pi_prior(x_cont, x_cat)
        pi_dict = {"p_never": pi_00, "p_complier": pi_01, "p_always": pi_11}
        
        if self.v8_scheme in [1, 2, 3] and pi_01 is not None:
            # Anchor update. Same as the earlier schemes.
            if self.training:
                q = 1.0 - self.truncation_pct
                batch_thresh = torch.quantile(pi_01.detach().float(), q)
                if not self.ema_initialized:
                    self.threshold_ema.copy_(batch_thresh)
                    self.ema_initialized.fill_(True)
                else:
                    new_ema = self.ema_momentum * self.threshold_ema + (1.0 - self.ema_momentum) * batch_thresh
                    self.threshold_ema.copy_(new_ema)
                current_thresh = batch_thresh
            else:
                current_thresh = self.threshold_ema

            scale = self.truncation_temp / (pi_01.std().clamp(min=1e-5))
            
            if self.v8_scheme == 1:
                delta_c = self.dynamic_net(shared_emb).squeeze(-1) 
                gate_val = torch.sigmoid(scale * (pi_01 - (current_thresh - delta_c)))
                gate_00, gate_01, gate_11 = gate_val, gate_val, gate_val

            elif self.v8_scheme == 2:
                v = self.dynamic_net(shared_emb).squeeze(-1) 
                prior_bias = scale * (pi_01 - current_thresh)
                gate_val = torch.sigmoid(v + prior_bias)
                gate_00, gate_01, gate_11 = gate_val, gate_val, gate_val

            elif self.v8_scheme == 3:
                v = self.dynamic_net(shared_emb) 
                prior_bias = scale * (pi_01 - current_thresh)
                gates = torch.sigmoid(v + prior_bias.unsqueeze(-1))
                gate_00, gate_01, gate_11 = gates[:, 0], gates[:, 1], gates[:, 2]

        elif self.v8_scheme == 4:
            gates = torch.sigmoid(self.dynamic_net(shared_emb))
            gate_00, gate_01, gate_11 = gates[:, 0], gates[:, 1], gates[:, 2]

        elif self.v8_scheme == 5:
            gates = F.softmax(self.dynamic_net(shared_emb), dim=-1)
            pi_00_final, pi_01_final, pi_11_final = gates[:, 0], gates[:, 1], gates[:, 2]

        # Forward pass for schemes 6, 7, and 8.
        elif self.v8_scheme == 6:
            feature_offsets = self.dynamic_net(shared_emb)
            # Scheme 6.
            pi_prior_3d = torch.stack([pi_00, pi_01, pi_11], dim=1)
            logit_prior = torch.log(pi_prior_3d + 1e-7)
            pi_final_3d = F.softmax((logit_prior + feature_offsets) / self.temp, dim=-1)
            pi_00_final, pi_01_final, pi_11_final = pi_final_3d[:, 0], pi_final_3d[:, 1], pi_final_3d[:, 2]

        elif self.v8_scheme == 7:
            out = self.dynamic_net(shared_emb) 
            pi_feature_3d = F.softmax(out[:, :3], dim=-1)
            alpha = torch.sigmoid(out[:, 3]) # [B]
            pi_prior_3d = torch.stack([pi_00, pi_01, pi_11], dim=1)
            # Scheme 7.
            pi_final_3d = alpha.unsqueeze(-1) * pi_prior_3d + (1.0 - alpha.unsqueeze(-1)) * pi_feature_3d
            pi_00_final, pi_01_final, pi_11_final = pi_final_3d[:, 0], pi_final_3d[:, 1], pi_final_3d[:, 2]

        elif self.v8_scheme == 8:
            pi_prior_3d = torch.stack([pi_00, pi_01, pi_11], dim=1)
            # Scheme 8.
            enhanced_emb = torch.cat([shared_emb, pi_prior_3d], dim=-1)
            gates = torch.sigmoid(self.dynamic_net(enhanced_emb))
            gate_00, gate_01, gate_11 = gates[:, 0], gates[:, 1], gates[:, 2]

        # Apply the gate. Scheme 8 also returns a gate, so it is included.
        if self.v8_scheme in [1, 2, 3, 4, 8]:
            pi_00_final = pi_00 * gate_00
            pi_01_final = pi_01 * gate_01
            pi_11_final = pi_11 * gate_11

        # Base and residual. Shared with the parent.
        base_y0 = self.base_head_0(shared_emb).squeeze(-1)
        base_y1 = self.base_head_1(shared_emb).squeeze(-1)
        r0_never, r1_never = self.res_never['res_0'](shared_emb).squeeze(-1), self.res_never['res_1'](shared_emb).squeeze(-1)
        r0_comp, r1_comp   = self.res_comp['res_0'](shared_emb).squeeze(-1), self.res_comp['res_1'](shared_emb).squeeze(-1)
        r0_always, r1_always = self.res_always['res_0'](shared_emb).squeeze(-1), self.res_always['res_1'](shared_emb).squeeze(-1)
        
        y0 = base_y0 + (pi_00_final * r0_never + pi_01_final * r0_comp + pi_11_final * r0_always)
        y1 = base_y1 + (pi_00_final * r1_never + pi_01_final * r1_comp + pi_11_final * r1_always)

        if not self.training:
            pi_dict["wb_shared_emb"] = shared_emb
            pi_dict["wb_final_pi_never"] = pi_00_final
            pi_dict["wb_final_pi_comp"] = pi_01_final
            pi_dict["wb_final_pi_always"] = pi_11_final
            
            if 'gate_00' in locals():
                pi_dict["wb_gate_never"] = gate_00
                pi_dict["wb_gate_comp"] = gate_01
                pi_dict["wb_gate_always"] = gate_11
                
            pi_dict["wb_base_y0"] = base_y0
            pi_dict["wb_base_y1"] = base_y1
            pi_dict["wb_base_u"] = base_y1 - base_y0
            pi_dict["wb_res_y0_never"] = r0_never
            pi_dict["wb_res_y1_never"] = r1_never
            pi_dict["wb_res_u_never"] = r1_never - r0_never
            pi_dict["wb_res_y0_comp"] = r0_comp
            pi_dict["wb_res_y1_comp"] = r1_comp
            pi_dict["wb_res_u_comp"] = r1_comp - r0_comp
            pi_dict["wb_res_y0_always"] = r0_always
            pi_dict["wb_res_y1_always"] = r1_always
            pi_dict["wb_res_u_always"] = r1_always - r0_always
        return y0, y1, pi_dict


# ==========================================
# Moving-average alignment of the mediator prior.
# ==========================================
class DynamicPriorAlignmentLayer(nn.Module):
    def __init__(self, align_type='lift', momentum=0.05, eps=1e-7):
        super().__init__()
        self.align_type = align_type
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(3, dtype=torch.float32))
        self.register_buffer('running_var', torch.ones(3, dtype=torch.float32))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, priors):
        if self.training:
            batch_mean = priors.mean(dim=0)
            batch_var = priors.var(dim=0, unbiased=False)
            with torch.no_grad():
                if self.num_batches_tracked == 0:
                    self.running_mean.copy_(batch_mean)
                    self.running_var.copy_(batch_var)
                else:
                    self.running_mean.copy_((1 - self.momentum) * self.running_mean + self.momentum * batch_mean)
                    self.running_var.copy_((1 - self.momentum) * self.running_var + self.momentum * batch_var)
                self.num_batches_tracked += 1
                
        global_mean = self.running_mean
        global_std = torch.sqrt(self.running_var + self.eps)

        if self.align_type == 'lift':
            aligned_logits = torch.log((priors + self.eps) / (global_mean + self.eps))
        elif self.align_type == 'z_score':
            aligned_logits = (priors - global_mean) / (global_std + self.eps)
        else:
            aligned_logits = priors
            
        return aligned_logits


# ==========================================
# Logit-space alignment. Schemes 4, 6, 7, and 8.
# ==========================================
class TARNET_V11_Aligned_MoE(TARNET_Residual_MoE): 
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 hidden_dims: list, dropout_rate: float, 
                 c_model: nn.Module = None, embedding_dim: int = 8,
                 v11_scheme: int = 4,            
                 align_type: str = 'lift',       
                 align_temp: float = 1.0):       
        
        super().__init__(continuous_dim, categorical_cardinalities, hidden_dims, 
                         dropout_rate, c_model, embedding_dim)
        self.v11_scheme = v11_scheme
        self.temp = align_temp
        shared_emb_dim = hidden_dims[-1]
        self.aligner = DynamicPriorAlignmentLayer(align_type=align_type)

        if self.v11_scheme == 4:
            # Scheme 4: feature MLP, then a product gate.
            self.dynamic_net = nn.Sequential(
                nn.Linear(shared_emb_dim, shared_emb_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_emb_dim // 2, 3) 
            )
        elif self.v11_scheme == 6:
            # Scheme 6: add an offset in logit space.
            self.dynamic_net = nn.Sequential(
                nn.Linear(shared_emb_dim, shared_emb_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_emb_dim // 2, 3) 
            )
            nn.init.zeros_(self.dynamic_net[-1].weight)
            nn.init.zeros_(self.dynamic_net[-1].bias)
        elif self.v11_scheme == 7:
            # Scheme 7: interpolate with a predicted trust score.
            self.dynamic_net = nn.Sequential(
                nn.Linear(shared_emb_dim, shared_emb_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_emb_dim // 2, 4) 
            )
            nn.init.zeros_(self.dynamic_net[-1].weight)
            nn.init.zeros_(self.dynamic_net[-1].bias)
            nn.init.constant_(self.dynamic_net[-1].bias[3], 2.0)
        elif self.v11_scheme == 8:
            # Scheme 8: concatenate the prior, then a product.
            self.dynamic_net = nn.Sequential(
                nn.Linear(shared_emb_dim + 3, shared_emb_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_emb_dim // 2, 3) 
            )

    def forward(self, x_cont, x_cat):
        x_main = self.encoder(x_cont, x_cat)
        shared_emb = self.shared_base(x_main)
        
        pi_00, pi_01, pi_11 = self.extract_pi_prior(x_cont, x_cat)
        pi_dict = {"p_never": pi_00, "p_complier": pi_01, "p_always": pi_11}
        raw_priors = torch.stack([pi_00, pi_01, pi_11], dim=1) 
        
        # Aligned log-odds and the reference probability.
        aligned_logits = self.aligner(raw_priors) 
        aligned_probs = F.softmax(aligned_logits / self.temp, dim=-1)

        # Dispatch.
        if self.v11_scheme == 4:
            feature_gates = torch.sigmoid(self.dynamic_net(shared_emb))
            pi_final_3d = aligned_probs * feature_gates

        elif self.v11_scheme == 6:
            feature_offsets = self.dynamic_net(shared_emb)
            # Logit addition: aligned log-odds plus a feature offset, then softmax.
            pi_final_3d = F.softmax((aligned_logits + feature_offsets) / self.temp, dim=-1)

        elif self.v11_scheme == 7:
            out = self.dynamic_net(shared_emb)
            feature_probs = F.softmax(out[:, :3] / self.temp, dim=-1)
            alpha = torch.sigmoid(out[:, 3:4])
            pi_final_3d = alpha * aligned_probs + (1.0 - alpha) * feature_probs

        elif self.v11_scheme == 8:
            # Concatenate the aligned log-odds onto the features.
            enhanced_emb = torch.cat([shared_emb, aligned_logits], dim=-1)
            feature_gates = torch.sigmoid(self.dynamic_net(enhanced_emb))
            pi_final_3d = aligned_probs * feature_gates

        pi_00_final, pi_01_final, pi_11_final = pi_final_3d[:, 0], pi_final_3d[:, 1], pi_final_3d[:, 2]

        # Base plus residual experts.
        base_y0, base_y1 = self.base_head_0(shared_emb).squeeze(-1), self.base_head_1(shared_emb).squeeze(-1)
        r0_never, r1_never = self.res_never['res_0'](shared_emb).squeeze(-1), self.res_never['res_1'](shared_emb).squeeze(-1)
        r0_comp, r1_comp   = self.res_comp['res_0'](shared_emb).squeeze(-1), self.res_comp['res_1'](shared_emb).squeeze(-1)
        r0_always, r1_always = self.res_always['res_0'](shared_emb).squeeze(-1), self.res_always['res_1'](shared_emb).squeeze(-1)
        
        y0 = base_y0 + (pi_00_final * r0_never + pi_01_final * r0_comp + pi_11_final * r0_always)
        y1 = base_y1 + (pi_00_final * r1_never + pi_01_final * r1_comp + pi_11_final * r1_always)

        if not self.training:
            pi_dict["wb_aligned_logit_comp"] = aligned_logits[:, 1]
            pi_dict["wb_final_pi_comp"] = pi_01_final

        return y0, y1, pi_dict



# ------------------------single task baseline
# ==========================================
# DragonNet.
# ==========================================
class DragonNet(nn.Module):
    def __init__(self, continuous_dim, categorical_cardinalities, hidden_dims=[128, 64, 32]):
        super().__init__()
        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities)
        
        # Shared representation. ELU, as in the paper.
        layers = []
        curr_dim = self.encoder.output_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(curr_dim, dim), nn.ELU()])
            curr_dim = dim
        self.shared_rep = nn.Sequential(*layers)
        
        # Three heads: control outcome, treated outcome, propensity.
        self.head_y0 = nn.Sequential(nn.Linear(curr_dim, 1))
        self.head_y1 = nn.Sequential(nn.Linear(curr_dim, 1))
        # The propensity head is one linear layer.
        self.head_t = nn.Sequential(nn.Linear(curr_dim, 1))
        
        # Targeted-regularization coefficient.
        self.epsilon = nn.Parameter(torch.tensor([0.0], requires_grad=True))

    def forward(self, x_cont, x_cat):
        z = self.shared_rep(self.encoder(x_cont, x_cat))
        y0_logit = self.head_y0(z).squeeze(-1)
        y1_logit = self.head_y1(z).squeeze(-1)
        t_logit = self.head_t(z).squeeze(-1)
        
        # t_logit and epsilon go to the loss through pi_dict.
        return y0_logit, y1_logit, {"t_logit": t_logit, "epsilon": self.epsilon}


# =========================================================================
# EUEN with separate control and uplift towers.
# Separate encoders, a detached residual, Kaiming initialization, and a head bias of 0.1.
# =========================================================================
class EUEN_Academic(nn.Module):
    def __init__(self, continuous_dim, categorical_cardinalities, hidden_dims=[128, 64], dropout_rate=0.0):
        super().__init__()
        
        # The two encoders do not share weights.
        self.encoder_control = FeatureEncoder(continuous_dim, categorical_cardinalities)
        self.encoder_uplift = FeatureEncoder(continuous_dim, categorical_cardinalities)
        
        c_in_dim = self.encoder_control.output_dim
        u_in_dim = self.encoder_uplift.output_dim
        
        # Control tower: baseline response.
        c_layers = []
        curr_dim = c_in_dim
        for dim in hidden_dims:
            c_layers.extend([nn.Linear(curr_dim, dim), nn.ELU()])
            if dropout_rate > 0: c_layers.append(nn.Dropout(dropout_rate))
            curr_dim = dim
        self.control_mlp = nn.Sequential(*c_layers)
        self.control_head = nn.Linear(curr_dim, 1)
        
        # Uplift tower: treatment increment. Weights are not shared with the control tower.
        u_layers = []
        curr_dim = u_in_dim
        for dim in hidden_dims:
            u_layers.extend([nn.Linear(curr_dim, dim), nn.ELU()])
            if dropout_rate > 0: u_layers.append(nn.Dropout(dropout_rate))
            curr_dim = dim
        self.uplift_mlp = nn.Sequential(*u_layers)
        self.uplift_head = nn.Linear(curr_dim, 1)
        
        # Kaiming normal initialization. Head bias is 0.1.
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Head bias.
        nn.init.constant_(self.control_head.bias, 0.1)
        nn.init.constant_(self.uplift_head.bias, 0.1)

    def forward(self, x_cont, x_cat):
        # Separate representations.
        c_feat = self.encoder_control(x_cont, x_cat)
        u_feat = self.encoder_uplift(x_cont, x_cat)
        
        # Control logit.
        c_logit = self.control_head(self.control_mlp(c_feat)).squeeze(-1)
        # Increment.
        u_tau = self.uplift_head(self.uplift_mlp(u_feat)).squeeze(-1)
        
        y0_logit = c_logit
        # Stop the gradient through the control logit.
        y1_logit = c_logit.detach() + u_tau
        
        # No stratum probabilities.
        return y0_logit, y1_logit, {}

# ==========================================
# Explicit uplift network with one shared encoder.
# ==========================================
class EUEN(nn.Module):
    def __init__(self, continuous_dim, categorical_cardinalities, hidden_dims=[128, 64]):
        super().__init__()
        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities)
        
        # Shared encoder, same layout as TARNet.
        layers = []
        curr_dim = self.encoder.output_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(curr_dim, dim), nn.ReLU()])
            curr_dim = dim
        self.shared_bottom = nn.Sequential(*layers)
        
        # One linear layer per head, so the parameter count matches the other baselines.
        # Control response.
        self.base_tower = nn.Linear(curr_dim, 1)
        # Treatment increment.
        self.uplift_tower = nn.Linear(curr_dim, 1)

    def forward(self, x_cont, x_cat):
        z = self.shared_bottom(self.encoder(x_cont, x_cat))
        
        # One linear layer. Drop the trailing dimension.
        base_logit = self.base_tower(z).squeeze(-1)     # \mu_c(X)
        uplift_logit = self.uplift_tower(z).squeeze(-1) # \tau(X)
        
        # Add in logit space.
        y0_logit = base_logit
        y1_logit = base_logit + uplift_logit
        
        return y0_logit, y1_logit, {}


# =========================================================================
# EFIN. Control and uplift branches split after the encoder.
# =========================================================================
# =========================================================================
# EFIN. Control and uplift branches split after the encoder.
# =========================================================================


class EFIN_ours(nn.Module):
    def __init__(self, continuous_dim, categorical_cardinalities, embed_dim=16, hidden_dims=[128, 64]):
        super().__init__()
        self.cont_dim = continuous_dim if continuous_dim is not None else 0
        self.cat_cards = categorical_cardinalities or {}
        self.embed_dim = embed_dim
        
        # -------------------------------------------
        # Module 1: The Feature Encoder (Paper Eq 3)
        # Each feature is encoded to embed_dim on its own. Do not concatenate first.
        # -------------------------------------------
        # Continuous feature: its own linear map.
        if self.cont_dim > 0:
            self.cont_W = nn.Parameter(torch.empty(self.cont_dim, embed_dim))
            self.cont_b = nn.Parameter(torch.empty(self.cont_dim, embed_dim))
            nn.init.xavier_uniform_(self.cont_W)
            nn.init.zeros_(self.cont_b)
            
        # Categorical feature: embedding table.
        self.cat_embs = nn.ModuleDict({
            col: nn.Embedding(card, embed_dim) for col, card in self.cat_cards.items()
        })
        
        # Treatment embedding for the treated arm.
        self.t_emb = nn.Embedding(2, embed_dim) 
        
        self.num_features = self.cont_dim + len(self.cat_cards)
        
        # -------------------------------------------
        # Module 2: The Self-interaction (Paper Eq 4, 5, 6)
        # Control response.
        # -------------------------------------------
        # Self-attention, as in the paper.
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=2, batch_first=True)
        
        # Control MLP. Depth follows the config.
        natural_layers = []
        curr_dim = self.num_features * embed_dim
        for dim in hidden_dims:
            natural_layers.extend([nn.Linear(curr_dim, dim), nn.ReLU()])
            curr_dim = dim
        natural_layers.append(nn.Linear(curr_dim, 1))
        self.natural_mlp = nn.Sequential(*natural_layers)
        
        # -------------------------------------------
        # Module 3: Treatment-aware Interaction (Paper Eq 8, 9, 10)
        # Uplift branch.
        # -------------------------------------------
        self.W_t1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_t2 = nn.Linear(embed_dim, embed_dim, bias=True)
        self.W_t0 = nn.Linear(embed_dim, 1, bias=False)
        
        # Uplift MLP. Depth follows the config.
        uplift_layers = []
        curr_dim_uplift = embed_dim
        for dim in hidden_dims:
            uplift_layers.extend([nn.Linear(curr_dim_uplift, dim), nn.ReLU()])
            curr_dim_uplift = dim
        uplift_layers.append(nn.Linear(curr_dim_uplift, 1))
        self.uplift_mlp = nn.Sequential(*uplift_layers)
        
        # -------------------------------------------
        # Module 4: Intervention Constraint (Paper Eq 13)
        # Predict the flipped treatment label. Used by the constraint loss.
        # -------------------------------------------
        self.constraint_mlp = nn.Linear(embed_dim, 1)

    def forward(self, x_cont, x_cat):
        device = x_cont.device if x_cont is not None else x_cat[list(x_cat.keys())[0]].device
        B = x_cont.shape[0] if x_cont is not None else x_cat[list(x_cat.keys())[0]].shape[0]
        
        # ==================================
        # Encode each field. Shape [B, F, D].
        # ==================================
        feat_embs = []
        if self.cont_dim > 0:
            # (B, F_cont, 1) * (1, F_cont, D) -> (B, F_cont, D)
            cont_e = x_cont.unsqueeze(-1) * self.cont_W.unsqueeze(0) + self.cont_b.unsqueeze(0)
            feat_embs.append(cont_e)
        if len(self.cat_cards) > 0:
            cat_e = [self.cat_embs[col](x_cat[col]).unsqueeze(1) for col in self.cat_cards.keys()]
            feat_embs.append(torch.cat(cat_e, dim=1))
            
        X = torch.cat(feat_embs, dim=1) # Shape: (B, F, D)
        
        # ==================================
        # Control response y(0).
        # ==================================
        attn_out, _ = self.self_attn(X, X, X) # (B, F, D)
        # Flatten, then the MLP.
        y0_logit = self.natural_mlp(attn_out.reshape(B, -1)).squeeze(-1) 
        
        # ==================================
        # Treated response y(1) = y(0) + tau.
        # ==================================
        # Embedding of the treated arm.
        t1_idx = torch.ones((B,), dtype=torch.long, device=device)
        e_t = self.t_emb(t1_idx) # (B, D)
        
        # Attention scores.
        interact = F.relu(self.W_t1(e_t).unsqueeze(1) + self.W_t2(X)) # (B, F, D)
        alpha = F.softmax(self.W_t0(interact), dim=1) # (B, F, 1)
        
        # Weighted sum.
        e_xt = torch.sum(alpha * X, dim=1) # (B, D)
        
        # Increment.
        tau_1 = self.uplift_mlp(e_xt).squeeze(-1) # (B,)
        
        # Add the increment.
        y1_logit = y0_logit + tau_1 
        
        # ==================================
        # Treatment logit for the constraint loss.
        # ==================================
        t_constraint_logit = self.constraint_mlp(e_xt).squeeze(-1)
        
        return y0_logit, y1_logit, {"efin_constraint_logit": t_constraint_logit}



class EFIN(nn.Module):
    """
    EFIN aligned with the official module.
    Layout:
    1. L2 normalization runs over the K axis (dim=1).
    2. Features are projected in chunks. The full vector is not passed through one linear layer.
    3. Three-layer funnel. Attention layers use bias=False.
    4. The width floor is max(R // X, 1).
    """
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict, 
                 efin_rank: int = 64, dropout_rate: float = 0.0):
        super().__init__()
        
        self.R = efin_rank  # Hidden width.
        categorical_cardinalities = categorical_cardinalities or {}
        self.K = continuous_dim + len(categorical_cardinalities)  # Number of field tokens.
        
        # Chunked projection.
        self.base_encoder = FeatureEncoder(continuous_dim, categorical_cardinalities)
        input_dim = self.base_encoder.output_dim
        
        # Width of one chunk.
        self.chunk_dim = input_dim // self.K
        # One projection per chunk, producing [B, K, R].
        self.virtual_projectors = nn.ModuleList([
            nn.Linear(self.chunk_dim, self.R) for _ in range(self.K)
        ])
        
        # -----------------------------------------------------------------
        # Control branch. Three layers. Attention bias is off.
        # -----------------------------------------------------------------
        self.W_q = nn.Linear(self.R, self.R, bias=False)
        self.W_k = nn.Linear(self.R, self.R, bias=False)
        self.W_v = nn.Linear(self.R, self.R, bias=False)
        self.scale = self.R ** 0.5
        
        # Three layers: K*R -> R, R -> R/2, R/2 -> R/4.
        self.control_mlp = nn.Sequential(
            nn.Linear(self.K * self.R, self.R),
            nn.ELU(),
            nn.Linear(self.R, max(self.R // 2, 1)),
            nn.ELU(),
            nn.Linear(max(self.R // 2, 1), max(self.R // 4, 1)),
            nn.ELU()
        )
        self.c_logit_head = nn.Linear(max(self.R // 4, 1), 1)
        
        # -----------------------------------------------------------------
        # Uplift branch. Three layers. Attention bias is off.
        # -----------------------------------------------------------------
        self.t_rep_layer = nn.Linear(1, self.R, bias=False)
        nn.init.xavier_uniform_(self.t_rep_layer.weight)
        
        # Attention without bias, as in the official module.
        self.att1 = nn.Linear(self.R, self.R, bias=False)
        self.att2 = nn.Linear(self.R, self.R, bias=False)
        self.att3 = nn.Linear(self.R, 1, bias=False)
        
        self.uplift_trunk = nn.Sequential(
            nn.Linear(self.R, self.R),
            nn.ELU(),
            nn.Linear(self.R, max(self.R // 2, 1)),
            nn.ELU(),
            nn.Linear(max(self.R // 2, 1), max(self.R // 4, 1)),
            nn.ELU()
        )
        self.u_tau_head = nn.Linear(max(self.R // 4, 1), 1)
        self.intervention_head = nn.Linear(max(self.R // 4, 1), 1)

    def forward(self, x_cont, x_cat):
        B = x_cont.shape[0] if x_cont is not None else x_cat[list(x_cat.keys())[0]].shape[0]
        device = x_cont.device if x_cont is not None else next(self.parameters()).device
        
        # Full encoded vector.
        phi = self.base_encoder(x_cont, x_cat)
        
        # Split into K chunks and project each one to [B, K, R].
        x_chunks = [phi[:, i*self.chunk_dim : (i+1)*self.chunk_dim] for i in range(self.K)]
        x_rep = torch.stack([proj(chunk) for proj, chunk in zip(self.virtual_projectors, x_chunks)], dim=1)
        
        # =================================================================
        # Control branch.
        # =================================================================
        # L2 normalization over the K axis (dim=1).
        x_rep_norm = F.normalize(x_rep, p=2, dim=1)
        
        Q = self.W_q(x_rep_norm)
        K = self.W_k(x_rep_norm)
        V = self.W_v(x_rep_norm)
        
        scores = torch.sigmoid(torch.matmul(Q, K.transpose(-1, -2)) / self.scale)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)
        
        # Three-layer funnel -> control logit.
        c_hidden = self.control_mlp(attn_out.reshape(B, -1))
        c_logit = self.c_logit_head(c_hidden).squeeze(-1)
        
        # =================================================================
        # Uplift branch. The attention reads the un-normalized representation.
        # =================================================================
        ones_t = torch.ones((B, 1), dtype=torch.float32, device=device)
        t_rep = self.t_rep_layer(ones_t)
        
        # Interaction attention.
        t_part = torch.sigmoid(self.att1(t_rep)).unsqueeze(1)
        x_part = torch.sigmoid(self.att2(x_rep))
        
        alpha = F.softmax(self.att3(F.relu(t_part + x_part)), dim=1)
        e_xt = torch.sum(alpha * x_rep, dim=1)
        
        # Shared three-layer trunk -> treatment logit and increment.
        uplift_hidden = self.uplift_trunk(e_xt)
        u_tau = self.u_tau_head(uplift_hidden).squeeze(-1)
        t_logit = self.intervention_head(uplift_hidden).squeeze(-1)
        
        # =================================================================
        # y0 is the control logit. y1 is the detached control logit plus the increment.
        # =================================================================
        y0_logit = c_logit
        y1_logit = c_logit.detach() + u_tau
        
        return y0_logit, y1_logit, {"efin_t_logit": t_logit}

# ==========================================
# S-learner. Treatment is one more feature.
# Treatment is concatenated with the continuous features and one network fits both arms.
# ==========================================
class S_Learner(nn.Module):
    def __init__(self, continuous_dim, categorical_cardinalities, hidden_dims=[128, 64]):
        super().__init__()
        # One extra continuous input for the treatment indicator.
        cont_dim_with_t = (continuous_dim if continuous_dim is not None else 0) + 1
        self.encoder = FeatureEncoder(cont_dim_with_t, categorical_cardinalities)
        
        layers = []
        curr_dim = self.encoder.output_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(curr_dim, dim), nn.ReLU()])
            curr_dim = dim
        layers.append(nn.Linear(curr_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cont, x_cat, t=None):
        # Training uses the observed treatment.
        if t is not None:
            t_feat = t.unsqueeze(1).float()
            x_c = torch.cat([x_cont, t_feat], dim=1) if x_cont is not None else t_feat
            
            feat = self.encoder(x_c, x_cat)
            y_logit = self.mlp(feat).squeeze(-1)
            
            # compute_stage3_loss selects a logit with torch.where.
            # Both returned logits are the same value, so either branch is the observed prediction.
            return y_logit, y_logit, {}
            
        # Evaluation does not know the treatment.
        # Run the network twice, once with T=0 and once with T=1.
        else:
            device = x_cont.device if x_cont is not None else x_cat[list(x_cat.keys())[0]].device
            batch_size = x_cont.shape[0] if x_cont is not None else x_cat[list(x_cat.keys())[0]].shape[0]
            
            # T = 0.
            t0 = torch.zeros((batch_size, 1), device=device)
            x_c0 = torch.cat([x_cont, t0], dim=1) if x_cont is not None else t0
            y0_logit = self.mlp(self.encoder(x_c0, x_cat)).squeeze(-1)
            
            # T = 1.
            t1 = torch.ones((batch_size, 1), device=device)
            x_c1 = torch.cat([x_cont, t1], dim=1) if x_cont is not None else t1
            y1_logit = self.mlp(self.encoder(x_c1, x_cat)).squeeze(-1)
            
            return y0_logit, y1_logit, {}

# ==========================================
# T-learner. One network per treatment arm.
# No shared trunk. T=0 and T=1 each have a network.
# ==========================================
class T_Learner(nn.Module):
    def __init__(self, continuous_dim, categorical_cardinalities, hidden_dims=[128, 64]):
        super().__init__()
        
        # Control network.
        self.encoder0 = FeatureEncoder(continuous_dim, categorical_cardinalities)
        layers0 = []
        curr_dim0 = self.encoder0.output_dim
        for dim in hidden_dims:
            layers0.extend([nn.Linear(curr_dim0, dim), nn.ReLU()])
            curr_dim0 = dim
        layers0.append(nn.Linear(curr_dim0, 1))
        self.mlp0 = nn.Sequential(*layers0)
        
        # Treated network.
        self.encoder1 = FeatureEncoder(continuous_dim, categorical_cardinalities)
        layers1 = []
        curr_dim1 = self.encoder1.output_dim
        for dim in hidden_dims:
            layers1.extend([nn.Linear(curr_dim1, dim), nn.ReLU()])
            curr_dim1 = dim
        layers1.append(nn.Linear(curr_dim1, 1))
        self.mlp1 = nn.Sequential(*layers1)

    def forward(self, x_cont, x_cat):
        # Separate predictions.
        # The loss selects y1 or y0 with torch.where, so the unused arm gets no label gradient.
        # Treated rows update the treated network. Control rows update the control network.
        y0_logit = self.mlp0(self.encoder0(x_cont, x_cat)).squeeze(-1)
        y1_logit = self.mlp1(self.encoder1(x_cont, x_cat)).squeeze(-1)
        return y0_logit, y1_logit, {}


# ==========================================
# CFRNet.
# Two heads, plus a penalty that matches the treated and control representations.
# ==========================================
class CFRNet(nn.Module):
    def __init__(self, continuous_dim, categorical_cardinalities, hidden_dims=[128, 64, 32]):
        super().__init__()
        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities)
        
        # Shared representation.
        layers = []
        curr_dim = self.encoder.output_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(curr_dim, dim), nn.ReLU()])
            curr_dim = dim
        self.shared_rep = nn.Sequential(*layers)
        
        # Two heads.
        self.head_y0 = nn.Linear(curr_dim, 1) # nn.Sequential(nn.Linear(curr_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_y1 = nn.Linear(curr_dim, 1) # nn.Sequential(nn.Linear(curr_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x_cont, x_cat):
        # Shared representation.
        z = self.shared_rep(self.encoder(x_cont, x_cat))
        
        # Separate predictions.
        y0_logit = self.head_y0(z).squeeze(-1)
        y1_logit = self.head_y1(z).squeeze(-1)
        
        # z is returned so the loss can compare the two treatment groups.
        return y0_logit, y1_logit, {"z": z}


# ==========================================
# DESCN, KDD 2022.
# ==========================================
class DESCN(nn.Module):
    def __init__(self, continuous_dim, categorical_cardinalities, hidden_dims=[128, 64, 32], dropout_rate=0.0):
        """
        Shared network: three ELU layers and an L2 normalization. Four separate three-layer heads.
        """
        super().__init__()
        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities)
        input_dim = self.encoder.output_dim
        
        # Shared network: three fully connected ELU layers.
        # Input -> 128 -> 128 -> 64.
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU()
        )
        
        # Four heads. Each is three ELU layers and a linear logit.
        # 64 -> 64 -> 32 -> 1.
        def build_base_model():
            return nn.Sequential(
                nn.Linear(64, 64),
                nn.ELU(),
                nn.Linear(64, 64),
                nn.ELU(),
                nn.Linear(64, 32),
                nn.ELU(),
                nn.Linear(32, 1)
            )
            
        self.propensity_net = build_base_model()  # Propensity.
        self.cr_net = build_base_model()          # Control response.
        self.tr_net = build_base_model()          # Treated response.
        self.pte_net = build_base_model()         # Pseudo treatment effect.

    def forward(self, x_cont, x_cat):
        # Encode.
        feat = self.encoder(x_cont, x_cat)
        
        # Shared network.
        shared_rep = self.shared_net(feat)
        
        # L2-normalize the shared representation.
        shared_rep_norm = F.normalize(shared_rep, p=2, dim=-1)
        
        # Four logits.
        pi_logit = self.propensity_net(shared_rep_norm).squeeze(-1)
        mu0_logit = self.cr_net(shared_rep_norm).squeeze(-1)
        mu1_logit = self.tr_net(shared_rep_norm).squeeze(-1)
        tau_logit = self.pte_net(shared_rep_norm).squeeze(-1)
        
        # Same return layout as the other models.
        pi_dict = {
            "descn_pi_logit": pi_logit,
            "descn_tau_logit": tau_logit
        }
        
        return mu0_logit, mu1_logit, pi_dict


    



import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicPriorAlignmentLayer(nn.Module):
    def __init__(self, align_type='lift', momentum=0.05, eps=1e-7):
        """
        Running-mean alignment of the mediator prior.
        momentum is the weight on the new batch. 0.05 keeps 95 percent of the history.
        """
        super().__init__()
        self.align_type = align_type
        self.momentum = momentum
        self.eps = eps
        
        # Buffers are stored in state_dict and receive no gradient.
        self.register_buffer('running_mean', torch.zeros(3, dtype=torch.float32))
        self.register_buffer('running_var', torch.ones(3, dtype=torch.float32))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, priors):
        """
        priors: mediator probabilities, shape (B, 3), ordered pi_00, pi_01, pi_11.
        """
        # Update the running statistics in training mode.
        if self.training:
            # Batch mean and variance.
            batch_mean = priors.mean(dim=0)
            batch_var = priors.var(dim=0, unbiased=False)
            
            # Statistics only.
            with torch.no_grad():
                if self.num_batches_tracked == 0:
                    # First batch.
                    self.running_mean.copy_(batch_mean)
                    self.running_var.copy_(batch_var)
                else:
                    # Later batches.
                    self.running_mean.copy_((1 - self.momentum) * self.running_mean + self.momentum * batch_mean)
                    self.running_var.copy_((1 - self.momentum) * self.running_var + self.momentum * batch_var)
                
                self.num_batches_tracked += 1

        # The forward alignment uses the running mean, not the current batch mean.
        # A batch mean would move when one batch is not representative.
        global_mean = self.running_mean
        global_std = torch.sqrt(self.running_var + self.eps)

        # Map the probabilities.
        if self.align_type == 'lift':
            # Log-odds against the running mean: log(P / E[P]).
            aligned_logits = torch.log((priors + self.eps) / (global_mean + self.eps))
            
        elif self.align_type == 'z_score':
            # Standardized residual.
            aligned_logits = (priors - global_mean) / (global_std + self.eps)
            
        elif self.align_type == 'rank':
            # Within-batch percentile. Uses the current batch.
            B = priors.shape[0]
            if B > 1:
                ranks = priors.argsort(dim=0).argsort(dim=0).float() / (B - 1)
            else:
                ranks = torch.ones_like(priors) * 0.5
            aligned_logits = ranks
        else:
            aligned_logits = priors
            
        return aligned_logits

# ---------------------------------------------------------------------------
# Ours. Readable implementation: prism.py.
# These names stay so trainer configs and checkpoints keep working.
# S4 is the paper model. S6 is the logit-additive control.
# ---------------------------------------------------------------------------
class TARNET_Ours_S4_Conflict(PRISM):
    """Multiplicative gate. See prism.PRISM."""


class TARNET_Ours_S6_Conflict(PRISMLogitGate):
    """Logit-additive gate. See prism.PRISMLogitGate."""

import torch
import torch.nn as nn

# ==========================================
# EFIN matching the official KDD 2023 module.
# ==========================================
class EFIN_0717new(nn.Module):
    def __init__(self, continuous_dim, categorical_cardinalities=None,
                 hu_dim=128, hc_dim=64, is_self=False, act_type="elu",
                 dropout_rate=0.0):
        super().__init__()
        categorical_cardinalities = categorical_cardinalities or {}
        if len(categorical_cardinalities) > 0:
            raise NotImplementedError(
                "EFIN_0717new follows the official module and accepts continuous features only "
                "(x_rep = feature_list.unsqueeze(2) * Embedding.weight). It has no categorical branch. "
                "categorical_cardinalities={} is not that continuous contract. "
                "Extend the official formula before using categoricals. This class does not approximate that case.".format(categorical_cardinalities))
        
        self.nums_feature = continuous_dim if continuous_dim is not None else 0
        self.hu_dim = hu_dim
        self.hc_dim = hc_dim
        self.is_self = is_self

        self.att_embed_1 = nn.Linear(hu_dim, hu_dim, bias=False)
        self.att_embed_2 = nn.Linear(hu_dim, hu_dim)
        self.att_embed_3 = nn.Linear(hu_dim, 1, bias=False)

        self.Q_w = nn.Linear(hu_dim, hu_dim, bias=True)
        self.K_w = nn.Linear(hu_dim, hu_dim, bias=True)
        self.V_w = nn.Linear(hu_dim, hu_dim, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        self.x_rep = nn.Embedding(self.nums_feature, hu_dim)
        self.t_rep = nn.Linear(1, hu_dim)

        self.c_fc1 = nn.Linear(self.nums_feature * hu_dim, hc_dim)
        self.c_fc2 = nn.Linear(hc_dim, hc_dim)
        self.c_fc3 = nn.Linear(hc_dim, max(hc_dim // 2, 1))
        self.c_fc4 = nn.Linear(max(hc_dim // 2, 1), max(hc_dim // 4, 1))
        c_out_dim = max(hc_dim // 4, 1)
        if is_self:
            self.c_fc5 = nn.Linear(c_out_dim, max(hc_dim // 8, 1))
            c_out_dim = max(hc_dim // 8, 1)
        self.c_logit_head = nn.Linear(c_out_dim, 1)

        self.u_fc1 = nn.Linear(hu_dim, hu_dim)
        self.u_fc2 = nn.Linear(hu_dim, max(hu_dim // 2, 1))
        self.u_fc3 = nn.Linear(max(hu_dim // 2, 1), max(hu_dim // 4, 1))
        u_out_dim = max(hu_dim // 4, 1)
        if is_self:
            self.u_fc4 = nn.Linear(u_out_dim, max(hu_dim // 8, 1))
            u_out_dim = max(hu_dim // 8, 1)
        self.t_logit_head = nn.Linear(u_out_dim, 1)
        self.u_tau_head = nn.Linear(u_out_dim, 1)

        if act_type == "elu":
            self.act = nn.ELU()
        elif act_type == "relu":
            self.act = nn.ReLU()
        elif act_type == "tanh":
            self.act = nn.Tanh()
        elif act_type == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise ValueError("unknown act_type {}".format(act_type))

    def self_attn(self, q, k, v):
        Q, K, V = self.Q_w(q), self.K_w(k), self.V_w(v)
        attn_weights = Q.matmul(K.transpose(1, 2)) / (K.shape[-1] ** 0.5)
        attn_weights = self.softmax(torch.sigmoid(attn_weights))
        if self.attn_dropout is not None:
            attn_weights = self.attn_dropout(attn_weights)
        return attn_weights.matmul(V), attn_weights

    def interaction_attn(self, t, x):
        attn = []
        for i in range(self.nums_feature):
            temp = self.att_embed_3(torch.relu(
                torch.sigmoid(self.att_embed_1(t)) + torch.sigmoid(self.att_embed_2(x[:, i, :]))))
            attn.append(temp)
        attn = torch.squeeze(torch.stack(attn, 1), 2)
        attn = torch.softmax(attn, 1)
        out = torch.squeeze(torch.matmul(torch.unsqueeze(attn, 1), x), 1)
        return out, attn

    def forward(self, x_cont, x_cat=None):
        B = x_cont.shape[0]
        ones_t = torch.ones((B, 1), dtype=x_cont.dtype, device=x_cont.device)

        x_rep = x_cont.unsqueeze(2) * self.x_rep.weight.unsqueeze(0)  # [B, K, hu]

        x_rep_norm = x_rep / torch.linalg.norm(x_rep, dim=1, keepdim=True).clamp_min(1e-12)
        xx, _ = self.self_attn(x_rep_norm, x_rep_norm, x_rep_norm)
        flat = xx.reshape(B, -1)
        c_last = self.act(self.c_fc1(flat))
        c_last = self.act(self.c_fc2(c_last))
        c_last = self.act(self.c_fc3(c_last))
        c_last = self.act(self.c_fc4(c_last))
        if self.is_self:
            c_last = self.act(self.c_fc5(c_last))
        c_logit = self.c_logit_head(c_last).squeeze(-1)

        t_rep = self.t_rep(ones_t)
        xt, _ = self.interaction_attn(t_rep, x_rep)
        u_last = self.act(self.u_fc1(xt))
        u_last = self.act(self.u_fc2(u_last))
        u_last = self.act(self.u_fc3(u_last))
        if self.is_self:
            u_last = self.act(self.u_fc4(u_last))
        t_logit = self.t_logit_head(u_last).squeeze(-1)
        u_tau = self.u_tau_head(u_last).squeeze(-1)

        y0_logit = c_logit
        y1_logit = c_logit.detach() + u_tau

        return y0_logit, y1_logit, {"efin_t_logit": t_logit}


# ==========================================
# ECUP revision.
# ==========================================
class TAUBranch_0717new(nn.Module):
    """One treatment-aware unit: self-attention (equations 9-11) and the treatment-information extractor (equation 12). Equation 13."""
    def __init__(self, d_dim, num_heads=2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_dim, num_heads=num_heads, batch_first=True)
        self.tie_mlp = nn.Sequential(nn.Linear(d_dim, d_dim), nn.ReLU())

    def forward(self, E_x, E_tr):
        E_att, _ = self.self_attn(E_x, E_x, E_x)   # Eq9-11
        E_bit = self.tie_mlp(E_tr)                  # Eq12
        return E_att * E_bit                        # Eq13: E^TAU = E^att ⊗ E^bit


class TENet_0717new(nn.Module):
    """Treatment-enhanced network. Two independent units. Equation 14."""
    def __init__(self, f_num, d_dim, num_heads=2):
        super().__init__()
        self.f_num = f_num
        self.d_dim = d_dim
        self.tau_g = TAUBranch_0717new(d_dim, num_heads=num_heads)   # Treatment-aware content.
        self.tau_w = TAUBranch_0717new(d_dim, num_heads=num_heads)   # Bit-level gate.

    def forward(self, E_x, E_tr):
        E_TAU_g = self.tau_g(E_x, E_tr)
        W_b = self.tau_w(E_x, E_tr)
        gate = torch.sigmoid(W_b)
        E_r = E_x * gate + E_TAU_g * (1 - gate)      # Eq15 TEGate
        E_r_final = torch.cat([E_r, E_tr], dim=1)    # Eq16
        return E_r_final


class TAENet_0717new(nn.Module):
    """Task-enhanced network. Cross-attention stops the gradient into E_r_final. Equation 5."""
    def __init__(self, d_dim, tae_h, tower_h, num_tasks=2, num_heads=2, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        self.num_tasks = num_tasks
        self.E_ta = nn.Parameter(torch.randn(num_tasks, d_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_dim, num_heads=num_heads, batch_first=True)
        self.tae_mlp = nn.Sequential(
            nn.Linear(d_dim, tae_h),
            nn.ReLU(),
            nn.Linear(tae_h, tower_h),
        )

    def forward(self, E_r_final):
        B = E_r_final.size(0)
        query = self.E_ta.unsqueeze(0).expand(B, -1, -1)
        kv = E_r_final.detach()   # Equation 5. The gradient updates E_ta and does not enter the treatment-enhanced network.
        E_pri, _ = self.cross_attn(query, kv, kv)   # Eq5
        delta = self.gamma * torch.sigmoid(self.tae_mlp(E_pri))   # Eq6
        return delta


class ECUP_0717new(nn.Module):
    """ECUP revision. Field projections match the previous class. The three networks follow the paper."""
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 tower_h: int = 128, tae_h: int = 64, d_dim: int = 16,
                 num_heads: int = 2, gamma: float = 1.0):
        super().__init__()
        self.d_dim = d_dim
        self.cont_dim = continuous_dim
        self.cat_cards = categorical_cardinalities or {}
        self.f_num = self.cont_dim + len(self.cat_cards)

        self.cont_projections = nn.ModuleList([
            nn.Linear(1, d_dim) for _ in range(self.cont_dim)
        ])
        self.cat_embeddings = nn.ModuleDict({
            col: nn.Embedding(card, d_dim)
            for col, card in self.cat_cards.items()
        })

        self.t_emb_0 = nn.Parameter(torch.randn(1, 1, d_dim))
        self.t_emb_1 = nn.Parameter(torch.randn(1, 1, d_dim))

        self.tenet = TENet_0717new(f_num=self.f_num, d_dim=d_dim, num_heads=num_heads)
        self.taenet = TAENet_0717new(d_dim=d_dim, tae_h=tae_h, tower_h=tower_h,
                                      num_tasks=2, num_heads=num_heads, gamma=gamma)

        self.shared_layer1 = nn.Sequential(nn.Linear((self.f_num + 1) * d_dim, tower_h), nn.ReLU())
        self.shared_layer2 = nn.Sequential(nn.Linear(tower_h, tower_h), nn.ReLU())
        self.ctr_out = nn.Linear(tower_h, 1)
        self.cvr_out = nn.Linear(tower_h, 1)

    def _get_initial_embeddings(self, x_cont, x_cat):
        all_field_embeds = []
        if x_cont is not None:
            for i in range(self.cont_dim):
                feat = x_cont[:, i:i + 1]
                all_field_embeds.append(self.cont_projections[i](feat).unsqueeze(1))
        if x_cat is not None:
            for col, val in x_cat.items():
                all_field_embeds.append(self.cat_embeddings[col](val).unsqueeze(1))
        return torch.cat(all_field_embeds, dim=1)

    def _forward_once(self, x_cont, x_cat, T):
        E_x = self._get_initial_embeddings(x_cont, x_cat)
        B = E_x.size(0)

        E_tr = self.t_emb_1.expand(B, -1, -1) if T == 1 else self.t_emb_0.expand(B, -1, -1)

        E_r_final = self.tenet(E_x, E_tr)
        delta = self.taenet(E_r_final)   # [B, 2, tower_h]

        l1_out = self.shared_layer1(E_r_final.view(B, -1))   # [B, tower_h]

        ctr_h1 = l1_out * delta[:, 0, :]
        ctr_h2 = self.shared_layer2(ctr_h1) * delta[:, 0, :]
        ctr_logit = self.ctr_out(ctr_h2).squeeze(-1)

        cvr_h1 = l1_out * delta[:, 1, :]
        cvr_h2 = self.shared_layer2(cvr_h1) * delta[:, 1, :]
        cvr_logit = self.cvr_out(cvr_h2).squeeze(-1)

        return ctr_logit, cvr_logit

    def forward(self, x_cont, x_cat):
        c0_logit, cvr0_logit = self._forward_once(x_cont, x_cat, T=0)
        c1_logit, cvr1_logit = self._forward_once(x_cont, x_cat, T=1)
        return {
            "c_logits": (c0_logit, c1_logit),
            "cvr_logits": (cvr0_logit, cvr1_logit),
            "pi_dict": {},
            "ecup_0717new": True,
        }


# ==========================================
# MTMT revision. The interaction uses softmax, as in the paper.
# ==========================================
class UserTreatmentInteraction_0717new(nn.Module):
    def __init__(self, t_dim, u_dim, out_dim):
        super().__init__()
        self.W_t = nn.Linear(t_dim, out_dim)
        self.W_u = nn.Linear(u_dim, out_dim)
        self.W_v = nn.Linear(u_dim, out_dim)
        self.scale = out_dim ** 0.5

    def forward(self, epsilon, phi):
        Q = self.W_t(epsilon)  # [B, out_dim]
        K = self.W_u(phi)      # [B, out_dim]
        V = self.W_v(phi)      # [B, out_dim]
        attn_score = torch.softmax((Q * K) / self.scale, dim=-1)
        psi = attn_score * V  # [B, out_dim]
        return psi


class MTMT_0717new(nn.Module):
    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 num_experts: int = 4, expert_type: str = "mlp", expert_hidden_dims: list = [64],
                 dropout_rate: float = 0.1, t_emb_dim: int = 16):
        super().__init__()
        # FeatureEncoder and MMoE_Layer are defined above.
        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities)

        self.mmoe = MMoE_Layer(
            input_dim=self.encoder.output_dim,
            num_experts=num_experts,
            num_tasks=2,
            expert_type=expert_type,
            expert_hidden_dims=expert_hidden_dims,
            dropout_rate=dropout_rate
        )
        expert_out_dim = self.mmoe.expert_out_dim

        self.t_emb = nn.Parameter(torch.randn(1, t_emb_dim))

        self.y0_head_main = nn.Linear(expert_out_dim, 1)
        self.interaction_main = UserTreatmentInteraction_0717new(t_emb_dim, expert_out_dim, expert_out_dim)
        self.enhancer_main = nn.Sequential(
            nn.Linear(expert_out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.y0_head_aux = nn.Linear(expert_out_dim, 1)
        self.interaction_aux = UserTreatmentInteraction_0717new(t_emb_dim, expert_out_dim, expert_out_dim)
        self.enhancer_aux = nn.Sequential(
            nn.Linear(expert_out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_cont, x_cat):
        x = self.encoder(x_cont, x_cat)
        phi_main, phi_aux = self.mmoe(x)

        y0_main = self.y0_head_main(phi_main).squeeze(-1)
        y0_aux = self.y0_head_aux(phi_aux).squeeze(-1)

        eps = self.t_emb.expand(x.size(0), -1)

        psi_main = self.interaction_main(eps, phi_main)
        psi_aux = self.interaction_aux(eps, phi_aux)

        tau_main = self.enhancer_main(psi_main).squeeze(-1)
        tau_aux = self.enhancer_aux(psi_aux).squeeze(-1)

        y1_main = y0_main + tau_main
        y1_aux = y0_aux + tau_aux

        return {
            "main_task": (y0_main, y1_main),
            "aux_task": (y0_aux, y1_aux),
            "pi_dict": {}
        }

class _CanniFieldAttn(nn.Module):
    """Field-token stand-in for treatment attention.
    Figure 2 attends a coupon query to a behavior sequence. These tables have neither.
    Each field is projected to one token of width d_dim and stacked as E_x,
    with shape [B, f_num, d_dim]. A learned query attends over those tokens.
    The vector is concatenated into z for the seller and RDD heads.
    Keys and values are field tokens, not a behavior sequence.
    See the CanniUplift notes on the role of this block.
    """
    def __init__(self, continuous_dim, categorical_cardinalities, d_dim=16, num_heads=2):
        super().__init__()
        self.cont_dim = continuous_dim or 0
        self.cat_cards = categorical_cardinalities or {}
        self.output_dim = d_dim
        self.cont_projections = nn.ModuleList([nn.Linear(1, d_dim) for _ in range(self.cont_dim)])
        self.cat_embeddings = nn.ModuleDict(
            {col: nn.Embedding(card, d_dim) for col, card in self.cat_cards.items()}
        )
        self.query = nn.Parameter(torch.randn(1, 1, d_dim) * 0.02)
        self.mha = nn.MultiheadAttention(embed_dim=d_dim, num_heads=num_heads, batch_first=True)
    def _field_tokens(self, x_cont, x_cat):
        toks = []
        if x_cont is not None:
            for i in range(self.cont_dim):
                toks.append(self.cont_projections[i](x_cont[:, i:i + 1]).unsqueeze(1))
        if x_cat is not None:
            for col in self.cat_embeddings:
                toks.append(self.cat_embeddings[col](x_cat[col]).unsqueeze(1))
        if not toks:
            raise ValueError("_CanniFieldAttn: need x_cont and/or categorical features")
        return torch.cat(toks, dim=1)  # [B, f_num, d_dim]
    def forward(self, x_cont, x_cat):
        e_x = self._field_tokens(x_cont, x_cat)  # [B, f_num, d_dim]
        q = self.query.expand(e_x.size(0), -1, -1)  # [B, 1, d_dim]
        attn_out, _ = self.mha(q, e_x, e_x)
        return attn_out.squeeze(1)  # [B, d_dim]
def _canni_mlp_tower(in_dim, hidden_dims, activation=nn.ELU, head_bias=0.1):
    """Tower used by CanniUplift.
    Defined here so this file does not import the baseline extension."""
    layers = []
    d = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d, h))
        layers.append(activation())
        d = h
    head = nn.Linear(d, 1)
    nn.init.constant_(head.bias, head_bias)
    layers.append(head)
    return nn.Sequential(*layers)
def _canni_tower_logits(seq, x):
    """Run every layer except the last. The last layer returns a logit, with no activation."""
    h = x
    for layer in list(seq)[:-1]:
        h = layer(h)
    return seq[-1](h).squeeze(-1)
class CanniUplift(nn.Module):
    """CanniUplift. No official repository is shipped with the KDD 2026 ADS submission.
    Three blocks: treatment attention, seller towers, and the mediator RDD.
    The TensorFlow source is model_bias_uplift_euen_ord_seq_canniuplift.py
    and analysis_artifacts/baseline_port/CANNIUPLIFT_PYTORCH_IMPLEMENTATION_SPEC.md.
    forward(x_cont, x_cat) returns (y0_logit, y1_logit, pi_dict):
      - y0_logit and y1_logit are the seller-tower logits. Evaluation uses sigmoid(y1) - sigmoid(y0),
        the same return contract as the other models.
      - pi_dict also carries four RDD logits:
          canni_p_r_logit       P(mediator=1 | T=1, X), supervised on treated rows
          canni_mu_c_logit      RDD control head. This is not y0. See the implementation notes.
          canni_delta_r_logit   P(Y=1 | T=1, R=1, X)
          canni_delta_1mr_logit P(Y=1 | T=1, R=0, X)
    """
    def __init__(
        self,
        continuous_dim,
        categorical_cardinalities,
        hidden_dims=None,
        use_treat_attn=True,
        attn_d_dim=16,
        attn_num_heads=2,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        self.use_treat_attn = bool(use_treat_attn)
        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities)
        z_dim = self.encoder.output_dim
        if self.use_treat_attn:
            self.treat_attn = _CanniFieldAttn(
                continuous_dim, categorical_cardinalities, d_dim=attn_d_dim, num_heads=attn_num_heads,
            )
            z_dim += self.treat_attn.output_dim
        else:
            self.treat_attn = None
        # Seller towers. Uplift is y1 - y0, the same contract as the other models.
        self.head_y0 = _canni_mlp_tower(z_dim, hidden_dims)
        self.head_y1 = _canni_mlp_tower(z_dim, hidden_dims)
        # RDD heads are one linear layer each.
        self.head_p_r = nn.Linear(z_dim, 1)
        self.head_mu_c = nn.Linear(z_dim, 1)
        self.head_delta_r = nn.Linear(z_dim, 1)
        self.head_delta_1mr = nn.Linear(z_dim, 1)
    def forward(self, x_cont, x_cat):
        phi = self.encoder(x_cont, x_cat)
        if self.use_treat_attn:
            z = torch.cat([phi, self.treat_attn(x_cont, x_cat)], dim=-1)
        else:
            z = phi
        y0_logit = _canni_tower_logits(self.head_y0, z)
        y1_logit = _canni_tower_logits(self.head_y1, z)
        pi_dict = {
            "canni_p_r_logit": self.head_p_r(z).squeeze(-1),
            "canni_mu_c_logit": self.head_mu_c(z).squeeze(-1),
            "canni_delta_r_logit": self.head_delta_r(z).squeeze(-1),
            "canni_delta_1mr_logit": self.head_delta_1mr(z).squeeze(-1),
        }
        return y0_logit, y1_logit, pi_dict

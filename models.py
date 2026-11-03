"""Models used by the released Criteo PRISM run.

Stage I mediator: ``TARNET_Baseline``.
Stage II outcome: ``PRISM`` (imported from ``prism.py``).
"""

import torch
import torch.nn as nn

from prism import PRISM


class FeatureEncoder(nn.Module):
    """Continuous features, categorical features, or both."""

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
        if not feats:
            return torch.empty(0)
        return torch.cat(feats, dim=1)


class TARNET_Baseline(nn.Module):
    """Stage-I mediator TARNet (two treatment-specific heads).

    Also usable as a plain outcome TARNet when no mediator is attached.
    """

    def __init__(self, continuous_dim: int, categorical_cardinalities: dict,
                 hidden_dims: list, dropout_rate: float, embedding_dim: int = 8,
                 head_hidden_dims: list = None):
        super().__init__()
        self.encoder = FeatureEncoder(continuous_dim, categorical_cardinalities, embedding_dim)

        layers = []
        curr_dim = self.encoder.output_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim

        self.shared_layers = nn.Sequential(*layers)

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

    def forward(self, x_cont, x_cat):
        x = self.encoder(x_cont, x_cat)
        shared_emb = self.shared_layers(x)
        out_0 = self.head_0(shared_emb).squeeze(-1)
        out_1 = self.head_1(shared_emb).squeeze(-1)
        if not self.training:
            return out_0, out_1, {"wb_shared_emb": shared_emb}
        return out_0, out_1, shared_emb

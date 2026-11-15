"""TopoFlowNet: six-layer GIN + path collaboration + skip concat (P1 eqs. 5-12)."""

from __future__ import annotations

import torch
import torch.nn as nn


class GINLayer(nn.Module):
    """One TopoFlowNet layer: GINConv, path-collaboration MLP, skip to h^0.

    Layer l takes features of size l and returns size l+1, matching P1 Table in
    Section 5.3 and eqs. (6)-(12).
    """

    def __init__(self, layer_index: int, num_paths: int):
        super().__init__()
        dim = layer_index  # input dim of layer l is l
        self.layer_index = layer_index
        self.num_paths = num_paths
        self.eps = nn.Parameter(torch.zeros(1))
        self.gin_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        collab_dim = num_paths * dim
        self.path_mlp = nn.Sequential(
            nn.Linear(collab_dim, collab_dim),
            nn.ReLU(),
            nn.Linear(collab_dim, collab_dim),
        )

    def forward(
        self,
        h_prev: torch.Tensor,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        num_edges: int,
    ) -> torch.Tensor:
        src, dst = edge_index
        aggregated = torch.zeros_like(h_prev)
        aggregated.index_add_(0, dst, h_prev[src])
        gin_out = self.gin_mlp((1.0 + self.eps) * h_prev + aggregated)

        edge_feats = gin_out[:num_edges]
        path_feats = gin_out[num_edges:]
        num_demands = path_feats.size(0) // self.num_paths
        reshaped = path_feats.reshape(num_demands, self.num_paths * h_prev.size(-1))
        path_updated = self.path_mlp(reshaped).reshape_as(path_feats)

        concat_nodes = torch.cat([edge_feats, path_updated], dim=0)
        return torch.cat([concat_nodes, h0], dim=-1)


class TopoFlowNet(nn.Module):
    """P1 Section 4.2: L=6, final path embedding dim = 7."""

    def __init__(self, num_layers: int = 6, num_paths: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.num_paths = num_paths
        self.layers = nn.ModuleList(
            [GINLayer(layer_index=i + 1, num_paths=num_paths) for i in range(num_layers)]
        )

    def forward(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        num_edges: int,
    ) -> torch.Tensor:
        h = h0
        for layer in self.layers:
            h = layer(h, h0, edge_index, num_edges)
        return h[num_edges:]  # path-node embeddings, dim = L+1

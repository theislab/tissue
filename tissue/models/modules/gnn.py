from typing import List

import torch.nn as nn
import torch_geometric.nn as geom_nn

from .interactive_gat_layer import GATInteractionConv


class ModuleGCN(nn.Module):
    def __init__(self, in_channels, units: List[int], dropout_rate=0.0, **kwargs):
        super().__init__()

        layers = []
        prev_dim = in_channels
        for dim in units:
            layers += [
                geom_nn.GCNConv(in_channels=prev_dim, out_channels=dim),
                # nn.ReLU(inplace=True),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                # nn.LogSoftmax(dim=1),
                nn.Dropout(dropout_rate),
            ]
            prev_dim = dim

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x


class ModuleGIN(nn.Module):
    def __init__(self, in_channels, units: List[int], **kwargs):
        super().__init__()

        layers = []
        prev_dim = in_channels
        for dim in units:
            layers += [
                geom_nn.GINConv(
                    nn.Sequential(
                        nn.Linear(prev_dim, dim), 
                        nn.BatchNorm1d(dim), 
                        nn.ReLU(),
                        nn.Linear(dim, dim), 
                        nn.ReLU()
                        )
                    )
            ]
            prev_dim = dim

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        h = []
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
                h.append(x)
            else:
                x = l(x)
                h.append(x)
        return h        


class ModuleGAT(nn.Module):
    def __init__(self, in_channels, units: List[int], heads: int = 1, **kwargs):
        super().__init__()

        layers = []
        prev_dim = in_channels
        for dim in units:
            layers += [
                geom_nn.GATConv(in_channels=prev_dim, out_channels=dim, heads=heads),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
            ]
            prev_dim = dim

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)#, return_attention_weights=True)
            else:
                x = l(x)
        return x


class ModuleGATInteraction(nn.Module):
    def __init__(self, in_channels, units: List[int], heads: int = 1, att_dim: int = 4, **kwargs):
        super().__init__()

        layers = []
        prev_dim = in_channels
        for dim in units:
            layers += [
                GATInteractionConv(att_dim=att_dim, in_channels=prev_dim, out_channels=dim, heads=heads),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
            ]
            prev_dim = dim

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)#, return_attention_weights=True)
            else:
                x = l(x)
        return x

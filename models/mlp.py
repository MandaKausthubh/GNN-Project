"""
MLP model wrapper.
Purely feature-based baseline with no graph structure usage.
"""

from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch.nn import Identity, Linear, Module

from .base import BaseModelWrapper


class MLPWrapper(BaseModelWrapper):
    """
    Multi-Layer Perceptron (MLP) wrapper.

    This model uses only node features and ignores graph edges.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of MLP layers (including output layer).
        out_channels (int, optional): Output size. If None, uses hidden size.
        dropout (float, optional): Dropout probability. (default: 0.)
        act (str or Callable, optional): Activation function. (default: "relu")
        norm (str or Callable, optional): Normalization function. (default: None)
    """

    _pyg_model_cls = None

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        norm: Union[str, Callable, None] = None,
    ):
        self._mlp_num_layers = max(1, int(num_layers))
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=self._mlp_num_layers,
            out_channels=out_channels,
            dropout=dropout,
            act=act,
            act_first=False,
            norm=norm,
            jk=None,
        )

    def _init_pyg_model(self, **kwargs):
        layers = []
        in_dim = self.in_channels
        hidden_dim = self.hidden_channels
        out_dim = self.out_channels

        for layer_idx in range(self._mlp_num_layers - 1):
            layers.append(Linear(in_dim, hidden_dim))
            layers.append(self._get_norm(hidden_dim))
            layers.append(self._get_activation())
            layers.append(torch.nn.Dropout(p=self.dropout_p))
            in_dim = hidden_dim

        layers.append(Linear(in_dim, out_dim))
        return torch.nn.Sequential(*layers)

    def _get_activation(self) -> Module:
        if callable(self.act):
            return self.act()
        if isinstance(self.act, str):
            act_name = self.act.lower()
            if act_name == "relu":
                return torch.nn.ReLU()
            if act_name == "gelu":
                return torch.nn.GELU()
            if act_name == "elu":
                return torch.nn.ELU()
            if act_name == "leaky_relu":
                return torch.nn.LeakyReLU()
            if act_name == "tanh":
                return torch.nn.Tanh()
            if act_name == "sigmoid":
                return torch.nn.Sigmoid()
        return Identity()

    def _get_norm(self, dim: int) -> Module:
        if callable(self.norm):
            return self.norm(dim)
        if isinstance(self.norm, str):
            norm_name = self.norm.lower()
            if norm_name == "layer":
                return torch.nn.LayerNorm(dim)
            if norm_name == "batch":
                return torch.nn.BatchNorm1d(dim)
        return Identity()

    def forward(
        self,
        x: Tensor,
        edge_index=None,
        edge_attr: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        return self._model(x)

    def get_embeddings(self, data) -> Tensor:
        return self.forward(data.x)

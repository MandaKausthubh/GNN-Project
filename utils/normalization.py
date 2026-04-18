"""
Normalization utilities for GNNs.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor


def apply_layer_norm(
    x: Tensor,
    normalized_shape: Union[int, tuple],
    elementwise_affine: bool = True,
) -> Tensor:
    """
    Apply layer normalization to input tensor.

    Args:
        x: Input tensor.
        normalized_shape: Shape for normalization.
        elementwise_affine: Whether to use learnable affine transform.

    Returns:
        Normalized tensor.
    """
    layer_norm = nn.LayerNorm(normalized_shape, elementwise_affine=elementwise_affine)
    return layer_norm(x)


def apply_batch_norm(x: Tensor) -> Tensor:
    """
    Apply batch normalization to input tensor.

    Args:
        x: Input tensor (expecting shape [batch, features]).

    Returns:
        Normalized tensor.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    batch_norm = nn.BatchNorm1d(x.size(-1))
    return batch_norm(x)


class GraphNorm(nn.Module):
    """
    Graph Normalization layer.

    Normalizes per-graph rather than per-batch, useful for graph classification.
    Reference: https://arxiv.org/abs/2003.00982
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.mean_scale = nn.Parameter(torch.ones(num_features))

    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: Node features [num_nodes, num_features].
            batch: Batch assignment vector [num_nodes].

        Returns:
            Normalized features.
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Compute mean per graph
        batch_size = batch.max().item() + 1
        mean = torch.zeros(batch_size, x.size(1), device=x.device)
        count = torch.zeros(batch_size, device=x.device)

        mean = mean.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
        count = count.scatter_add_(0, batch.unsqueeze(-1), torch.ones_like(x))

        mean = mean / count.clamp(min=1).unsqueeze(-1)

        # Normalize
        x = x - mean[batch]
        var = torch.zeros(batch_size, x.size(1), device=x.device)
        var = var.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x ** 2)
        var = var / count.clamp(min=1).unsqueeze(-1)

        x = x * self.mean_scale[None, :]
        x = x / (var[batch] + self.eps).sqrt()
        x = x * self.weight[None, :] + self.bias[None, :]

        return x


class PairNorm(nn.Module):
    """
    Pair Normalization layer.

    Normalizes node features based on their pairwise distances.
    Reference: https://arxiv.org/abs/1910.10811
    """

    def __init__(self, scale: float = 1.0, shift: float = 1.0):
        super().__init__()
        self.scale = scale
        self.shift = shift

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Node features [num_nodes, num_features].

        Returns:
            Normalized features.
        """
        # Compute pairwise Euclidean distances
        d = torch.cdist(x, x, p=2)

        # Mean of pairwise distances
        d_mean = d.sum() / (d.size(0) ** 2 - d.size(0))

        # Normalize
        x = x / (d_mean + self.shift) * self.scale

        return x


class DropoutRegularization(nn.Module):
    """
    Combined dropout and activation module for flexible regularization.
    """

    def __init__(
        self,
        p: float = 0.5,
        activation: Optional[str] = "relu",
        inplace: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=p)
        self.activation = None
        if activation:
            self.activation = getattr(nn.functional, activation, None)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        if self.activation:
            x = self.activation(x, inplace=getattr(self, "inplace", False))
        return x


__all__ = [
    "apply_layer_norm",
    "apply_batch_norm",
    "GraphNorm",
    "PairNorm",
    "DropoutRegularization",
]
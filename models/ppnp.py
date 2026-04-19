"""
PPNP/APPNP model wrapper.
Wraps PyTorch Geometric's APPNP propagation with an MLP predictor.

The PPNP (Predict then Propagate) model from the paper:
"Predict then Propagate: Graph Neural Networks meet Personalized PageRank"
https://arxiv.org/abs/1810.05997
"""

from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import APPNP as APPNPConv
from torch_geometric.typing import SparseTensor

from .base import BaseModelWrapper


class APPNPWrapper(BaseModelWrapper):
    """
    Approximate Personalized Propagation of Neural Predictions (APPNP) wrapper.

    Wraps the APPNP propagation layer with an MLP for feature transformation.
    This is the approximate version of PPNP that uses power iteration for
    efficient O(m) complexity instead of O(n²).

    From the paper: "Predict then Propagate: Graph Neural Networks meet
    Personalized PageRank" (ICLR 2019)

    The model first predicts node features with an MLP, then propagates
    using Personalized PageRank:

        X^(0) = MLP(X)
        X^(k+1) = (1 - α) * Ã @ X^(k) + α * X^(0)

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of MLP layers for prediction.
        out_channels (int, optional): Size of output. If None, uses hidden_channels.
        dropout (float, optional): Dropout probability. (default: 0.0)
        K (int, optional): Number of propagation iterations. (default: 10)
        alpha (float, optional): Teleport probability (restart probability).
            Controls how much to retain original predictions vs. propagate.
            (default: 0.1)
        act (str or Callable, optional): Activation function. (default: "relu")
        act_first (bool, optional): Apply activation before normalization. (default: False)
        norm (str or Callable, optional): Normalization function. (default: None)
        cached (bool, optional): Cache normalized adjacency matrix. (default: False)
        add_self_loops (bool, optional): Add self-loops to graph. (default: True)
        normalize (bool, optional): Apply symmetric normalization. (default: True)
    """

    _pyg_model_cls = None  # We build the model manually

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        K: int = 10,
        alpha: float = 0.1,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        norm: Union[str, Callable, None] = None,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
    ):
        self.K = K
        self.alpha = alpha
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            dropout=dropout,
            act=act,
            act_first=act_first,
            norm=norm,
            jk=None,
        )

    def _init_pyg_model(self, **kwargs):
        """Build the MLP predictor and APPNP propagation layer."""
        # Build MLP layers for feature transformation
        mlp_layers = []

        # Input layer
        mlp_layers.append(nn.Linear(self.in_channels, self.hidden_channels))
        if self.act:
            mlp_layers.append(self._get_activation())
        if self.dropout.p > 0:
            mlp_layers.append(nn.Dropout(self.dropout.p))

        # Hidden layers
        for _ in range(self.num_layers - 2):
            mlp_layers.append(nn.Linear(self.hidden_channels, self.hidden_channels))
            if self.act:
                mlp_layers.append(self._get_activation())
            if self.dropout.p > 0:
                mlp_layers.append(nn.Dropout(self.dropout.p))

        # Output layer
        mlp_layers.append(nn.Linear(self.hidden_channels, self.out_channels))

        self.mlp = nn.Sequential(*mlp_layers)

        # APPNP propagation layer
        self.prop = APPNPConv(
            K=self.K,
            alpha=self.alpha,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
            normalize=self.normalize,
        )

        return None  # We don't use a single PyG model

    def _get_activation(self):
        """Get activation function as a module for nn.Sequential."""
        if isinstance(self.act, str):
            act_name = self.act.lower()
            if act_name == "relu":
                return nn.ReLU()
            elif act_name == "leaky_relu":
                return nn.LeakyReLU()
            elif act_name == "elu":
                return nn.ELU()
            elif act_name == "gelu":
                return nn.GELU()
            elif act_name == "tanh":
                return nn.Tanh()
            elif act_name == "sigmoid":
                return nn.Sigmoid()
            else:
                return nn.ReLU()
        elif callable(self.act):
            # If it's already a module instance, return it
            if isinstance(self.act, nn.Module):
                return self.act
            # If it's a function, wrap it (fallback)
            return nn.ReLU()
        return nn.ReLU()

    def forward(
        self,
        x: Tensor,
        edge_index: Union[Tensor, SparseTensor],
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass: predict with MLP, then propagate with APPNP.

        Args:
            x: Node feature tensor.
            edge_index: Edge indices.
            edge_weight: Optional edge weights.

        Returns:
            Node embeddings.
        """
        # Predict: MLP transformation
        h = self.mlp(x)

        # Propagate: Personalized PageRank
        out = self.prop(h, edge_index, edge_weight=edge_weight)

        return out

    def get_embeddings(self, data) -> Tensor:
        """
        Get node embeddings for a Data object.

        Args:
            data: PyG Data object with x and edge_index.

        Returns:
            Node embeddings tensor.
        """
        return self.forward(data.x, data.edge_index)

    def reset_parameters(self):
        """Reset all learnable parameters."""
        for layer in self.mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        if hasattr(self.prop, 'reset_parameters'):
            self.prop.reset_parameters()

    def parameters(self, recurse: bool = True):
        """Return an iterator over model parameters."""
        return self.mlp.parameters(recurse)

    def train(self, mode: bool = True):
        """Set the model to training mode."""
        super().train(mode)
        return self

    def eval(self):
        """Set the model to evaluation mode."""
        super().eval()
        return self

    def to(self, *args, **kwargs):
        """Move the model to the specified device."""
        self.mlp = self.mlp.to(*args, **kwargs)
        return self


class PPNPWrapper(BaseModelWrapper):
    """
    Exact Personalized Propagation of Neural Predictions (PPNP) wrapper.

    This implements the exact PPNP from the paper using a precomputed
    Personalized PageRank matrix. Unlike APPNP which uses power iteration,
    this computes the full propagation matrix:

        H = PPR @ MLP(X)

    where PPR = α * (I - (1-α) * Â)^(-1)

    Note: This requires O(n²) memory for storing the PPR matrix, making it
    suitable only for smaller graphs (typically < 10,000 nodes). For larger
    graphs, use APPNPWrapper instead.

    From the paper: "Predict then Propagate: Graph Neural Networks meet
    Personalized PageRank" (ICLR 2019)

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of MLP layers for prediction.
        out_channels (int, optional): Size of output. If None, uses hidden_channels.
        dropout (float, optional): Dropout probability. (default: 0.0)
        alpha (float, optional): Teleport probability (restart probability).
            (default: 0.1)
        act (str or Callable, optional): Activation function. (default: "relu")
        act_first (bool, optional): Apply activation before normalization. (default: False)
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
        alpha: float = 0.1,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        norm: Union[str, Callable, None] = None,
    ):
        self.alpha = alpha
        self.ppr_matrix = None  # Precomputed PPR matrix

        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            dropout=dropout,
            act=act,
            act_first=act_first,
            norm=norm,
            jk=None,
        )

    def _init_pyg_model(self, **kwargs):
        """Build the MLP predictor."""
        mlp_layers = []

        # Input layer
        mlp_layers.append(nn.Linear(self.in_channels, self.hidden_channels))
        if self.act:
            mlp_layers.append(self._get_activation())
        if self.dropout.p > 0:
            mlp_layers.append(nn.Dropout(self.dropout.p))

        # Hidden layers
        for _ in range(self.num_layers - 2):
            mlp_layers.append(nn.Linear(self.hidden_channels, self.hidden_channels))
            if self.act:
                mlp_layers.append(self._get_activation())
            if self.dropout.p > 0:
                mlp_layers.append(nn.Dropout(self.dropout.p))

        # Output layer
        mlp_layers.append(nn.Linear(self.hidden_channels, self.out_channels))

        self.mlp = nn.Sequential(*mlp_layers)

        return None

    def _get_activation(self):
        """Get activation function as a module for nn.Sequential."""
        if isinstance(self.act, str):
            act_name = self.act.lower()
            if act_name == "relu":
                return nn.ReLU()
            elif act_name == "leaky_relu":
                return nn.LeakyReLU()
            elif act_name == "elu":
                return nn.ELU()
            elif act_name == "gelu":
                return nn.GELU()
            elif act_name == "tanh":
                return nn.Tanh()
            elif act_name == "sigmoid":
                return nn.Sigmoid()
            else:
                return nn.ReLU()
        elif callable(self.act):
            if isinstance(self.act, nn.Module):
                return self.act
            return nn.ReLU()
        return nn.ReLU()

    def compute_ppr_matrix(
        self,
        edge_index: Tensor,
        num_nodes: int,
        edge_weight: Optional[Tensor] = None,
        add_self_loops: bool = True,
        normalize: bool = True,
    ) -> Tensor:
        """
        Compute the Personalized PageRank matrix.

        PPR = α * (I - (1-α) * Â)^(-1)

        Args:
            edge_index: Edge indices.
            num_nodes: Number of nodes.
            edge_weight: Optional edge weights.
            add_self_loops: Whether to add self-loops.
            normalize: Whether to apply symmetric normalization.

        Returns:
            The PPR matrix as a sparse tensor.
        """
        from torch_geometric.utils import add_self_loops, degree
        from torch_geometric.utils.num_nodes import maybe_num_nodes

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        # Add self-loops if requested
        if add_self_loops:
            edge_index, edge_weight = self._add_self_loops(
                edge_index, edge_weight, num_nodes
            )

        # Compute normalized adjacency
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        row, col = edge_index
        deg = degree(col, num_nodes, dtype=edge_weight.dtype)

        if normalize:
            # Symmetric normalization: A_norm = D^(-1/2) @ A @ D^(-1/2)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # Build sparse normalized adjacency
        adj = torch.sparse_coo_tensor(
            edge_index, edge_weight, size=(num_nodes, num_nodes)
        ).coalesce()

        # Compute PPR using power iteration (exact for sparse matrices)
        # PPR = α * Σ_{k=0}^∞ (1-α)^k * Â^k
        # This is more efficient than matrix inversion for sparse graphs

        # Initialize with identity
        ppr = torch.eye(num_nodes, device=edge_index.device) * self.alpha

        # Power series expansion
        accum = torch.eye(num_nodes, device=edge_index.device)
        coeff = (1 - self.alpha)

        # Use power iteration to approximate PPR
        for _ in range(50):  # Sufficient iterations for convergence
            accum = torch.sparse.mm(adj, accum) if accum.is_sparse else accum @ adj.to_dense()
            ppr = ppr + self.alpha * coeff * accum
            coeff = coeff * (1 - self.alpha)

        return ppr

    def _add_self_loops(
        self,
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
        num_nodes: int,
    ) -> tuple:
        """Add self-loops to the graph."""
        from torch_geometric.utils import add_self_loops

        if edge_weight is None:
            edge_index, edge_weight = add_self_loops(
                edge_index, num_nodes=num_nodes
            )
            edge_weight = torch.cat([
                torch.ones(edge_index.size(1) - num_nodes, device=edge_index.device),
                torch.ones(num_nodes, device=edge_index.device)
            ])
        else:
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_weight, num_nodes=num_nodes
            )

        return edge_index, edge_weight

    def precompute_ppr(
        self,
        edge_index: Tensor,
        num_nodes: int,
        edge_weight: Optional[Tensor] = None,
    ):
        """
        Precompute and cache the PPR matrix.

        Call this once before training/inference for efficiency.

        Args:
            edge_index: Edge indices.
            num_nodes: Number of nodes.
            edge_weight: Optional edge weights.
        """
        self.ppr_matrix = self.compute_ppr_matrix(
            edge_index, num_nodes, edge_weight
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        precomputed_ppr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass: predict with MLP, then propagate with precomputed PPR.

        Args:
            x: Node feature tensor.
            edge_index: Edge indices.
            edge_weight: Optional edge weights.
            precomputed_ppr: Precomputed PPR matrix (optional).

        Returns:
            Node embeddings.
        """
        # Predict: MLP transformation
        h = self.mlp(x)

        # Get PPR matrix
        ppr = precomputed_ppr if precomputed_ppr is not None else self.ppr_matrix

        if ppr is None:
            # Compute PPR on-the-fly (slower)
            num_nodes = x.size(0)
            ppr = self.compute_ppr_matrix(edge_index, num_nodes, edge_weight)

        # Propagate: H = PPR @ h
        if ppr.is_sparse:
            out = torch.sparse.mm(ppr, h)
        else:
            out = ppr @ h

        return out

    def get_embeddings(self, data) -> Tensor:
        """
        Get node embeddings for a Data object.

        Args:
            data: PyG Data object with x and edge_index.

        Returns:
            Node embeddings tensor.
        """
        return self.forward(data.x, data.edge_index)

    def reset_parameters(self):
        """Reset all learnable parameters."""
        for layer in self.mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.ppr_matrix = None

    def parameters(self, recurse: bool = True):
        """Return an iterator over model parameters."""
        return self.mlp.parameters(recurse)

    def train(self, mode: bool = True):
        """Set the model to training mode."""
        super().train(mode)
        return self

    def eval(self):
        """Set the model to evaluation mode."""
        super().eval()
        return self

    def to(self, *args, **kwargs):
        """Move the model to the specified device."""
        self.mlp = self.mlp.to(*args, **kwargs)
        if self.ppr_matrix is not None:
            self.ppr_matrix = self.ppr_matrix.to(*args, **kwargs)
        return self
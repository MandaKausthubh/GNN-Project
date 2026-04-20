"""
Residual GNN wrapper with normalization, dropout, and residual connections.
"""

import copy
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import SparseTensor


class ResidualGNNLayer(nn.Module):
    """
    A single GNN layer with residual connections, dropout, and normalization.

    Architecture: Input -> [Conv/MessagePassing] -> [Norm] -> [Dropout] -> [Residual Add] -> [Activation]

    Args:
        conv: The message passing layer (e.g., GCNConv, GATConv).
        norm: Normalization layer (e.g., LayerNorm, BatchNorm, GraphNorm).
        dropout: Dropout probability.
        activation: Activation function.
        use_residual: Whether to use residual connections.
        residual_alpha: Alpha for weighted residual connection.
    """

    def __init__(
        self,
        conv: nn.Module,
        norm: Optional[nn.Module] = None,
        dropout: float = 0.0,
        activation: Optional[str] = "relu",
        use_residual: bool = True,
        residual_alpha: float = 1.0,
    ):
        super().__init__()
        self.conv = conv
        self.norm = norm
        self.dropout = nn.Dropout(p=dropout)
        self.activation = getattr(nn.functional, activation, None) if activation else None
        self.use_residual = use_residual
        self.residual_alpha = residual_alpha

    def forward(
        self,
        x: Tensor,
        edge_index: Union[Tensor, SparseTensor],
        edge_weight: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        identity: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with optional residual connection.

        Args:
            x: Node features.
            edge_index: Edge indices.
            edge_weight: Optional edge weights.
            edge_attr: Optional edge features.
            identity: Skip connection input (if None, uses x).

        Returns:
            Updated node features.
        """
        identity = identity if identity is not None else x

        # Message passing
        # Different conv layers accept different parameters:
        # - GCNConv: edge_weight
        # - GATConv: edge_attr (not edge_weight)
        # - SAGEConv/GINConv: no edge weighting
        conv_class_name = self.conv.__class__.__name__
        if conv_class_name == 'GCNConv':
            h = self.conv(x, edge_index, edge_weight=edge_weight)
        elif conv_class_name == 'GATConv':
            if edge_attr is not None:
                h = self.conv(x, edge_index, edge_attr=edge_attr)
            else:
                h = self.conv(x, edge_index)
        else:
            # SAGEConv, GINConv, and others don't accept edge_weight
            h = self.conv(x, edge_index)

        # Normalization
        if self.norm is not None:
            h = self.norm(h)

        # Dropout
        h = self.dropout(h)

        # Residual connection with alpha blending
        if self.use_residual:
            h = self.residual_alpha * h + (1 - self.residual_alpha) * identity

        # Activation
        if self.activation:
            h = self.activation(h)

        return h


class ResidualGNNWrapper(nn.Module):
    """
    GNN model with residual connections, dropout, and normalization.

    Wraps any base GNN model and adds:
    - Residual connections between layers
    - Normalization after each conv layer
    - Dropout for regularization
    - Optional identity mappings

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Hidden layer dimension.
        num_layers: Number of GNN layers.
        out_channels: Output dimension (for classification).
        model_type: Type of GNN ("gcn", "gat", "sage", "gin").
        dropout: Dropout probability.
        norm: Normalization type ("layer", "batch", "graph", None).
        activation: Activation function name.
        use_residual: Whether to use residual connections.
        residual_alpha: Weight for residual blending.
        **kwargs: Additional arguments for the base GNN.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        model_type: str = "gcn",
        dropout: float = 0.5,
        norm: str = "layer",
        activation: str = "relu",
        use_residual: bool = True,
        residual_alpha: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.dropout = dropout
        self.norm_type = norm
        self.use_residual = use_residual
        self.residual_alpha = residual_alpha

        # Build base convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Input embedding
        self.input_proj = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else nn.Identity()

        for i in range(num_layers):
            in_ch = hidden_channels
            out_ch = hidden_channels if i < num_layers - 1 else out_channels

            # Create base conv layer
            conv = self._create_conv(model_type, in_ch, out_ch, **kwargs)
            self.convs.append(conv)

            # Create normalization
            if i < num_layers - 1:  # No norm on last layer
                norm_layer = self._create_norm(norm, out_ch)
                self.norms.append(norm_layer)
            else:
                self.norms.append(nn.Identity())

        # Store embeddings for inference
        self.emb = None

    def _create_conv(self, model_type: str, in_channels: int, out_channels: int, **kwargs):
        """Create the base convolution layer."""
        if model_type.lower() == "gcn":
            from torch_geometric.nn import GCNConv
            return GCNConv(in_channels, out_channels, **kwargs)
        elif model_type.lower() == "gat":
            from torch_geometric.nn import GATConv
            heads = kwargs.pop("heads", 1)
            concat = kwargs.pop("concat", True)
            return GATConv(in_channels, out_channels, heads=heads, concat=concat, **kwargs)
        elif model_type.lower() == "sage":
            from torch_geometric.nn import SAGEConv
            return SAGEConv(in_channels, out_channels, **kwargs)
        elif model_type.lower() == "gin":
            from torch_geometric.nn import GINConv
            from torch_geometric.nn.models import MLP
            mlp = MLP([in_channels, out_channels], norm="batch_norm")
            return GINConv(mlp, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _create_norm(self, norm_type: str, num_features: int):
        """Create the normalization layer."""
        if norm_type.lower() == "layer":
            return nn.LayerNorm(num_features)
        elif norm_type.lower() == "batch":
            return nn.BatchNorm1d(num_features)
        elif norm_type.lower() == "graph":
            from .normalization import GraphNorm
            return GraphNorm(num_features)
        elif norm_type is None or norm_type.lower() == "none":
            return nn.Identity()
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")

    def forward(
        self,
        x: Tensor,
        edge_index: Union[Tensor, SparseTensor],
        edge_weight: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        return_emb: bool = False,
    ) -> Tensor:
        """
        Forward pass with residual connections.

        Args:
            x: Node features [num_nodes, in_channels].
            edge_index: Edge indices [2, num_edges].
            edge_weight: Optional edge weights.
            edge_attr: Optional edge features.
            return_emb: Whether to return intermediate embeddings.

        Returns:
            Node embeddings or (embeddings, all_layer_outputs) if return_emb=True.
        """
        # Input projection
        x = self.input_proj(x)

        # Store all layer outputs for Jumping Knowledge if needed
        all_outputs = [x]

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            identity = x  # Residual connection

            # Convolution
            # Different conv layers accept different parameters:
            # - GCNConv: edge_weight
            # - GATConv: edge_attr (not edge_weight)
            # - SAGEConv/GINConv: no edge weighting
            conv_class_name = conv.__class__.__name__
            if conv_class_name == 'GCNConv':
                h = conv(x, edge_index, edge_weight=edge_weight)
            elif conv_class_name == 'GATConv':
                if edge_attr is not None:
                    h = conv(x, edge_index, edge_attr=edge_attr)
                else:
                    h = conv(x, edge_index)
            else:
                # SAGEConv, GINConv, and others don't accept edge_weight
                h = conv(x, edge_index)

            # Normalization (skip last layer)
            if i < self.num_layers - 1:
                h = norm(h)

            # Dropout during training
            if self.training:
                h = F.dropout(h, p=self.dropout, training=True)

            # Residual connection with alpha blending (only when dimensions match, i.e., not last layer)
            if self.use_residual and i < self.num_layers - 1:
                h = self.residual_alpha * h + (1 - self.residual_alpha) * identity

            # Activation (skip last layer)
            if i < self.num_layers - 1 and hasattr(nn.functional, "relu"):
                h = F.relu(h)

            x = h
            all_outputs.append(x)

        # Store embeddings
        self.emb = x

        if return_emb:
            return x, all_outputs
        return x

    def __repr__(self) -> str:
        return (
            f"ResidualGNNWrapper(\n"
            f"  in_channels={self.in_channels},\n"
            f"  hidden_channels={self.hidden_channels},\n"
            f"  num_layers={self.num_layers},\n"
            f"  out_channels={self.out_channels},\n"
            f"  dropout={self.dropout},\n"
            f"  norm={self.norm_type},\n"
            f"  use_residual={self.use_residual}\n"
            f")"
        )


class ResidualAPPNPWrapper(nn.Module):
    """
    APPNP/PPNP model with residual connections in the MLP predictor.

    This implements the "Predict then Propagate" architecture from:
    "Predict then Propagate: Graph Neural Networks meet Personalized PageRank"
    (ICLR 2019)

    Architecture:
    1. MLP with residual connections for feature transformation
    2. APPNP propagation for personalized PageRank

    The residual connections are applied in the MLP part, not in propagation.
    This avoids oversmoothing while capturing larger neighborhoods.

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Hidden layer dimension.
        num_layers: Number of MLP layers.
        out_channels: Output dimension (for classification).
        K: Number of propagation iterations (default: 10).
        alpha: Teleport probability for PageRank (default: 0.1).
        dropout: Dropout probability.
        norm: Normalization type ("layer", "batch", "graph", None).
        activation: Activation function name.
        use_residual: Whether to use residual connections in MLP.
        residual_alpha: Weight for residual blending.
        cached: Cache normalized adjacency matrix.
        add_self_loops: Add self-loops to graph.
        normalize: Apply symmetric normalization.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        K: int = 10,
        alpha: float = 0.1,
        dropout: float = 0.5,
        norm: str = "layer",
        activation: str = "relu",
        use_residual: bool = True,
        residual_alpha: float = 1.0,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.K = K
        self.alpha = alpha
        self.dropout = dropout
        self.norm_type = norm
        self.use_residual = use_residual
        self.residual_alpha = residual_alpha

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else nn.Identity()

        # MLP layers with residual connections
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers - 1):  # MLP layers (excluding output)
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
            norm_layer = self._create_norm(norm, hidden_channels)
            self.norms.append(norm_layer)

        # Output layer
        self.output_layer = nn.Linear(hidden_channels, out_channels)

        # APPNP propagation
        from torch_geometric.nn import APPNP
        self.prop = APPNP(
            K=K,
            alpha=alpha,
            cached=cached,
            add_self_loops=add_self_loops,
            normalize=normalize,
        )

        self.emb = None

    def _create_norm(self, norm_type: str, num_features: int):
        """Create the normalization layer."""
        if norm_type.lower() == "layer":
            return nn.LayerNorm(num_features)
        elif norm_type.lower() == "batch":
            return nn.BatchNorm1d(num_features)
        elif norm_type.lower() == "graph":
            from .normalization import GraphNorm
            return GraphNorm(num_features)
        elif norm_type is None or norm_type.lower() == "none":
            return nn.Identity()
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")

    def forward(
        self,
        x: Tensor,
        edge_index: Union[Tensor, SparseTensor],
        edge_weight: Optional[Tensor] = None,
        return_emb: bool = False,
    ) -> Tensor:
        """
        Forward pass: MLP prediction followed by APPNP propagation.

        Args:
            x: Node features [num_nodes, in_channels].
            edge_index: Edge indices [2, num_edges].
            edge_weight: Optional edge weights.
            return_emb: Whether to return intermediate embeddings.

        Returns:
            Node embeddings or (embeddings, all_layer_outputs) if return_emb=True.
        """
        # Input projection
        h = self.input_proj(x)
        all_outputs = [h]

        # MLP with residual connections
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            identity = h

            # Linear transformation
            h = layer(h)

            # Normalization
            h = norm(h)

            # Dropout
            if self.training:
                h = F.dropout(h, p=self.dropout, training=True)

            # Residual connection
            if self.use_residual:
                h = self.residual_alpha * h + (1 - self.residual_alpha) * identity

            # Activation
            h = F.relu(h)

            all_outputs.append(h)

        # Output layer (no activation for classification)
        h = self.output_layer(h)

        # APPNP propagation
        out = self.prop(h, edge_index, edge_weight=edge_weight)

        self.emb = out

        if return_emb:
            return out, all_outputs
        return out

    def __repr__(self) -> str:
        return (
            f"ResidualAPPNPWrapper(\n"
            f"  in_channels={self.in_channels},\n"
            f"  hidden_channels={self.hidden_channels},\n"
            f"  num_layers={self.num_layers},\n"
            f"  out_channels={self.out_channels},\n"
            f"  K={self.K},\n"
            f"  alpha={self.alpha},\n"
            f"  dropout={self.dropout},\n"
            f"  norm={self.norm_type},\n"
            f"  use_residual={self.use_residual}\n"
            f")"
        )
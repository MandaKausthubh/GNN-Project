"""
GAT model wrapper.
Wraps PyTorch Geometric's GAT model.
"""

from typing import Callable, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn.models import GAT as PyGAT
from torch_geometric.typing import SparseTensor

from .base import BaseModelWrapper


class GATWrapper(BaseModelWrapper):
    """
    Graph Attention Network (GAT) wrapper.

    Wraps :class:`torch_geometric.nn.models.GAT`.

    The GAT model from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ papers.

    Args:
        in_channels (int or tuple): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        v2 (bool, optional): If set to :obj:`True`, will make use of
            :class:`~torch_geometric.nn.conv.GATv2Conv` rather than
            :class:`~torch_geometric.nn.conv.GATConv`. (default: :obj:`False`)
        heads (int, optional): Number of attention heads. (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, will not concatenate
            head outputs. (default: :obj:`True`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        jk (str, optional): Jumping Knowledge mode. (:obj:`None`,
            :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"lstm"`).
            (default: :obj:`None`)
    """

    _pyg_model_cls = PyGAT

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        v2: bool = False,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        norm: Union[str, Callable, None] = None,
        jk: Optional[str] = None,
    ):
        self.v2 = v2
        self.heads = heads
        self.concat = concat

        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            dropout=dropout,
            act=act,
            act_first=act_first,
            norm=norm,
            jk=jk,
        )

    def _init_pyg_model(self, **kwargs):
        return self._pyg_model_cls(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            out_channels=self.out_channels,
            v2=self.v2,
            heads=self.heads,
            concat=self.concat,
            dropout=self.dropout if hasattr(self, 'dropout') else 0.0,
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            jk=self.jk_mode,
            **kwargs,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Union[Tensor, SparseTensor],
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node feature tensor.
            edge_index: Edge indices.
            edge_attr: Optional edge features.

        Returns:
            Node embeddings.
        """
        return self._model(x, edge_index, edge_attr=edge_attr)

    def get_embeddings(self, data) -> Tensor:
        """
        Get node embeddings for a Data object.

        Args:
            data: PyG Data object with x and edge_index.

        Returns:
            Node embeddings tensor.
        """
        return self.forward(data.x, data.edge_index)
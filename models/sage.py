"""
GraphSAGE model wrapper.
Wraps PyTorch Geometric's GraphSAGE model.
"""

from typing import Callable, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn.models import GraphSAGE as PyGGraphSAGE
from torch_geometric.typing import SparseTensor

from .base import BaseModelWrapper


class SAGEWrapper(BaseModelWrapper):
    """
    GraphSAGE (SAmple and aggreGatE) wrapper.

    Wraps :class:`torch_geometric.nn.models.GraphSAGE`.

    The GraphSAGE model from the `"Inductive Representation Learning on Large
    Graphs" <https://arxiv.org/abs/1706.02216>`_ paper.

    Args:
        in_channels (int or tuple): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
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
        aggr (str, optional): Aggregation scheme (:obj:`"mean"`,
            :obj:`"max"`, :obj:`"lstm"`). (default: :obj:`"mean"`)
        normalize (bool, optional): If set to :obj:`True`, output features
            will be L2-normalized. (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, will not add
            transformed root node features to the output. (default: :obj:`True`)
        project (bool, optional): If set to :obj:`True`, will apply a linear
            transformation before aggregation. (default: :obj:`False`)
    """

    _pyg_model_cls = PyGGraphSAGE

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        norm: Union[str, Callable, None] = None,
        jk: Optional[str] = None,
        aggr: str = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
    ):
        self.aggr = aggr
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

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
            dropout=self.dropout_p,
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            jk=self.jk_mode,
            aggr=self.aggr,
            normalize=self.normalize,
            root_weight=self.root_weight,
            project=self.project,
            **kwargs,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Union[Tensor, SparseTensor],
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node feature tensor.
            edge_index: Edge indices.
            edge_weight: Optional edge weights (not used by SAGEConv).

        Returns:
            Node embeddings.
        """
        # Note: SAGEConv doesn't support edge_weight, so we ignore it
        return self._model(x, edge_index)

    def get_embeddings(self, data) -> Tensor:
        """
        Get node embeddings for a Data object.

        Args:
            data: PyG Data object with x and edge_index.

        Returns:
            Node embeddings tensor.
        """
        return self.forward(data.x, data.edge_index)
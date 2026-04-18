"""
Base model wrapper for GNN models.
Provides a common interface for GNN model wrappers.
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import Module


class BaseModelWrapper(ABC):
    """
    Abstract base wrapper for PyG GNN models.

    Provides a common interface for GNN model wrappers.
    """

    # The underlying PyG model class
    _pyg_model_cls: type = None

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
    ):
        """
        Initialize the model wrapper.

        Args:
            in_channels: Size of each input sample.
            hidden_channels: Size of each hidden sample.
            num_layers: Number of message passing layers.
            out_channels: Size of output sample. If None, no final linear layer.
            dropout: Dropout probability.
            act: Activation function name or callable.
            act_first: If True, apply activation before normalization.
            norm: Normalization function name or callable.
            jk: Jumping Knowledge mode.
        """
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels if out_channels is not None else hidden_channels
        self.dropout = torch.nn.Dropout(p=dropout)
        self.act = act
        self.act_first = act_first
        self.norm = norm
        self.jk_mode = jk

        self._model: Optional[Module] = None

        # Build the model on initialization
        self._build_model()

    @abstractmethod
    def _init_pyg_model(self, **kwargs):
        """
        Initialize the underlying PyG model.

        Args:
            **kwargs: Additional arguments for the PyG model.

        Returns:
            The PyG model instance.
        """
        pass

    def _build_model(self):
        """Build the underlying PyG model."""
        if self._model is None:
            self._model = self._init_pyg_model()

    def reset_parameters(self):
        """Reset all learnable parameters."""
        if self._model is not None and hasattr(self._model, 'reset_parameters'):
            self._model.reset_parameters()

    def forward(self, *args, **kwargs) -> Tensor:
        """
        Forward pass.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError()

    @torch.no_grad()
    def get_embeddings(self, *args, **kwargs) -> Tensor:
        """
        Get node embeddings.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError()

    def train(self, mode: bool = True):
        """Set the model to training mode."""
        if self._model is not None:
            self._model.train(mode)
        return self

    def eval(self):
        """Set the model to evaluation mode."""
        return self.train(False)

    def parameters(self, recurse: bool = True):
        """Return an iterator over model parameters."""
        if self._model is not None:
            return self._model.parameters(recurse)
        # For wrappers that don't use _model, return empty iterator
        return iter(())

    def to(self, *args, **kwargs):
        """Move the model to the specified device."""
        if self._model is not None:
            self._model = self._model.to(*args, **kwargs)
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, num_layers={self.num_layers})"
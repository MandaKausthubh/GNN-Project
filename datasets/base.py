"""
Base wrapper class for PyTorch Geometric datasets.
Provides a common interface for all dataset wrappers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Any

import torch
from torch_geometric.data import Data, Dataset


class BaseDatasetWrapper(Dataset, ABC):
    """
    Abstract base wrapper for PyTorch Geometric datasets.

    Provides common functionality while delegating core dataset
    operations to the underlying PyG dataset.
    """

    # The underlying PyG dataset class
    _pyg_dataset_cls: type = None

    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        """
        Initialize the dataset wrapper.

        Args:
            root: Root directory where data is stored.
            transform: Transform to apply to each graph.
            pre_transform: Transform to apply once before loading.
            pre_filter: Filter to apply before returning graphs.
            force_reload: Force reload of cached data.
        """
        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )

        # Initialize the underlying PyG dataset
        self._dataset = self._pyg_dataset_cls(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )

    @abstractmethod
    def _get_data(self, idx: int) -> Data:
        """
        Get data from underlying dataset.

        Args:
            idx: Index of the graph.

        Returns:
            PyG Data object.
        """
        pass

    def len(self) -> int:
        """Returns the number of graphs in the dataset."""
        return len(self._dataset)

    def get(self, idx: int) -> Data:
        """Get a single graph by index."""
        return self._get_data(idx)

    @property
    def name(self) -> str:
        """Dataset name."""
        return self._dataset.name

    @property
    def raw_dir(self) -> str:
        """Raw directory path."""
        return self._dataset.raw_dir

    @property
    def processed_dir(self) -> str:
        """Processed directory path."""
        return self._dataset.processed_dir

    @property
    def raw_file_names(self) -> list:
        """Raw file names."""
        return self._dataset.raw_file_names

    @property
    def processed_file_names(self) -> list:
        """Processed file names."""
        return self._dataset.processed_file_names

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(root={self.root})"


class SingleGraphWrapper(BaseDatasetWrapper):
    """
    Wrapper for PyG datasets that contain a single graph.
    """

    def _get_data(self, idx: int) -> Data:
        """Get the single graph from the underlying dataset."""
        if idx != 0:
            raise IndexError(f"Index {idx} out of range for single graph dataset")
        return self._dataset[0]

    def __len__(self) -> int:
        return 1
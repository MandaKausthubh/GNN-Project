"""
Base wrapper class for PyTorch Geometric datasets.
Provides a common interface for all dataset wrappers.
"""

import os
import zipfile
from abc import ABC, abstractmethod
from typing import Optional, Callable, List

from torch_geometric.data import Data


class BaseDatasetWrapper(ABC):
    """
    Abstract base wrapper for PyTorch Geometric datasets.

    Provides a common interface for dataset wrappers.
    """

    # The underlying PyG dataset class
    _pyg_dataset_cls: type = None

    # URL for downloading the dataset (override in subclass)
    url: Optional[str] = None

    # List of raw file names to check for
    _raw_file_names: List[str] = []

    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        """
        Initialize the dataset wrapper.

        Args:
            root: Root directory where data is stored.
            transform: Transform to apply to each graph.
            pre_transform: Transform to apply once before loading.
            force_reload: Force reload of cached data.
        """
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.force_reload = force_reload

        # Initialize the underlying PyG dataset
        self._dataset = self._init_pyg_dataset()

    @abstractmethod
    def _init_pyg_dataset(self):
        """
        Initialize the underlying PyG dataset.

        Returns:
            The PyG dataset instance.
        """
        pass

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
        return self._raw_file_names

    @property
    def processed_file_names(self) -> list:
        """Processed file names."""
        return self._dataset.processed_file_names

    def _download(self) -> None:
        """
        Download and extract the dataset if raw files are not found.

        Downloads from self.url, extracts the zip file, and cleans up.
        """
        if self.url is None:
            return

        raw_dir = self.raw_dir
        os.makedirs(raw_dir, exist_ok=True)

        # Check if all raw files exist
        if all(os.path.exists(os.path.join(raw_dir, f)) for f in self._raw_file_names):
            return

        import urllib.request

        zip_path = os.path.join(raw_dir, "dataset.zip")

        print(f"Downloading {self.url}...")
        urllib.request.urlretrieve(self.url, zip_path)

        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)

        os.remove(zip_path)
        print("Download complete.")

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
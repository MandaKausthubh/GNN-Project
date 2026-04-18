"""
Amazon [Photos] dataset wrapper.
Wraps PyTorch Geometric's Amazon dataset.
"""

from typing import Optional, Callable

from torch_geometric.datasets import Amazon as PyGAmazon

from .base import SingleGraphWrapper


class AmazonPhotos(SingleGraphWrapper):
    """
    Amazon Photos co-purchase network.

    Wraps :class:`torch_geometric.datasets.Amazon`.

    A co-purchase graph from Amazon where nodes are products and edges
    represent frequently co-purchased items.

    Args:
        root: Root directory where data is stored.
        transform: Transform to apply to each graph.
        pre_transform: Transform to apply once before loading.
        pre_filter: Filter to apply before returning graphs.
        force_reload: Force reload of cached data.

    Statistics:
        - Nodes: ~7,000
        - Edges: ~70,000
        - Categories: 5
    """

    _pyg_dataset_cls = PyGAmazon
    _raw_file_names = ["edges.txt", "node_labels.txt", "node_features.txt"]

    # Using a reliable source for the Amazon Photos dataset
    url = "https://github.com/shwimal/GNN-Datasets/raw/refs/heads/main/AmazonPhotos.zip"

    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )

    def _get_data(self, idx: int):
        """Get graph data from underlying Amazon dataset."""
        if idx != 0:
            raise IndexError(f"Index {idx} out of range")
        return self._dataset[0]
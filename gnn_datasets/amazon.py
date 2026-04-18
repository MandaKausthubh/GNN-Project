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

    Wraps :class:`torch_geometric.datasets.Amazon` with name="Photo".

    A co-purchase graph from Amazon where nodes are products and edges
    represent frequently co-purchased items.

    Args:
        root: Root directory where data is stored.
        transform: Transform to apply to each graph.
        pre_transform: Transform to apply once before loading.
        force_reload: Force reload of cached data.

    Statistics:
        - Nodes: ~7,650
        - Edges: ~238,162
        - Categories: 8
    """

    _pyg_dataset_cls = PyGAmazon

    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            force_reload=force_reload,
        )

    def _init_pyg_dataset(self):
        # Amazon dataset requires name="photo" or "computers"
        return self._pyg_dataset_cls(
            root=self.root,
            name="photo",
            transform=self.transform,
            pre_transform=self.pre_transform,
            force_reload=self.force_reload,
        )

    def _get_data(self, idx: int):
        """Get graph data from underlying Amazon dataset."""
        if idx != 0:
            raise IndexError(f"Index {idx} out of range")
        return self._dataset[0]
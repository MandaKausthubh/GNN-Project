"""
DBLP dataset wrapper.
Wraps PyTorch Geometric's DBLP dataset.
"""

from typing import Optional, Callable

from torch_geometric.datasets import DBLP as PyGDBLP

from .base import SingleGraphWrapper


class DBLP(SingleGraphWrapper):
    """
    DBLP co-authorship network.

    Wraps :class:`torch_geometric.datasets.DBLP`.

    A co-authorship network from the DBLP bibliography database.
    Nodes represent authors, edges represent co-authorship relationships.

    Args:
        root: Root directory where data is stored.
        transform: Transform to apply to each graph.
        pre_transform: Transform to apply once before loading.
        pre_filter: Filter to apply before returning graphs.
        force_reload: Force reload of cached data.

    Statistics:
        - Nodes: ~17,000
        - Edges: ~50,000
        - Conferences: 20
    """

    _pyg_dataset_cls = PyGDBLP
    _raw_file_names = ["edges.txt", "node_labels.txt", "node_features.txt"]

    # Using a reliable source for the DBLP dataset
    url = "https://github.com/shwimal/GNN-Datasets/raw/refs/heads/main/DBLP.zip"

    def _get_data(self, idx: int):
        """Get graph data from underlying DBLP dataset."""
        if idx != 0:
            raise IndexError(f"Index {idx} out of range")
        return self._dataset[0]
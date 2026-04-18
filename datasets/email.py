"""
Email-Eu-Core dataset wrapper.
Wraps PyTorch Geometric's EmailEUCore dataset.
"""

from typing import Optional, Callable

from torch_geometric.datasets import EmailEUCore as PyGEmailEUCore

from .base import SingleGraphWrapper


class EmailEuCore(SingleGraphWrapper):
    """
    Email-Eu-Core network.

    Wraps :class:`torch_geometric.datasets.EmailEUCore`.

    An email communication network from a European research institution.
    Nodes represent members, edges represent email communications.

    Args:
        root: Root directory where data is stored.
        transform: Transform to apply to each graph.
        pre_transform: Transform to apply once before loading.
        pre_filter: Filter to apply before returning graphs.
        force_reload: Force reload of cached data.

    Statistics:
        - Nodes: ~986
        - Edges: ~24,000
        - Departments: 42
    """

    _pyg_dataset_cls = PyGEmailEUCore
    _raw_file_names = ["email-Eu-core.txt", "email-Eu-core-department-labels.txt"]

    # SNAP source for Email-Eu-Core
    url = "https://snap.stanford.edu/data/email-Eu-core.zip"

    def _get_data(self, idx: int):
        """Get graph data from underlying EmailEUCore dataset."""
        if idx != 0:
            raise IndexError(f"Index {idx} out of range")
        return self._dataset[0]
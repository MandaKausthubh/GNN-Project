"""
Email-Eu-Core dataset wrapper.
Wraps PyTorch Geometric's EmailEUCore dataset.
"""

from typing import Optional, Callable, Literal

import torch
from torch import Tensor
from torch_geometric.datasets import EmailEUCore as PyGEmailEUCore
from torch_geometric.utils import to_undirected

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
        force_reload: Force reload of cached data.
        feature_mode: Feature mode for ablation studies:
            - "none": Use original node features (department labels as input)
            - "degree": Degree-based features (in_degree, out_degree, total_degree, log_total_degree)
            - "centrality": Centrality-based features (betweenness, closeness, pagerank, eigenvector)
            - "local": Local structure features (clustering_coef, triangle_count, k_core_number)

    Statistics:
        - Nodes: ~986
        - Edges: ~24,000
        - Departments: 42
    """

    _pyg_dataset_cls = PyGEmailEUCore

    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        feature_mode: Literal["none", "degree", "centrality", "local"] = "none",
    ):
        self.feature_mode = feature_mode
        self._cached_data = None
        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            force_reload=force_reload,
        )

    def _init_pyg_dataset(self):
        return self._pyg_dataset_cls(
            root=self.root,
            transform=self.transform,
            pre_transform=self.pre_transform,
            force_reload=self.force_reload,
        )

    def _get_data(self, idx: int):
        """Get graph data from underlying EmailEUCore dataset with selected feature mode."""
        if idx != 0:
            raise IndexError(f"Index {idx} out of range")

        # Return cached data if available
        if self._cached_data is not None:
            return self._cached_data

        data = self._dataset[0]

        # Apply feature mode
        if self.feature_mode != "none":
            data = self._apply_feature_mode(data, self.feature_mode)

        self._cached_data = data
        return data

    def _apply_feature_mode(self, data, mode: str):
        """Apply feature mode to generate ablation features."""
        from torch_geometric.data import Data

        # Ensure undirected graph for feature computation
        edge_index = to_undirected(data.edge_index)
        num_nodes = data.num_nodes

        if mode == "degree":
            x = self._compute_degree_features(edge_index, num_nodes)
        elif mode == "centrality":
            x = self._compute_centrality_features(edge_index, num_nodes)
        elif mode == "local":
            x = self._compute_local_features(edge_index, num_nodes)
        else:
            return data

        # Preserve labels and masks
        new_data = Data(
            x=x,
            edge_index=edge_index,
            y=data.y if hasattr(data, 'y') else None,
            train_mask=data.train_mask if hasattr(data, 'train_mask') else None,
            val_mask=data.val_mask if hasattr(data, 'val_mask') else None,
            test_mask=data.test_mask if hasattr(data, 'test_mask') else None,
        )

        return new_data

    def _compute_degree_features(self, edge_index: Tensor, num_nodes: int) -> Tensor:
        """
        Compute degree-based features.

        Features (4-dimensional):
            - in_degree: Number of incoming edges
            - out_degree: Number of outgoing edges
            - total_degree: Total degree (in + out)
            - log_total_degree: log(total_degree + 1) for normalization
        """
        # Compute in-degree and out-degree
        in_degree = torch.bincount(edge_index[1], minlength=num_nodes).float()
        out_degree = torch.bincount(edge_index[0], minlength=num_nodes).float()

        # For undirected graph, in_degree == out_degree
        total_degree = in_degree + out_degree
        log_total_degree = torch.log(total_degree + 1)

        # Stack features: [num_nodes, 4]
        x = torch.stack([in_degree, out_degree, total_degree, log_total_degree], dim=1)

        # Normalize features
        x = self._normalize_features(x)

        return x

    def _compute_centrality_features(self, edge_index: Tensor, num_nodes: int) -> Tensor:
        """
        Compute centrality-based features.

        Features (4-dimensional):
            - betweenness_centrality: Fraction of shortest paths passing through node
            - closeness_centrality: Reciprocal of average distance to all nodes
            - pagerank: PageRank score
            - eigenvector_centrality: Eigenvector centrality score
        """
        import networkx as nx

        # Build networkx graph directly from edge_index
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)

        # Compute centralities using networkx
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        pagerank = nx.pagerank(G, max_iter=100)

        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        except (nx.PowerIterationFailedConvergence, Exception):
            # Fallback: use pagerank if eigenvector centrality fails to converge
            eigenvector = pagerank

        # Convert to tensors
        betweenness_feat = torch.tensor([betweenness.get(i, 0.0) for i in range(num_nodes)])
        closeness_feat = torch.tensor([closeness.get(i, 0.0) for i in range(num_nodes)])
        pagerank_feat = torch.tensor([pagerank.get(i, 0.0) for i in range(num_nodes)])
        eigenvector_feat = torch.tensor([eigenvector.get(i, 0.0) for i in range(num_nodes)])

        # Stack features: [num_nodes, 4]
        x = torch.stack([betweenness_feat, closeness_feat, pagerank_feat, eigenvector_feat], dim=1)

        # Normalize features
        x = self._normalize_features(x)

        return x

    def _compute_local_features(self, edge_index: Tensor, num_nodes: int) -> Tensor:
        """
        Compute local structure features.

        Features (3-dimensional):
            - clustering_coefficient: Local clustering coefficient
            - triangle_count: Number of triangles the node participates in
            - k_core_number: Core number of the node
        """
        import networkx as nx

        # Remove self-loops before building graph
        mask = edge_index[0] != edge_index[1]
        edge_index_clean = edge_index[:, mask]

        # Build networkx graph directly from edge_index
        G = nx.Graph()  # Undirected for local clustering
        G.add_nodes_from(range(num_nodes))
        edges = edge_index_clean.t().tolist()
        G.add_edges_from(edges)

        # Compute local clustering coefficients
        clustering = nx.clustering(G)
        clustering_feat = torch.tensor([clustering.get(i, 0.0) for i in range(num_nodes)])

        # Compute triangle counts (clustering * possible_triangles)
        # For each node, triangles = clustering * degree * (degree - 1) / 2
        degree = torch.bincount(edge_index[0], minlength=num_nodes)
        triangle_count = torch.zeros(num_nodes, dtype=torch.float)
        for i in range(num_nodes):
            d = degree[i].item()
            if d >= 2:
                possible_triangles = d * (d - 1) / 2
                triangle_count[i] = clustering.get(i, 0.0) * possible_triangles

        # Compute k-core numbers
        core_numbers = nx.core_number(G)
        kcore_feat = torch.tensor([core_numbers.get(i, 0) for i in range(num_nodes)], dtype=torch.float)

        # Stack features: [num_nodes, 3]
        x = torch.stack([clustering_feat, triangle_count, kcore_feat], dim=1)

        # Normalize features
        x = self._normalize_features(x)

        return x

    def _normalize_features(self, x: Tensor) -> Tensor:
        """
        Normalize features to [0, 1] range using min-max normalization.

        Args:
            x: Feature tensor of shape [num_nodes, num_features]

        Returns:
            Normalized feature tensor
        """
        eps = 1e-8
        x_min = x.min(dim=0, keepdim=True).values
        x_max = x.max(dim=0, keepdim=True).values
        x_range = x_max - x_min
        x_range[x_range < eps] = 1.0  # Avoid division by zero for constant features
        x_normalized = (x - x_min) / x_range
        return x_normalized

    def reset_cache(self):
        """Reset cached data, useful for switching feature modes."""
        self._cached_data = None
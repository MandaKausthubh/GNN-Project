"""
DBLP dataset wrapper.
Wraps PyTorch Geometric's DBLP dataset.
"""

from typing import Optional, Callable

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.datasets import DBLP as PyGDBLP
from torch_geometric.utils import to_undirected

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
        force_reload: Force reload of cached data.

    Statistics:
        - Authors: 4,057
        - Papers: 14,328
        - Terms: 7,723
        - Conferences: 20
        - Author-Paper edges: 196,425
        - Research areas: 4 (database, data mining, AI, IR)
    """

    _pyg_dataset_cls = PyGDBLP

    def _init_pyg_dataset(self):
        return self._pyg_dataset_cls(
            root=self.root,
            transform=self.transform,
            pre_transform=self.pre_transform,
            force_reload=self.force_reload,
        )

    def _get_data(self, idx: int):
        """Get graph data from underlying DBLP dataset."""
        if idx != 0:
            raise IndexError(f"Index {idx} out of range")
        return self._dataset[0]

    def get_homograph_apa(self) -> Data:
        """
        Compute the Author-Paper-Author (APA) meta-path based projection.

        Creates a homogeneous author-author graph where two authors are
        connected if they co-authored at least one paper.

        Returns:
            Data object with:
                - x: Author node features (334-dim)
                - edge_index: Author-author edges
                - y: Author labels (research area, 4 classes)
                - train_mask, val_mask, test_mask: Split masks
        """
        data = self._dataset[0]

        if not isinstance(data, HeteroData):
            raise ValueError("DBLP data is not HeteroData")

        # Extract author-paper edges
        author_paper_edge = data['author', 'to', 'paper'].edge_index
        authors = author_paper_edge[0]
        papers = author_paper_edge[1]

        num_authors = data['author'].num_nodes
        num_papers = data['paper'].num_nodes

        # For each paper, collect all authors who wrote it
        # Then create author-author edges for each pair
        # Using scatter to group author indices by paper

        # Sort by paper ID to group authors by paper
        sorted_paper_ids, paper_order = papers.sort()
        sorted_authors = authors[paper_order]

        # Find boundaries between papers
        paper_diff = torch.cat([
            torch.tensor([0], device=papers.device),
            (sorted_paper_ids[1:] != sorted_paper_ids[:-1]).nonzero(as_tuple=True)[0] + 1,
            torch.tensor([sorted_paper_ids.size(0)], device=papers.device)
        ])

        # Create author-author edges from co-authorship
        src_authors = []
        dst_authors = []

        for i in range(len(paper_diff) - 1):
            start, end = paper_diff[i], paper_diff[i + 1]
            paper_authors = sorted_authors[start:end]

            # Create edges between all pairs of co-authors
            num_coauthors = paper_authors.size(0)
            if num_coauthors > 1:
                # Add all pairs (will dedupe later)
                for j in range(num_coauthors):
                    for k in range(j + 1, num_coauthors):
                        src_authors.append(paper_authors[j].item())
                        dst_authors.append(paper_authors[k].item())

        if not src_authors:
            # No co-authorships found (unlikely case)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor([src_authors, dst_authors], dtype=torch.long)

            # Make undirected and remove self-loops
            edge_index = to_undirected(edge_index)
            edge_index, _ = torch.unique(edge_index, dim=1, return_inverse=True)

        # Check for isolated nodes
        if edge_index.numel() > 0:
            edge_index, _ = to_undirected(edge_index)
            # Remove any remaining self-loops
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]

        # Create Data object
        proj_data = Data(
            x=data['author'].x,
            edge_index=edge_index,
            y=data['author'].y,
        )

        # Add masks if available
        if hasattr(data['author'], 'train_mask'):
            proj_data.train_mask = data['author'].train_mask
        if hasattr(data['author'], 'val_mask'):
            proj_data.val_mask = data['author'].val_mask
        if hasattr(data['author'], 'test_mask'):
            proj_data.test_mask = data['author'].test_mask

        return proj_data
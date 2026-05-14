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

    def _require_hetero_data(self) -> HeteroData:
        """Return DBLP graph as HeteroData."""
        data = self._dataset[0]
        if not isinstance(data, HeteroData):
            raise ValueError("DBLP data is not HeteroData")
        return data

    @staticmethod
    def _get_edge_index(data: HeteroData, src_type: str, dst_type: str) -> torch.Tensor:
        """Get edge_index between node types, accepting reverse edge direction."""
        for edge_type in data.edge_types:
            src, _, dst = edge_type
            edge_index = data[edge_type].edge_index
            if src == src_type and dst == dst_type:
                return edge_index
            if src == dst_type and dst == src_type:
                return edge_index.flip(0)
        raise ValueError(f"DBLP edge type {src_type}->{dst_type} not found")

    @staticmethod
    def _project_authors_by_group(
        data: HeteroData,
        authors: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> Data:
        """Create author-author graph by connecting authors sharing a group ID."""
        device = authors.device

        # Sort by group ID to collect authors per paper/conference.
        sorted_group_ids, group_order = group_ids.sort()
        sorted_authors = authors[group_order]

        group_diff = torch.cat([
            torch.tensor([0], device=device),
            (sorted_group_ids[1:] != sorted_group_ids[:-1]).nonzero(as_tuple=True)[0] + 1,
            torch.tensor([sorted_group_ids.size(0)], device=device),
        ])

        src_authors = []
        dst_authors = []

        for i in range(len(group_diff) - 1):
            start, end = group_diff[i], group_diff[i + 1]
            group_authors = torch.unique(sorted_authors[start:end])

            num_coauthors = group_authors.size(0)
            if num_coauthors > 1:
                for j in range(num_coauthors):
                    for k in range(j + 1, num_coauthors):
                        src_authors.append(group_authors[j].item())
                        dst_authors.append(group_authors[k].item())

        if not src_authors:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.tensor([src_authors, dst_authors], dtype=torch.long, device=device)
            edge_index = to_undirected(edge_index)
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
            edge_index = torch.unique(edge_index, dim=1)

        proj_data = Data(
            x=data['author'].x,
            edge_index=edge_index,
            y=data['author'].y,
        )

        if hasattr(data['author'], 'train_mask'):
            proj_data.train_mask = data['author'].train_mask
        if hasattr(data['author'], 'val_mask'):
            proj_data.val_mask = data['author'].val_mask
        if hasattr(data['author'], 'test_mask'):
            proj_data.test_mask = data['author'].test_mask

        return proj_data

    def get_homograph_apa(self) -> Data:
        """
        Compute Author-Paper-Author (APA) meta-path projection.

        Authors are connected when they co-authored at least one paper.
        """
        data = self._require_hetero_data()
        author_paper_edge = self._get_edge_index(data, 'author', 'paper')
        return self._project_authors_by_group(
            data=data,
            authors=author_paper_edge[0],
            group_ids=author_paper_edge[1],
        )

    def get_homograph_aca(self) -> Data:
        """
        Compute Author-Conference-Author (ACA) meta-path projection.

        Authors are connected when they published in at least one same conference.
        """
        data = self._require_hetero_data()
        author_paper_edge = self._get_edge_index(data, 'author', 'paper')
        paper_conference_edge = self._get_edge_index(data, 'paper', 'conference')

        authors = author_paper_edge[0]
        papers = author_paper_edge[1]
        paper_ids = paper_conference_edge[0]
        conference_ids = paper_conference_edge[1]

        paper_to_conference = torch.full(
            (data['paper'].num_nodes,),
            -1,
            dtype=torch.long,
            device=papers.device,
        )
        paper_to_conference[paper_ids.to(papers.device)] = conference_ids.to(papers.device)
        author_conferences = paper_to_conference[papers]
        valid_mask = author_conferences >= 0

        return self._project_authors_by_group(
            data=data,
            authors=authors[valid_mask],
            group_ids=author_conferences[valid_mask],
        )

    def get_homograph_apa_aca(self) -> Data:
        """
        Compute combined APA + ACA author projection.

        Edges are union of shared-paper and shared-conference author links.
        """
        apa_data = self.get_homograph_apa()
        aca_data = self.get_homograph_aca()
        edge_index = torch.cat([apa_data.edge_index, aca_data.edge_index], dim=1)
        edge_index = torch.unique(edge_index, dim=1)
        apa_data.edge_index = edge_index
        return apa_data

    def get_homograph(self, metapath: str = "apa") -> Data:
        """Return DBLP author homograph for selected meta-path."""
        if metapath == "apa":
            return self.get_homograph_apa()
        if metapath == "aca":
            return self.get_homograph_aca()
        if metapath in {"apa_aca", "both"}:
            return self.get_homograph_apa_aca()
        raise ValueError(f"Unknown DBLP meta-path: {metapath}")

    def aca(self) -> Data:
        """Alias for get_homograph_aca."""
        return self.get_homograph_aca()

    def apa(self) -> Data:
        """Alias for get_homograph_apa."""
        return self.get_homograph_apa()

    def apa_aca(self) -> Data:
        """Alias for get_homograph_apa_aca."""
        return self.get_homograph_apa_aca()

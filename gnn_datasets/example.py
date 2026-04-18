"""
Example usage of the GNN dataset wrappers.
"""

import argparse
import os
from gnn_datasets import AmazonPhotos, EmailEuCore, DBLP
from torch_geometric.data import Data, HeteroData


def print_homo_data_stats(data: Data, name: str):
    """Print stats for homogeneous graph Data object."""
    print(f"Type: Homogeneous Graph")
    print(f"Number of nodes: {data.num_nodes}")
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        print(f"Number of edges: {data.edge_index.shape[1]}")
    if hasattr(data, 'x') and data.x is not None:
        print(f"Node features shape: {data.x.shape}")
    if hasattr(data, 'y') and data.y is not None:
        print(f"Node labels shape: {data.y.shape}")
        print(f"Number of classes: {data.y.max().item() + 1}")


def print_hetero_data_stats(data: HeteroData, name: str):
    """Print stats for heterogeneous graph HeteroData object."""
    print(f"Type: Heterogeneous Graph")
    print(f"Node types: {data.node_types}")
    print(f"Edge types: {data.edge_types}")

    # Print per-type stats
    print("\n  Node counts:")
    for node_type in data.node_types:
        num = data[node_type].num_nodes
        print(f"    {node_type}: {num}")

    print("\n  Edge counts:")
    for edge_type in data.edge_types:
        num = data[edge_type].num_edges
        print(f"    {edge_type}: {num}")

    # Print features/labels if available
    print("\n  Node features/labels:")
    for node_type in data.node_types:
        node_data = data[node_type]
        info = []
        if hasattr(node_data, 'x') and node_data.x is not None:
            info.append(f"x={node_data.x.shape}")
        if hasattr(node_data, 'y') and node_data.y is not None:
            info.append(f"y={node_data.y.shape}")
        if info:
            print(f"    {node_type}: {', '.join(info)}")


def main():
    parser = argparse.ArgumentParser(
        description="GNN Dataset Loader - Download and inspect graph datasets"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Root directory for downloading datasets (default: ./data)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["amazon", "email", "dblp", "all"],
        default="all",
        help="Which dataset to load (default: all)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print detailed statistics for each graph",
    )

    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    print("=== GNN Dataset Examples ===\n")
    print(f"Data directory: {os.path.abspath(args.data_dir)}\n")

    datasets = {}

    if args.dataset in ("amazon", "all"):
        print("Loading AmazonPhotos dataset...")
        datasets["AmazonPhotos"] = AmazonPhotos(
            root=os.path.join(args.data_dir, "AmazonPhotos")
        )

    if args.dataset in ("email", "all"):
        print("Loading EmailEuCore dataset...")
        datasets["EmailEuCore"] = EmailEuCore(
            root=os.path.join(args.data_dir, "EmailEuCore")
        )

    if args.dataset in ("dblp", "all"):
        print("Loading DBLP dataset...")
        datasets["DBLP"] = DBLP(
            root=os.path.join(args.data_dir, "DBLP")
        )

    for name, dataset in datasets.items():
        print(f"\n{'=' * 40}")
        print(f"{name}")
        print(f"{'=' * 40}")
        print(f"Root: {dataset.root}")

        data = dataset[0]

        if isinstance(data, HeteroData):
            print_hetero_data_stats(data, name)
        else:
            print_homo_data_stats(data, name)

    print()


if __name__ == "__main__":
    main()
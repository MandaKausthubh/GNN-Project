"""
Example usage of the GNN dataset wrappers.
"""

import argparse
import os
from datasets import AmazonPhotos, EmailEuCore, DBLP


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

        data = dataset[0]
        print(f"Root: {dataset.root}")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.edge_index.shape[1]}")

        if args.stats:
            print(f"Has node features: {hasattr(data, 'x') and data.x is not None}")
            print(f"Has node labels: {hasattr(data, 'y') and data.y is not None}")
            print(f"Has edge attributes: {hasattr(data, 'edge_attr') and data.edge_attr is not None}")

            if hasattr(data, "x") and data.x is not None:
                print(f"Node feature dim: {data.x.shape}")
            if hasattr(data, "y") and data.y is not None:
                num_classes = data.y.max().item() + 1
                print(f"Number of classes: {num_classes}")
            if hasattr(data, "edge_attr") and data.edge_attr is not None:
                print(f"Edge attribute dim: {data.edge_attr.shape}")

    print()


if __name__ == "__main__":
    main()
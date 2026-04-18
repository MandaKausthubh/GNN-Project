"""
Example usage of the GNN dataset wrappers.
"""

from datasets import AmazonPhotos, EmailEuCore, DBLP


def main():
    print("=== GNN Dataset Wrapper Examples ===\n")

    # Amazon Photos
    print("AmazonPhotos dataset:")
    amazon = AmazonPhotos(root="./data/AmazonPhotos")
    data = amazon[0]
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.shape[1]}")
    print(f"  Features: {data.x.shape if hasattr(data, 'x') else 'None'}")
    print(f"  Labels: {data.y.shape if hasattr(data, 'y') else 'None'}")

    # Email-Eu-Core
    print("\nEmailEuCore dataset:")
    email = EmailEuCore(root="./data/EmailEuCore")
    data = email[0]
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.shape[1]}")
    print(f"  Features: {data.x.shape if hasattr(data, 'x') else 'None'}")
    print(f"  Labels: {data.y.shape if hasattr(data, 'y') else 'None'}")

    # DBLP
    print("\nDBLP dataset:")
    dblp = DBLP(root="./data/DBLP")
    data = dblp[0]
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.shape[1]}")
    print(f"  Features: {data.x.shape if hasattr(data, 'x') else 'None'}")
    print(f"  Labels: {data.y.shape if hasattr(data, 'y') else 'None'}")


if __name__ == "__main__":
    main()
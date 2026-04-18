"""
Example usage of GNN model wrappers.
"""

import argparse
import os

import torch
from gnn_datasets import AmazonPhotos, EmailEuCore, DBLP
from models import GCNWrapper, GATWrapper, SAGEWrapper


def train_epoch(model, data, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    """Evaluate model on given mask."""
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    correct = (pred[data.y[mask]] == data.y[mask]).sum()
    acc = int(correct) / int(mask.sum())

    return acc


def main():
    parser = argparse.ArgumentParser(description="GNN Model Training")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["amazon", "dblp"],
        default="amazon",
        help="Dataset to use (default: amazon)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gcn", "gat", "sage"],
        default="gcn",
        help="Model to use (default: gcn)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Data directory (default: ./data)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=128,
        help="Hidden channels (default: 128)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of GNN layers (default: 2)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability (default: 0.5)",
    )

    args = parser.parse_args()

    print("=== GNN Model Example ===\n")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Hidden channels: {args.hidden_channels}")
    print(f"Num layers: {args.num_layers}")
    print(f"Dropout: {args.dropout}\n")

    # Load dataset
    if args.dataset == "amazon":
        dataset = AmazonPhotos(root=os.path.join(args.data_dir, "AmazonPhotos"))
    elif args.dataset == "dblp":
        dataset = DBLP(root=os.path.join(args.data_dir, "DBLP"))
        # Use APA homograph for DBLP
        data = dataset.get_homograph_apa()
        print("Using DBLP APA homograph projection")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    data = dataset[0]

    print(f"Graph: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
    num_classes = data.y.max().item() + 1
    in_channels = data.x.shape[1]
    print(f"Features: {in_channels}, Classes: {num_classes}")

    # Create model
    if args.model == "gcn":
        model = GCNWrapper(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            out_channels=num_classes,
            dropout=args.dropout,
        )
    elif args.model == "gat":
        model = GATWrapper(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            out_channels=num_classes,
            dropout=args.dropout,
        )
    elif args.model == "sage":
        model = SAGEWrapper(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            out_channels=num_classes,
            dropout=args.dropout,
        )

    print(f"\nModel: {model}")
    print(f"Parameters: {sum(p.numel() for p in model._model.parameters()):,}")

    # Setup training
    optimizer = torch.optim.Adam(model._model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    print("\nTraining...")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, data, optimizer, criterion)

        if epoch % 20 == 0 or epoch == args.epochs:
            train_acc = evaluate(model, data, data.train_mask)
            val_acc = evaluate(model, data, data.val_mask)
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")

    # Final evaluation
    test_acc = evaluate(model, data, data.test_mask)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
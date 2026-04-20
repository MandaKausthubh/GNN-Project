"""
Example usage of the training utilities with wandb, residual connections, etc.
"""

import argparse
import os

import torch
import torch.nn as nn
from gnn_datasets import AmazonPhotos, EmailEuCore, DBLP
from models import GCNWrapper, GATWrapper
from utils import Trainer, ResidualGNNWrapper


def create_masks(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Create train/val/test masks for node classification if they don't exist.

    Args:
        data: PyG Data object
        train_ratio: Fraction of nodes for training
        val_ratio: Fraction of nodes for validation
        test_ratio: Fraction of nodes for testing
        seed: Random seed for reproducibility
    """
    if hasattr(data, 'train_mask') and data.train_mask is not None:
        print("Masks already exist, skipping mask creation")
        return data

    num_nodes = data.num_nodes
    torch.manual_seed(seed)

    # Create random permutation of node indices
    perm = torch.randperm(num_nodes)

    # Calculate split sizes
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)

    # Create masks
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[perm[:train_size]] = True
    data.val_mask[perm[train_size:train_size + val_size]] = True
    data.test_mask[perm[train_size + val_size:]] = True

    print(f"Created masks: train={data.train_mask.sum().item()}, "
          f"val={data.val_mask.sum().item()}, test={data.test_mask.sum().item()}")

    return data


def main():
    parser = argparse.ArgumentParser(description="GNN Training with Utils")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["amazon", "dblp", "email"],
        default="amazon",
        help="Dataset to use (default: amazon)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gcn", "gat", "residual_gcn", "residual_gat"],
        default="residual_gcn",
        help="Model type (default: residual_gcn)",
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
    parser.add_argument(
        "--norm",
        type=str,
        choices=["layer", "batch", "graph", "none"],
        default="layer",
        help="Normalization type (default: layer)",
    )
    parser.add_argument(
        "--use-residual",
        action="store_true",
        default=True,
        help="Use residual connections (default: True)",
    )
    parser.add_argument(
        "--residual-alpha",
        type=float,
        default=1.0,
        help="Residual blending alpha (default: 1.0)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Weight decay (default: 5e-4)",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Gradient clipping value (default: 1.0)",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=0,
        help="Early stopping patience (0 to disable, default: 0)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="gnn-experiments",
        help="Wandb project name (default: gnn-experiments)",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Wandb run name (default: auto-generated)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GNN Training with Residual Connections & Wandb Support")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Hidden channels: {args.hidden_channels}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Normalization: {args.norm}")
    print(f"  Use residual: {args.use_residual}")
    print(f"  Residual alpha: {args.residual_alpha}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Gradient clip: {args.gradient_clip}")
    print(f"  Early stopping: {args.early_stopping}")
    print(f"  Wandb enabled: {args.wandb}")
    print()

    # Load dataset
    if args.dataset == "amazon":
        dataset = AmazonPhotos(root=os.path.join(args.data_dir, "AmazonPhotos"))
    elif args.dataset == "dblp":
        dataset = DBLP(root=os.path.join(args.data_dir, "DBLP"))
        print("Using DBLP APA homograph projection")
        data = dataset.get_homograph_apa()
    elif args.dataset == "email":
        dataset = EmailEuCore(root=os.path.join(args.data_dir, "EmailEuCore"))
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.dataset != "dblp":
        data = dataset[0]

    # Create train/val/test masks if they don't exist
    data = create_masks(data)

    print(f"Graph stats: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
    num_classes = data.y.max().item() + 1
    in_channels = data.x.shape[1]
    print(f"Features: {in_channels}, Classes: {num_classes}")

    # Create model
    if args.model == "residual_gcn" or args.model == "residual_gat":
        model = ResidualGNNWrapper(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            out_channels=num_classes,
            model_type=args.model.replace("residual_", ""),
            dropout=args.dropout,
            norm=args.norm,
            activation="relu",
            use_residual=args.use_residual,
            residual_alpha=args.residual_alpha,
        )
    elif args.model == "gcn":
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

    print(f"\nModel: {model.__class__.__name__}")
    if hasattr(model, "_model"):
        print(f"Parameters: {sum(p.numel() for p in model._model.parameters()):,}")

    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Optional scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-5,
    )

    # Create trainer
    wandb_kwargs = None
    if args.wandb:
        wandb_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_name,
            "config": vars(args),
        }

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        use_wandb=args.wandb,
        wandb_kwargs=wandb_kwargs,
        scheduler=scheduler,
        early_stopping_patience=args.early_stopping,
        gradient_clip_val=args.gradient_clip if args.gradient_clip > 0 else None,
    )

    print(f"\nTrainer: {trainer}")

    # Train
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    history = trainer.train(
        data=data,
        epochs=args.epochs,
        val_every=1,
        print_every=10,
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    results = trainer.predict(data, "test_mask")
    print(f"Test Accuracy: {results.get('accuracy', 0):.4f}")
    print(f"Test F1 Score: {results.get('f1_score', 0):.4f}")

    # Save checkpoint
    checkpoint_path = f"./checkpoints/{args.model}_{args.dataset}.pt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    trainer.save_checkpoint(checkpoint_path)
    print(f"\nCheckpoint saved to: {checkpoint_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
"""
Robustness Analysis: Graph Structural Robustness Under Edge Removal.

Investigates how GNN models perform as graph edges are randomly removed
(10% to 50%), simulating graph structural attacks or data corruption.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (
    create_model,
    create_masks,
    get_dataset_stats,
    load_dataset,
    seed_everything,
    train_single_config,
)
from utils import Trainer


# =============================================================================
# Edge Removal Utilities
# =============================================================================

def remove_edges_fraction(
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor] = None,
    fraction: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Randomly remove a fraction of edges from the graph.

    Args:
        edge_index: Edge indices [2, num_edges].
        edge_weight: Optional edge weights.
        fraction: Fraction of edges to remove (0.0 to 1.0).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (reduced_edge_index, reduced_edge_weight or None).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    num_edges = edge_index.size(1)
    num_remove = int(num_edges * fraction)

    if num_remove == 0:
        return edge_index, edge_weight

    # Select edges to remove
    idxs = list(range(num_edges))
    remove_idxs = random.sample(idxs, num_remove)
    keep_idxs = [i for i in idxs if i not in remove_idxs]

    reduced_edge_index = edge_index[:, keep_idxs]
    reduced_edge_weight = edge_weight[keep_idxs] if edge_weight is not None else None

    return reduced_edge_index, reduced_edge_weight


def create_robustness_data(
    data,
    removal_fraction: float,
    seed: int,
) -> Any:
    """
    Create a modified data object with edges randomly removed.

    Args:
        data: PyG Data object.
        removal_fraction: Fraction of edges to remove.
        seed: Random seed.

    Returns:
        New Data object with reduced edges.
    """
    edge_index, edge_weight = remove_edges_fraction(
        data.edge_index,
        edge_weight=data.edge_attr if data.edge_attr is not None else None,
        fraction=removal_fraction,
        seed=seed,
    )

    from torch_geometric.data import Data
    return Data(
        x=data.x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
    )


# =============================================================================
# Robustness Benchmark
# =============================================================================

def benchmark_robustness(
    data_dir: str = "./data",
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    removal_fractions: Optional[List[float]] = None,
    epochs: int = 100,
    seed: int = 42,
    device: Optional[str] = None,
    output_dir: str = "./outputs",
    verbose: bool = False,
    save_plots: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark model robustness under edge removal attacks.

    Args:
        data_dir: Data root directory.
        models: List of model types.
        datasets: List of datasets to run on.
        removal_fractions: List of edge removal fractions (default: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]).
        epochs: Training epochs per run.
        seed: Fixed random seed for reproducibility.
        device: Device to train on.
        output_dir: Output directory.
        verbose: Print progress.
        save_plots: Whether to save comparison plots.

    Returns:
        Dictionary with all results.
    """
    datasets = datasets or ["amazon", "dblp", "email"]
    models = models or ["gcn", "gat", "sage", "appnp", "residual_gcn", "residual_gat", "residual_sage", "residual_appnp"]
    removal_fractions = removal_fractions or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_hyperparams = {
        "hidden_channels": 128,
        "num_layers": 2,
        "dropout": 0.5,
        "norm": "layer",
        "lr": 0.01,
        "weight_decay": 5e-4,
        "gat_heads": 4,
        "K": 10,
        "alpha": 0.1,
    }

    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    print("\n" + "=" * 80)
    print("ROBUSTNESS ANALYSIS: Edge Removal Attack (10% to 50%)")
    print("=" * 80)

    for dataset_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'=' * 60}")

        _, data = load_dataset(dataset_name, data_dir)
        data = create_masks(data) if not hasattr(data, "train_mask") else data
        stats = get_dataset_stats(data)
        if verbose:
            print(f"Dataset stats: {stats}")

        # Record original edge count
        original_edges = data.edge_index.size(1)
        print(f"Original edges: {original_edges}")

        all_results[dataset_name] = {
            "stats": stats,
            "original_edges": original_edges,
            "removals": {},
        }

        for model_name in models:
            print(f"\n  Model: {model_name.upper()}")
            all_results[dataset_name]["removals"][model_name] = {}

            for frac in removal_fractions:
                frac_pct = int(frac * 100)
                seed_everything(seed)

                # Create modified graph
                mod_data = create_robustness_data(data, frac, seed=seed + frac_pct)
                edge_count = mod_data.edge_index.size(1)

                try:
                    history, test_metrics, _ = train_single_config(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        data=mod_data,
                        hyperparams=default_hyperparams,
                        epochs=epochs,
                        device=device,
                        verbose=verbose,
                    )

                    acc = float(test_metrics["accuracy"])
                    f1 = float(test_metrics["f1_score"])
                    print(f"    {model_name.upper()} {frac_pct}% removal: Acc={acc:.4f}, F1={f1:.4f}, edges={edge_count}")

                except Exception as e:
                    print(f"    {model_name.upper()} {frac_pct}% removal: Failed ({str(e)[:60]})")
                    acc, f1, edge_count = 0.0, 0.0, 0

                all_results[dataset_name]["removals"][model_name][frac] = {
                    "accuracy": acc,
                    "f1": f1,
                    "edges_remaining": edge_count,
                }

    # Print summary table
    print("\n" + "=" * 80)
    print("[SUMMARY] Robustness: Accuracy at Each Edge Removal Level")
    print("=" * 80)

    for dataset_name in datasets:
        print(f"\n{dataset_name.upper()}")
        print(f"{'Model':<20}", end="")
        frac_headers = [f"{int(f*100):>10}%" for f in removal_fractions]
        print("".join(frac_headers))
        print("-" * (20 + 10 * len(removal_fractions)))

        removals = all_results[dataset_name]["removals"]
        for model_name in models:
            if model_name not in removals:
                continue
            row = f"{model_name:<20}"
            for frac in removal_fractions:
                if frac in removals[model_name]:
                    acc = removals[model_name][frac]["accuracy"]
                    row += f"{acc:>10.4f}"
                else:
                    row += f"{'N/A':>10}"
            print(row)
    print("=" * 80)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"robustness_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Generate plots
    if save_plots:
        print("\nGenerating robustness plots...")
        from utils.training_time_plots import plot_robustness_comparison
        plot_robustness_comparison(all_results, removal_fractions=removal_fractions, models=models, save_dir=output_dir)
        print(f"Plots saved to: {output_dir}")

    return all_results


# =============================================================================
# Argument Parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Robustness Analysis: Edge Removal Attack on GNN Models"
    )
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["gcn", "gat", "sage", "appnp", "residual_gcn", "residual_gat", "residual_sage", "residual_appnp"],
        help="Model types to evaluate",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["amazon", "dblp", "email"],
        choices=["amazon", "dblp", "email"],
        help="Datasets to run on",
    )
    parser.add_argument(
        "--removals",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        help="Edge removal fractions (0.0 to 0.5)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    return parser


# =============================================================================
# Main
# =============================================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    device = args.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    benchmark_robustness(
        data_dir=args.data_dir,
        models=args.models,
        datasets=args.datasets,
        removal_fractions=args.removals,
        epochs=args.epochs,
        seed=args.seed,
        device=device,
        output_dir=args.output_dir,
        verbose=args.verbose,
        save_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
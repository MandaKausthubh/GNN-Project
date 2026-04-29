"""
Ablation Study: Oversmoothing Analysis.

Investigates oversmoothing in GNNs by comparing performance across
2, 4, and 8 layers for GCN and ResidualGCN. Also includes GAT variants.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

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
# Ablation: Oversmoothing (Layer Depth)
# =============================================================================

def benchmark_layers(
    data_dir: str = "./data",
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    layer_counts: Optional[List[int]] = None,
    epochs: int = 100,
    n_runs: int = 3,
    device: Optional[str] = None,
    output_dir: str = "./outputs",
    verbose: bool = False,
    save_plots: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark models across different layer counts to analyze oversmoothing.

    Args:
        data_dir: Data root directory.
        models: List of model types (e.g. gcn, residual_gcn, gat).
        datasets: List of datasets to run on.
        layer_counts: List of layer counts to compare (default: [2, 4, 8]).
        epochs: Training epochs per run.
        n_runs: Number of runs per configuration.
        device: Device to train on.
        output_dir: Output directory.
        verbose: Print progress.
        save_plots: Whether to save comparison plots.

    Returns:
        Dictionary with all results.
    """
    datasets = datasets or ["amazon", "dblp", "email"]
    models = models or ["gcn", "residual_gcn", "gat", "residual_gat", "appnp", "residual_appnp"]
    layer_counts = layer_counts or [2, 4, 8]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_hyperparams = {
        "hidden_channels": 128,
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
    print("ABLATION: Oversmoothing Analysis (2, 4, 8 Layers)")
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

        all_results[dataset_name] = {"stats": stats, "layers": {}}

        for model_name in models:
            all_results[dataset_name]["layers"][model_name] = {}

            for num_layers in layer_counts:
                hyperparams = {**default_hyperparams, "num_layers": num_layers}
                model_results = {"accuracy": [], "f1_score": [], "histories": []}

                config_str = f"hid={hyperparams['hidden_channels']},layers={num_layers},lr={hyperparams['lr']}"
                run_pbar = tqdm(
                    total=n_runs,
                    desc=f"  [{model_name} L{num_layers}] {config_str}",
                    bar_format="{desc} |{bar}| {postfix}",
                    ncols=100,
                    position=1,
                )

                for run in range(n_runs):
                    seed = 42 + run
                    seed_everything(seed)
                    data_run = (
                        create_masks(data, seed=seed)
                        if not hasattr(data, "train_mask")
                        else data
                    )

                    try:
                        history, test_metrics, _ = train_single_config(
                            model_name=model_name,
                            dataset_name=dataset_name,
                            data=data_run,
                            hyperparams=hyperparams,
                            epochs=epochs,
                            device=device,
                            verbose=False,
                        )

                        model_results["accuracy"].append(test_metrics["accuracy"])
                        model_results["f1_score"].append(test_metrics["f1_score"])
                        model_results["histories"].append(history)

                        run_pbar.set_postfix_str(f"R{run+1}: Acc={test_metrics['accuracy']:.4f}")
                        run_pbar.update(1)

                    except Exception as e:
                        run_pbar.set_postfix_str(f"R{run+1}: Failed")
                        run_pbar.update(1)
                        if verbose:
                            print(f"  Run failed: {e}")

                run_pbar.close()

                all_results[dataset_name]["layers"][model_name][num_layers] = {
                    "accuracy_mean": float(np.mean(model_results["accuracy"])),
                    "accuracy_std": float(np.std(model_results["accuracy"])),
                    "f1_mean": float(np.mean(model_results["f1_score"])),
                    "f1_std": float(np.std(model_results["f1_score"])),
                    "runs": n_runs,
                    "histories": model_results["histories"],
                }

                acc = np.mean(model_results["accuracy"])
                std = np.std(model_results["accuracy"])
                print(f"  {model_name} L{num_layers}: Acc={acc:.4f} (+/- {std:.4f})")

    # Print summary table
    print("\n" + "=" * 80)
    print("[SUMMARY] Oversmoothing Analysis")
    print("=" * 80)
    header = f"{'Dataset':<12}"
    for model_name in models:
        for lc in layer_counts:
            header += f" {model_name[:6]}_L{lc}".ljust(12)
    print(header)
    print("-" * (14 * len(models) * len(layer_counts)))
    for dataset_name in datasets:
        row = f"{dataset_name.upper():<12}"
        for model_name in models:
            for lc in layer_counts:
                acc = all_results[dataset_name]["layers"][model_name][lc]["accuracy_mean"]
                row += f" {acc:.4f}".ljust(12)
        print(row)
    print("=" * 80)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"ablation_oversmoothing_{timestamp}.json")

    # Serialise without model objects
    serializable = {}
    for ds, ds_data in all_results.items():
        serializable[ds] = {}
        for k, v in ds_data.items():
            if k == "layers":
                serializable[ds]["layers"] = {}
                for m, m_data in v.items():
                    serializable[ds]["layers"][m] = {}
                    for lc, lc_data in m_data.items():
                        serializable[ds]["layers"][m][lc] = {
                            r: lc_data[r]
                            for r in lc_data
                            if r != "histories"
                        }
                        serializable[ds]["layers"][m][lc]["histories"] = [
                            {"epoch_times": h.get("epoch_times", [])}
                            for h in lc_data.get("histories", [])
                        ]
            else:
                serializable[ds][k] = v

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Generate plots
    if save_plots:
        print("\nGenerating oversmoothing plots...")
        from utils.training_time_plots import plot_oversmoothing_comparison
        plot_oversmoothing_comparison(
            all_results,
            layer_counts=layer_counts,
            models=models,
            save_dir=output_dir,
        )
        print(f"Plots saved to: {output_dir}")

    return all_results


# =============================================================================
# Argument Parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ablation Study: Oversmoothing Analysis (Layer Depth)"
    )
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["gcn", "residual_gcn", "gat", "residual_gat"],
        choices=["gcn", "gat", "sage", "appnp", "residual_gcn", "residual_gat", "residual_sage", "residual_appnp"],
        help="Model types to compare",
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
        "--layers",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Layer counts to compare",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of runs per config")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument(
        "--output-dir", type=str, default="./outputs", help="Output directory"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip generating plots"
    )
    return parser


# =============================================================================
# Main
# =============================================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    device = args.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    benchmark_layers(
        data_dir=args.data_dir,
        models=args.models,
        datasets=args.datasets,
        layer_counts=args.layers,
        epochs=args.epochs,
        n_runs=args.n_runs,
        device=device,
        output_dir=args.output_dir,
        verbose=args.verbose,
        save_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()

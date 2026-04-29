"""
Ablation Study: Residual vs Non-Residual Architectures.

Compares GCN, GAT, and SAGE with and without residual connections
across Amazon, DBLP, and Email datasets.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from tqdm import tqdm

# Import from train.py
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
from utils.training_time_plots import plot_residual_comparison


# =============================================================================
# Ablation: Residual vs Non-Residual
# =============================================================================

def benchmark_residual_vs_base(
    data_dir: str = "./data",
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    epochs: int = 100,
    n_runs: int = 3,
    device: Optional[str] = None,
    output_dir: str = "./outputs",
    verbose: bool = False,
    save_plots: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark residual vs non-residual versions of GNN models.

    Args:
        data_dir: Data root directory.
        models: List of base model types to compare (gcn, gat, sage).
        datasets: List of datasets to run on.
        epochs: Training epochs per run.
        n_runs: Number of runs per configuration.
        device: Device to train on.
        output_dir: Output directory.
        verbose: Print progress.
        save_plots: Whether to save comparison plots.

    Returns:
        Dictionary with all results.
    """
    # Map base model -> (residual_model, display_name)
    model_pairs = {
        "gcn": ("residual_gcn", "GCN", "GCN+Res"),
        "gat": ("residual_gat", "GAT", "GAT+Res"),
        "sage": ("residual_sage", "GraphSAGE", "GraphSAGE+Res"),
        "appnp": ("residual_appnp", "APPNP", "APPNP+Res"),
    }

    datasets = datasets or ["amazon", "dblp", "email"]
    models = models or ["gcn", "gat", "sage", "appnp"]
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
    print("ABLATION: Residual vs Non-Residual Architectures")
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

        all_results[dataset_name] = {"stats": stats, "comparison": {}}

        for base_model in models:
            if base_model not in model_pairs:
                continue

            residual_model, base_name, res_name = model_pairs[base_model]

            for model_name, display_name in [(base_model, base_name), (residual_model, res_name)]:
                model_results = {"accuracy": [], "f1_score": [], "configs": [], "histories": []}

                config_str = f"hid={default_hyperparams['hidden_channels']},layers={default_hyperparams['num_layers']},lr={default_hyperparams['lr']}"
                run_pbar = tqdm(
                    total=n_runs,
                    desc=f"  [{model_name}] {config_str}",
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
                            hyperparams=default_hyperparams,
                            epochs=epochs,
                            device=device,
                            verbose=False,
                        )

                        model_results["accuracy"].append(test_metrics["accuracy"])
                        model_results["f1_score"].append(test_metrics["f1_score"])
                        model_results["configs"].append(default_hyperparams)
                        model_results["histories"].append(history)

                        run_pbar.set_postfix_str(f"R{run+1}: Acc={test_metrics['accuracy']:.4f}")
                        run_pbar.update(1)

                    except Exception as e:
                        run_pbar.set_postfix_str(f"R{run+1}: Failed")
                        run_pbar.update(1)
                        if verbose:
                            print(f"  Run failed: {e}")

                run_pbar.close()

                # Store results
                all_results[dataset_name]["comparison"][model_name] = {
                    "display_name": display_name,
                    "accuracy_mean": float(np.mean(model_results["accuracy"])),
                    "accuracy_std": float(np.std(model_results["accuracy"])),
                    "f1_mean": float(np.mean(model_results["f1_score"])),
                    "f1_std": float(np.std(model_results["f1_score"])),
                    "runs": n_runs,
                    "histories": model_results["histories"],
                }

                print(
                    f"  {display_name}: Acc={np.mean(model_results['accuracy']):.4f} "
                    f"(+/- {np.std(model_results['accuracy']):.4f})"
                )

    # Print summary table
    print("\n" + "=" * 80)
    print("[SUMMARY] Residual vs Non-Residual")
    print("=" * 80)
    print(f"{'Dataset':<12} {'Base Model':<15} {'+Residual':<15} {'Improvement':<12}")
    print("-" * 60)
    for dataset_name in datasets:
        comparison = all_results[dataset_name]["comparison"]
        for base_model in models:
            if base_model not in model_pairs:
                continue
            residual_model = model_pairs[base_model][0]
            base_acc = comparison[base_model]["accuracy_mean"]
            res_acc = comparison[residual_model]["accuracy_mean"]
            diff = res_acc - base_acc
            sign = "+" if diff >= 0 else ""
            print(
                f"{dataset_name.upper():<12} {comparison[base_model]['display_name']:<15} "
                f"{comparison[residual_model]['display_name']:<15} {sign}{diff:.4f}"
            )
    print("=" * 80)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"ablation_residual_{timestamp}.json")
    with open(results_path, "w") as f:
        # Strip histories for JSON serialization
        serializable = {}
        for ds, ds_data in all_results.items():
            serializable[ds] = {k: v for k, v in ds_data.items() if k != "histories"}
            serializable[ds]["histories"] = {
                m: [{"epoch_times": h.get("epoch_times", [])} for h in runs]
                for m, runs in {m: all_results[ds].get("comparison", {}).get(m, {}).get("histories", []) for m in all_results[ds].get("comparison", {}).keys()}.items()
            }
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Generate plots
    if save_plots:
        print("\nGenerating comparison plots...")
        from utils.training_time_plots import plot_residual_comparison
        plot_residual_comparison(all_results, save_dir=output_dir)
        print(f"Plots saved to: {output_dir}")

    return all_results


# =============================================================================
# Argument Parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ablation Study: Residual vs Non-Residual Architectures"
    )
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["gcn", "gat", "sage", "appnp"],
        choices=["gcn", "gat", "sage", "appnp"],
        help="Base model types to compare",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["amazon", "dblp", "email"],
        choices=["amazon", "dblp", "email"],
        help="Datasets to run on",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of runs per config")
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/cpu)"
    )
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

    benchmark_residual_vs_base(
        data_dir=args.data_dir,
        models=args.models,
        datasets=args.datasets,
        epochs=args.epochs,
        n_runs=args.n_runs,
        device=device,
        output_dir=args.output_dir,
        verbose=args.verbose,
        save_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()

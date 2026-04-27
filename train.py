"""
Comprehensive training script with hyperparameter tuning and benchmarking.

Features:
    - Hyperparameter tuning with grid search or random search
    - Benchmarking across all datasets (Amazon, DBLP, Email-Eu-Core)
    - Support for all model architectures (GCN, GAT, SAGE, PPNP, APPNP)
    - Combined feature modes for Email-Eu-Core dataset
    - Learning curve visualization with matplotlib
    - JSON export for training history
    - Wandb integration for experiment tracking
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.seed import seed_everything

from gnn_datasets import AmazonPhotos, EmailEuCore, DBLP
from models import GCNWrapper, GATWrapper, SAGEWrapper, PPNPWrapper, APPNPWrapper
from utils import Trainer, ResidualGNNWrapper, ResidualAPPNPWrapper


# =============================================================================
# Hyperparameter Search Spaces
# =============================================================================

HYPERPARAM_GRID = {
    "gcn": {
        "hidden_channels": [64, 128, 256],
        "num_layers": [2, 3, 4],
        "dropout": [0.3, 0.5, 0.7],
        "norm": ["layer", "batch", "none"],
        "lr": [0.01, 0.005, 0.001],
        "weight_decay": [5e-4, 1e-4, 5e-5],
    },
    "gat": {
        "hidden_channels": [64, 128, 256],
        "num_layers": [2, 3, 4],
        "dropout": [0.3, 0.5, 0.7],
        "norm": ["layer", "batch", "none"],
        "lr": [0.01, 0.005, 0.001],
        "weight_decay": [5e-4, 1e-4, 5e-5],
        "gat_heads": [4, 8],
    },
    "sage": {
        "hidden_channels": [64, 128, 256],
        "num_layers": [2, 3, 4],
        "dropout": [0.3, 0.5, 0.7],
        "norm": ["layer", "batch", "none"],
        "lr": [0.01, 0.005, 0.001],
        "weight_decay": [5e-4, 1e-4, 5e-5],
    },
    "ppnp": {
        "hidden_channels": [64, 128, 256],
        "num_layers": [2, 3, 4],
        "dropout": [0.3, 0.5, 0.7],
        "norm": ["layer", "batch", "none"],
        "lr": [0.01, 0.005, 0.001],
        "weight_decay": [5e-4, 1e-4, 5e-5],
        "alpha": [0.1, 0.15, 0.2],
    },
    "appnp": {
        "hidden_channels": [64, 128, 256],
        "num_layers": [2, 3, 4],
        "dropout": [0.3, 0.5, 0.7],
        "norm": ["layer", "batch", "none"],
        "lr": [0.01, 0.005, 0.001],
        "weight_decay": [5e-4, 1e-4, 5e-5],
        "K": [5, 10, 20],
        "alpha": [0.1, 0.15, 0.2],
    },
}

HYPERPARAM_RANDOM_DIST = {
    "hidden_channels": lambda: random.choice([32, 64, 128, 256, 512]),
    "num_layers": lambda: random.choice([2, 3, 4, 5, 6]),
    "dropout": lambda: random.uniform(0.1, 0.8),
    "norm": lambda: random.choice(["layer", "batch", "graph", "none"]),
    "lr": lambda: 10 ** random.uniform(-4, -2),
    "weight_decay": lambda: 10 ** random.uniform(-5, -3),
    "gat_heads": lambda: random.choice([1, 2, 4, 8]),
    "K": lambda: random.choice([5, 10, 15, 20, 30]),
    "alpha": lambda: random.uniform(0.05, 0.3),
}


# =============================================================================
# Dataset Utilities
# =============================================================================

def load_dataset(
    dataset_name: str,
    data_dir: str = "./data",
    email_features: Optional[Dict[str, bool]] = None,
) -> Tuple[Any, Data]:
    """
    Load a dataset and return the dataset object and data.

    Args:
        dataset_name: Name of the dataset ("amazon", "dblp", "email").
        data_dir: Root directory for data.
        email_features: Dictionary of feature flags for Email-Eu-Core.

    Returns:
        Tuple of (dataset, data).
    """
    if dataset_name == "amazon":
        dataset = AmazonPhotos(root=os.path.join(data_dir, "AmazonPhotos"))
        data = dataset[0]
    elif dataset_name == "dblp":
        dataset = DBLP(root=os.path.join(data_dir, "DBLP"))
        data = dataset.get_homograph_apa()
        # Flatten 2D masks to 1D
        for mask_name in ("train_mask", "val_mask", "test_mask"):
            mask = getattr(data, mask_name, None)
            if mask is not None and mask.dim() == 2:
                setattr(data, mask_name, mask[:, 0])
    elif dataset_name == "email":
        email_features = email_features or {}
        dataset = EmailEuCore(
            root=os.path.join(data_dir, "EmailEuCore"),
            use_degree=email_features.get("use_degree", False),
            use_centrality=email_features.get("use_centrality", False),
            use_local=email_features.get("use_local", False),
            use_original=email_features.get("use_original", True),
        )
        data = dataset[0]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset, data


def create_masks(data: Data, train_ratio: float = 0.6, val_ratio: float = 0.2,
                 test_ratio: float = 0.2, seed: int = 42) -> Data:
    """Create train/val/test masks if they don't exist."""
    if hasattr(data, 'train_mask') and data.train_mask is not None:
        return data

    num_nodes = data.num_nodes
    seed_everything(seed)
    perm = torch.randperm(num_nodes)

    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[perm[:train_size]] = True
    data.val_mask[perm[train_size:train_size + val_size]] = True
    data.test_mask[perm[train_size + val_size:]] = True

    return data


def get_dataset_stats(data: Data) -> Dict[str, Any]:
    """Get statistics about a dataset."""
    stats = {
        "num_nodes": data.num_nodes,
        "num_edges": data.edge_index.shape[1],
        "num_features": data.x.shape[1] if data.x is not None else 0,
        "num_classes": int(data.y.max().item() + 1) if data.y is not None else 0,
        "avg_degree": float(data.edge_index.shape[1] / data.num_nodes) if data.num_nodes > 0 else 0,
    }
    return stats


# =============================================================================
# Model Creation
# =============================================================================

def create_model(
    model_name: str,
    in_channels: int,
    out_channels: int,
    **hyperparams,
) -> nn.Module:
    """
    Create a model instance with given hyperparameters.

    Args:
        model_name: Name of the model architecture.
        in_channels: Input feature dimension.
        out_channels: Output dimension (num classes).
        **hyperparams: Model hyperparameters.

    Returns:
        The instantiated model.
    """
    common_kwargs = {
        "in_channels": in_channels,
        "hidden_channels": hyperparams.get("hidden_channels", 128),
        "num_layers": hyperparams.get("num_layers", 2),
        "out_channels": out_channels,
        "dropout": hyperparams.get("dropout", 0.5),
        "norm": hyperparams.get("norm", "layer"),
    }

    if model_name == "gcn":
        return GCNWrapper(**common_kwargs)
    elif model_name == "gat":
        gat_kwargs = {
            "heads": hyperparams.get("gat_heads", 4),
            "concat": False if common_kwargs["num_layers"] > 1 else True,
        }
        return GATWrapper(**common_kwargs, **gat_kwargs)
    elif model_name == "sage":
        return SAGEWrapper(**common_kwargs)
    elif model_name == "ppnp":
        ppnp_kwargs = {
            "alpha": hyperparams.get("alpha", 0.1),
        }
        return PPNPWrapper(**common_kwargs, **ppnp_kwargs)
    elif model_name == "appnp":
        appnp_kwargs = {
            "K": hyperparams.get("K", 10),
            "alpha": hyperparams.get("alpha", 0.1),
        }
        return APPNPWrapper(**common_kwargs, **appnp_kwargs)
    elif model_name == "residual_gcn":
        return ResidualGNNWrapper(model_type="gcn", **common_kwargs)
    elif model_name == "residual_gat":
        gat_kwargs = {"heads": hyperparams.get("gat_heads", 4)}
        return ResidualGNNWrapper(model_type="gat", **common_kwargs, **gat_kwargs)
    elif model_name == "residual_sage":
        return ResidualGNNWrapper(model_type="sage", **common_kwargs)
    elif model_name == "residual_appnp":
        return ResidualAPPNPWrapper(
            **common_kwargs,
            K=hyperparams.get("K", 10),
            alpha=hyperparams.get("alpha", 0.1),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# Training Functions
# =============================================================================

def train_single_config(
    model_name: str,
    dataset_name: str,
    data: Data,
    hyperparams: Dict[str, Any],
    epochs: int = 100,
    device: Optional[str] = None,
    verbose: bool = False,
    wandb_enabled: bool = False,
    wandb_kwargs: Optional[Dict] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, float], nn.Module]:
    """
    Train a single model configuration.

    Returns:
        Tuple of (history, test_metrics, model).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = data.x.shape[1] if data.x is not None else data.num_nodes
    out_channels = int(data.y.max().item() + 1)

    model = create_model(model_name, in_channels, out_channels, **hyperparams)
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams.get("lr", 0.01),
        weight_decay=hyperparams.get("weight_decay", 5e-4),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        use_wandb=wandb_enabled,
        wandb_kwargs=wandb_kwargs,
        scheduler=scheduler,
        early_stopping_patience=hyperparams.get("early_stopping", 0),
        gradient_clip_val=hyperparams.get("gradient_clip", 1.0),
    )

    history = trainer.train(
        data=data,
        epochs=epochs,
        val_every=1,
        print_every=epochs if not verbose else 10,
    )

    # Final evaluation
    results = trainer.predict(data, "test_mask")
    test_metrics = {
        "accuracy": results.get("accuracy", 0.0),
        "f1_score": results.get("f1_score", 0.0),
    }

    return history, test_metrics, model


def hyperparameter_search(
    model_name: str,
    dataset_name: str,
    data: Data,
    search_type: str = "grid",
    n_trials: int = 20,
    epochs: int = 100,
    val_metric: str = "accuracy",
    device: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, List[float]]]:
    """
    Perform hyperparameter search.

    Args:
        model_name: Model architecture name.
        dataset_name: Dataset name.
        data: PyG Data object.
        search_type: "grid" or "random".
        n_trials: Number of trials (for random search).
        epochs: Training epochs per trial.
        val_metric: Metric to optimize ("accuracy" or "loss").
        device: Device to train on.
        verbose: Print progress.

    Returns:
        Tuple of (best_hyperparams, best_test_metrics, all_trial_results).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param_grid = HYPERPARAM_GRID.get(model_name, HYPERPARAM_GRID["gcn"])

    if search_type == "grid":
        # Generate all combinations
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        all_configs = [dict(zip(keys, v)) for v in product(*values)]
        if verbose:
            print(f"Grid search: {len(all_configs)} configurations")
    else:
        # Random search
        all_configs = []
        for _ in range(n_trials):
            config = {k: v() for k, v in HYPERPARAM_RANDOM_DIST.items() if k in param_grid or k in HYPERPARAM_RANDOM_DIST}
            # Add model-specific params
            if model_name == "ppnp" and "alpha" not in config:
                config["alpha"] = random.uniform(0.05, 0.3)
            if model_name == "appnp":
                if "K" not in config:
                    config["K"] = random.choice([5, 10, 15, 20])
                if "alpha" not in config:
                    config["alpha"] = random.uniform(0.05, 0.3)
            all_configs.append(config)
        if verbose:
            print(f"Random search: {n_trials} trials")

    best_val_score = -float("inf") if val_metric == "accuracy" else float("inf")
    best_config = None
    best_test_metrics = None
    all_results = []

    for i, config in enumerate(all_configs):
        if verbose:
            print(f"\nTrial {i+1}/{len(all_configs)}: {config}")

        try:
            _, val_metrics, _ = train_single_config(
                model_name=model_name,
                dataset_name=dataset_name,
                data=data,
                hyperparams=config,
                epochs=epochs,
                device=device,
                verbose=False,
            )

            val_score = val_metrics.get(val_metric, 0)
            is_better = (val_score > best_val_score) if val_metric == "accuracy" else (val_score < best_val_score)

            result = {
                "trial": i + 1,
                "config": config,
                "val_score": val_score,
                "val_metrics": val_metrics,
            }
            all_results.append(result)

            if is_better:
                best_val_score = val_score
                best_config = config
                # Retrain with best config to get test metrics
                _, best_test_metrics, _ = train_single_config(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    data=data,
                    hyperparams=config,
                    epochs=epochs,
                    device=device,
                    verbose=False,
                )
                if verbose:
                    print(f"  -> New best! Val {val_metric}: {val_score:.4f}")

        except Exception as e:
            if verbose:
                print(f"  -> Failed: {e}")
            continue

    return best_config, best_test_metrics, {"trials": all_results}


# =============================================================================
# Benchmarking Functions
# =============================================================================

def benchmark_all_models(
    dataset_name: str,
    data: Data,
    models: Optional[List[str]] = None,
    hyperparams: Optional[Dict[str, Dict]] = None,
    epochs: int = 100,
    n_runs: int = 5,
    device: Optional[str] = None,
    output_dir: str = "./outputs",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Benchmark all specified models on a dataset.

    Args:
        dataset_name: Name of the dataset.
        data: PyG Data object.
        models: List of model names to benchmark.
        hyperparams: Fixed hyperparameters for each model.
        epochs: Training epochs.
        n_runs: Number of runs per model (for statistical significance).
        device: Device to train on.
        output_dir: Directory for saving results.
        verbose: Print progress.

    Returns:
        Dictionary with benchmark results.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = models or ["gcn", "gat", "sage", "ppnp", "appnp"]

    # Default hyperparameters if not specified
    default_hyperparams = {
        "hidden_channels": 128,
        "num_layers": 2,
        "dropout": 0.5,
        "norm": "layer",
        "lr": 0.01,
        "weight_decay": 5e-4,
    }

    all_results = {}

    for model_name in models:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Benchmarking {model_name.upper()} on {dataset_name}")
            print(f"{'='*60}")

        model_results = {
            "accuracy": [],
            "f1_score": [],
            "configs": [],
        }

        for run in range(n_runs):
            if verbose:
                print(f"\nRun {run+1}/{n_runs}")

            # Add slight seed variation for each run
            seed = 42 + run
            seed_everything(seed)

            # Create new data copy with different mask split
            data_run = create_masks(data, seed=seed) if not hasattr(data, 'train_mask') else data

            hyperparams_for_run = {**default_hyperparams}
            if hyperparams and model_name in hyperparams:
                hyperparams_for_run.update(hyperparams[model_name])

            try:
                _, test_metrics, _ = train_single_config(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    data=data_run,
                    hyperparams=hyperparams_for_run,
                    epochs=epochs,
                    device=device,
                    verbose=False,
                )

                model_results["accuracy"].append(test_metrics["accuracy"])
                model_results["f1_score"].append(test_metrics["f1_score"])
                model_results["configs"].append(hyperparams_for_run)

                if verbose:
                    print(f"  Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_score']:.4f}")

            except Exception as e:
                if verbose:
                    print(f"  Run failed: {e}")
                continue

        # Compute statistics
        if model_results["accuracy"]:
            all_results[model_name] = {
                "accuracy_mean": float(np.mean(model_results["accuracy"])),
                "accuracy_std": float(np.std(model_results["accuracy"])),
                "f1_mean": float(np.mean(model_results["f1_score"])),
                "f1_std": float(np.std(model_results["f1_score"])),
                "runs": n_runs,
                "individual_runs": model_results,
            }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"benchmark_{dataset_name}_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    if verbose:
        print(f"\nBenchmark results saved to: {results_path}")
        print("\nSummary:")
        for model_name, results in all_results.items():
            print(f"  {model_name}: Acc={results['accuracy_mean']:.4f} (+/- {results['accuracy_std']:.4f})")

    return all_results


def benchmark_email_feature_combinations(
    data_dir: str = "./data",
    models: Optional[List[str]] = None,
    epochs: int = 100,
    n_runs: int = 3,
    device: Optional[str] = None,
    output_dir: str = "./outputs",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Benchmark Email-Eu-Core with all feature combinations.

    Args:
        data_dir: Data root directory.
        models: List of models to benchmark.
        epochs: Training epochs.
        n_runs: Number of runs per configuration.
        device: Device to train on.
        output_dir: Output directory.
        verbose: Print progress.

    Returns:
        Dictionary with all benchmark results.
    """
    # All feature combinations
    feature_combos = [
        {"use_original": True},  # Baseline
        {"use_original": True, "use_degree": True},
        {"use_original": True, "use_centrality": True},
        {"use_original": True, "use_local": True},
        {"use_original": True, "use_degree": True, "use_centrality": True},
        {"use_original": True, "use_degree": True, "use_local": True},
        {"use_original": True, "use_centrality": True, "use_local": True},
        {"use_original": True, "use_degree": True, "use_centrality": True, "use_local": True},
    ]

    models = models or ["gcn", "gat", "sage"]
    all_results = {}

    for combo in feature_combos:
        combo_name = "_".join([k.replace("use_", "") for k, v in combo.items() if v])
        if verbose:
            print(f"\n{'='*60}")
            print(f"Feature combination: {combo_name}")
            print(f"{'='*60}")

        # Load dataset with specific feature combination
        _, data = load_dataset("email", data_dir, email_features=combo)
        data = create_masks(data) if not hasattr(data, 'train_mask') else data

        # Benchmark all models with this feature combo
        combo_results = benchmark_all_models(
            dataset_name=f"email_{combo_name}",
            data=data,
            models=models,
            epochs=epochs,
            n_runs=n_runs,
            device=device,
            output_dir=output_dir,
            verbose=verbose,
        )

        all_results[combo_name] = combo_results

    # Save aggregated results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"email_features_benchmark_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    if verbose:
        print(f"\nAll results saved to: {results_path}")

    return all_results


def benchmark_all_datasets(
    data_dir: str = "./data",
    models: Optional[List[str]] = None,
    epochs: int = 100,
    n_runs: int = 3,
    device: Optional[str] = None,
    output_dir: str = "./outputs",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Benchmark all models across all datasets.

    Args:
        data_dir: Data root directory.
        models: List of models to benchmark.
        epochs: Training epochs.
        n_runs: Number of runs per configuration.
        device: Device to train on.
        output_dir: Output directory.
        verbose: Print progress.

    Returns:
        Dictionary with all benchmark results.
    """
    datasets = ["amazon", "dblp", "email"]
    all_results = {}

    for dataset_name in datasets:
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# BENCHMARKING {dataset_name.upper()}")
            print(f"{'#'*60}")

        # Load dataset
        _, data = load_dataset(dataset_name, data_dir)
        data = create_masks(data) if not hasattr(data, 'train_mask') else data

        # Get dataset statistics
        stats = get_dataset_stats(data)
        if verbose:
            print(f"\nDataset stats: {stats}")

        # Benchmark
        dataset_results = benchmark_all_models(
            dataset_name=dataset_name,
            data=data,
            models=models,
            epochs=epochs,
            n_runs=n_runs,
            device=device,
            output_dir=output_dir,
            verbose=verbose,
        )

        all_results[dataset_name] = {
            "stats": stats,
            "benchmark": dataset_results,
        }

    # Save aggregated results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"all_datasets_benchmark_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    if verbose:
        print(f"\nAll results saved to: {results_path}")

    return all_results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GNN Training with Hyperparameter Tuning & Benchmarking")

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "tune", "benchmark", "benchmark_all", "benchmark_email_features"],
        default="train",
        help="Operation mode",
    )

    # Dataset options
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["amazon", "dblp", "email"],
        default="email",
        help="Dataset to use",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Data directory",
    )

    # Email feature flags
    parser.add_argument("--email-use-degree", action="store_true", default=False)
    parser.add_argument("--email-use-centrality", action="store_true", default=False)
    parser.add_argument("--email-use-local", action="store_true", default=False)
    parser.add_argument("--email-use-original", action="store_true", default=True)

    # Model options
    parser.add_argument(
        "--model",
        type=str,
        choices=["gcn", "gat", "sage", "ppnp", "appnp", "residual_gcn", "residual_gat", "residual_sage", "residual_appnp"],
        default="gcn",
        help="Model architecture",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="List of models for benchmarking",
    )

    # Hyperparameters
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--norm", type=str, choices=["layer", "batch", "graph", "none"], default="layer")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--K", type=int, default=10, help="APPNP propagation steps")
    parser.add_argument("--alpha", type=float, default=0.1, help="PPNP/APPNP teleport probability")

    # Training options
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n-runs", type=int, default=3, help="Number of runs for benchmarking")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    # Hyperparameter search
    parser.add_argument(
        "--search-type",
        type=str,
        choices=["grid", "random"],
        default="random",
        help="Hyperparameter search type",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of trials for hyperparameter search",
    )
    parser.add_argument(
        "--val-metric",
        type=str,
        choices=["accuracy", "loss"],
        default="accuracy",
        help="Metric to optimize during hyperparameter search",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for results",
    )
    parser.add_argument("--save-history-json", action="store_true", default=False)
    parser.add_argument("--save-learning-curves", action="store_true", default=False)

    # Logging
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default="gnn-benchmark")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args()

    # Setup device
    device = args.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare email features
    email_features = {
        "use_degree": args.email_use_degree,
        "use_centrality": args.email_use_centrality,
        "use_local": args.email_use_local,
        "use_original": args.email_use_original,
    }

    # Prepare hyperparameters
    hyperparams = {
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "norm": args.norm,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "gat_heads": args.gat_heads,
        "K": args.K,
        "alpha": args.alpha,
    }

    # Wandb setup
    wandb_kwargs = None
    if args.wandb:
        wandb_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_name or f"{args.model}_{args.dataset}",
            "config": vars(args),
        }

    # Execute based on mode
    if args.mode == "train":
        # Single training run
        print(f"\nTraining {args.model} on {args.dataset}")

        _, data = load_dataset(args.dataset, args.data_dir, email_features)
        data = create_masks(data) if not hasattr(data, 'train_mask') else data

        history, test_metrics, model = train_single_config(
            model_name=args.model,
            dataset_name=args.dataset,
            data=data,
            hyperparams=hyperparams,
            epochs=args.epochs,
            device=device,
            verbose=args.verbose,
            wandb_enabled=args.wandb,
            wandb_kwargs=wandb_kwargs,
        )

        print(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")

        # Save outputs
        if args.save_history_json or args.save_learning_curves:
            os.makedirs(args.output_dir, exist_ok=True)
            base_name = f"{args.model}_{args.dataset}"

            if args.save_history_json:
                json_path = os.path.join(args.output_dir, f"{base_name}_history.json")
                # Create a temporary trainer to use its save method
                from utils import Trainer
                import torch.nn as nn
                temp_trainer = Trainer(model, torch.optim.Adam(model.parameters()))
                temp_trainer.history = history
                temp_trainer.save_history_to_json(json_path)

            if args.save_learning_curves:
                plot_path = os.path.join(args.output_dir, f"{base_name}_curves.png")
                from utils import Trainer
                import torch.nn as nn
                temp_trainer = Trainer(model, torch.optim.Adam(model.parameters()))
                temp_trainer.history = history
                temp_trainer.plot_learning_curves(save_path=plot_path)

    elif args.mode == "tune":
        # Hyperparameter search
        print(f"\nHyperparameter search for {args.model} on {args.dataset}")
        print(f"Search type: {args.search_type}, Trials: {args.n_trials}")

        _, data = load_dataset(args.dataset, args.data_dir, email_features)
        data = create_masks(data) if not hasattr(data, 'train_mask') else data

        best_config, best_metrics, all_results = hyperparameter_search(
            model_name=args.model,
            dataset_name=args.dataset,
            data=data,
            search_type=args.search_type,
            n_trials=args.n_trials,
            epochs=args.epochs,
            val_metric=args.val_metric,
            device=device,
            verbose=args.verbose,
        )

        print(f"\nBest configuration: {best_config}")
        print(f"Best test metrics: {best_metrics}")

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(args.output_dir, f"tuning_{args.model}_{args.dataset}_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump({
                "best_config": best_config,
                "best_metrics": best_metrics,
                "all_trials": all_results,
            }, f, indent=2)
        print(f"Results saved to: {results_path}")

    elif args.mode == "benchmark":
        # Benchmark specific models on specific dataset
        print(f"\nBenchmarking on {args.dataset}")
        models = args.models or ["gcn", "gat", "sage", "ppnp", "appnp"]

        _, data = load_dataset(args.dataset, args.data_dir, email_features)
        data = create_masks(data) if not hasattr(data, 'train_mask') else data

        results = benchmark_all_models(
            dataset_name=args.dataset,
            data=data,
            models=models,
            epochs=args.epochs,
            n_runs=args.n_runs,
            device=device,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )

        print("\nBenchmark Summary:")
        for model_name, model_results in results.items():
            print(f"  {model_name}: Acc={model_results['accuracy_mean']:.4f} (+/- {model_results['accuracy_std']:.4f})")

    elif args.mode == "benchmark_email_features":
        # Benchmark all feature combinations on Email-Eu-Core
        print("\nBenchmarking Email-Eu-Core with all feature combinations")
        models = args.models or ["gcn", "gat", "sage"]

        results = benchmark_email_feature_combinations(
            data_dir=args.data_dir,
            models=models,
            epochs=args.epochs,
            n_runs=args.n_runs,
            device=device,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )

    elif args.mode == "benchmark_all":
        # Full benchmark across all datasets
        print("\nFull benchmark across all datasets")
        models = args.models or ["gcn", "gat", "sage", "ppnp", "appnp"]

        results = benchmark_all_datasets(
            data_dir=args.data_dir,
            models=models,
            epochs=args.epochs,
            n_runs=args.n_runs,
            device=device,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )

        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        for dataset_name, dataset_results in results.items():
            print(f"\n{dataset_name.upper()}:")
            benchmark = dataset_results.get("benchmark", dataset_results)
            for model_name, model_results in benchmark.items():
                if isinstance(model_results, dict) and "accuracy_mean" in model_results:
                    print(f"  {model_name}: Acc={model_results['accuracy_mean']:.4f} (+/- {model_results['accuracy_std']:.4f})")


if __name__ == "__main__":
    main()

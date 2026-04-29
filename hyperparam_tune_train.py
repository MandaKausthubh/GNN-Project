"""
Hyperparameter Tuning Training Script with Multi-Dataset and Multi-Model Support.

Features:
    - Bayesian optimization for hyperparameter tuning
    - Support for multiple datasets and models via argparse
    - Configurable number of tuning iterations
    - Generates comparison plots:
        (1) Train + Test Accuracy
        (2) Train + Test Loss (Validation)
        (3) F1 Score
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from gnn_datasets import AmazonPhotos, EmailEuCore, DBLP
from models import GCNWrapper, GATWrapper, SAGEWrapper, PPNPWrapper, APPNPWrapper
from utils import Trainer, ResidualGNNWrapper, ResidualAPPNPWrapper
from utils.bayes_hp import (
    get_bayesian_optimizer_for_model,
    convert_bayesian_params_to_trainable,
)


# =============================================================================
# Dataset Utilities
# =============================================================================

def load_dataset(
    dataset_name: str,
    data_dir: str = "./data",
    email_features: Optional[Dict[str, bool]] = None,
) -> Tuple[Any, Data]:
    """Load a dataset and return the dataset object and data."""
    if dataset_name == "amazon":
        dataset = AmazonPhotos(root=os.path.join(data_dir, "AmazonPhotos"))
        data = dataset[0]
    elif dataset_name == "dblp":
        dataset = DBLP(root=os.path.join(data_dir, "DBLP"))
        data = dataset.get_homograph_apa()
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


def create_masks(
    data: Data,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Data:
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


# =============================================================================
# Model Creation
# =============================================================================

def create_model(
    model_name: str,
    in_channels: int,
    out_channels: int,
    **hyperparams,
) -> nn.Module:
    """Create a model instance with given hyperparameters."""
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
        ppnp_kwargs = {"alpha": hyperparams.get("alpha", 0.1)}
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
# Training Function
# =============================================================================

def train_single_config(
    model_name: str,
    dataset_name: str,
    data: Data,
    hyperparams: Dict[str, Any],
    epochs: int = 100,
    device: Optional[str] = None,
    verbose: bool = False,
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

    if verbose:
        print(f"Training {model_name.upper()} with hyperparameters: {hyperparams}")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        scheduler=scheduler,
        early_stopping_patience=hyperparams.get("early_stopping", 0),
        gradient_clip_val=hyperparams.get("gradient_clip", 1.0),
    )

    history = trainer.train(
        data=data,
        epochs=epochs,
        val_every=1,
        print_every=epochs if not verbose else 10,
        use_tqdm=verbose,
    )

    # Final evaluation
    results = trainer.predict(data, "test_mask")
    test_metrics = {
        "accuracy": results.get("accuracy", 0.0),
        "f1_score": results.get("f1_score", 0.0),
    }

    return history, test_metrics, model


# =============================================================================
# Hyperparameter Tuning
# =============================================================================

def hyperparameter_tuning(
    model_name: str,
    dataset_name: str,
    data: Data,
    n_iterations: int = 20,
    epochs: int = 100,
    device: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, List[float]], List[Dict]]:
    """
    Perform Bayesian hyperparameter tuning.

    Args:
        model_name: Model architecture name.
        dataset_name: Dataset name.
        data: PyG Data object.
        n_iterations: Number of Bayesian optimization iterations.
        epochs: Training epochs per trial.
        device: Device to train on.
        verbose: Print progress.

    Returns:
        Tuple of (best_hyperparams, best_test_metrics, best_history, all_trial_results).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Bayesian optimizer
    bayes_opt = get_bayesian_optimizer_for_model(model_name)

    best_val_score = -float("inf")
    best_config = None
    best_test_metrics = None
    best_history = None
    all_results = []

    pbar = tqdm(total=n_iterations, desc=f"Tuning {model_name}", disable=not verbose)

    for i in range(n_iterations):
        # Get next suggestion from Bayesian optimizer
        raw_config = bayes_opt.suggest_next()
        config = convert_bayesian_params_to_trainable(raw_config)

        # Add model-specific params if needed
        if model_name == "ppnp" and "alpha" not in config:
            config["alpha"] = 0.1
        if model_name == "appnp":
            if "K" not in config:
                config["K"] = 10
            if "alpha" not in config:
                config["alpha"] = 0.1

        config_str = f"hid={config.get('hidden_channels', '?')},layers={config.get('num_layers', '?')},lr={config.get('lr', '?')}"

        try:
            history, val_metrics, _ = train_single_config(
                model_name=model_name,
                dataset_name=dataset_name,
                data=data,
                hyperparams=config,
                epochs=epochs,
                device=device,
                verbose=False,
            )

            val_score = val_metrics.get("accuracy", 0)
            result = {
                "trial": i + 1,
                "config": config,
                "val_score": val_score,
                "val_metrics": val_metrics,
            }
            all_results.append(result)

            # Observe the result
            bayes_opt.observe(raw_config, val_score)

            if val_score > best_val_score:
                best_val_score = val_score
                best_config = config
                best_test_metrics = val_metrics
                best_history = history
                pbar.set_postfix_str(f"Trial {i+1}: {config_str} | Val Acc={val_score:.4f} (BEST)")
            else:
                pbar.set_postfix_str(f"Trial {i+1}: {config_str} | Val Acc={val_score:.4f}")

        except Exception as e:
            pbar.set_postfix_str(f"Trial {i+1}: {config_str} | FAILED: {str(e)[:30]}")
            if verbose:
                print(f"  -> Failed: {e}")
            continue

        pbar.update(1)

    pbar.close()

    # Print final results summary
    print("\n" + "=" * 60)
    print(f"[TUNING COMPLETE - {model_name} on {dataset_name}]")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_val_score:.4f}")
    print(f"Best Config: {best_config}")
    if best_test_metrics:
        print(f"Test Accuracy: {best_test_metrics.get('accuracy', 0):.4f}")
        print(f"Test F1 Score: {best_test_metrics.get('f1_score', 0):.4f}")
    print("=" * 60)

    return best_config, best_test_metrics, best_history, all_results


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_comparison(
    all_results: Dict[str, Dict],
    output_dir: str,
    timestamp: str,
    show: bool = False,
) -> Dict[str, str]:
    """
    Generate comparison plots for all dataset-model combinations.
    Creates separate plots per dataset with validation vs testing comparisons.

    Args:
        all_results: Dictionary with structure {dataset: {model: {metrics}}}
        output_dir: Directory to save plots.
        timestamp: Timestamp string for filenames.
        show: Whether to display plots.

    Returns:
        Dictionary of saved plot paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = {}

    # Collect all data for plotting
    datasets = list(all_results.keys())
    models = set()
    for ds_data in all_results.values():
        models.update(ds_data.keys())
    models = sorted(list(models))

    # ========================================================================
    # Create one plot per dataset with validation vs testing metrics
    # ========================================================================
    for dataset in datasets:
        models_data = all_results.get(dataset, {})

        # Count models that have history for this dataset
        models_with_history = [m for m in models if m in models_data and "history" in models_data[m]]

        if not models_with_history:
            continue

        # ---- Accuracy: Validation vs Testing ----
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Bar chart - Val vs Test Accuracy per model
        val_accs = []
        test_accs = []
        model_labels = []

        for model in models_with_history:
            metrics = models_data[model]
            history = metrics.get("history") or {}

            # Get final validation accuracy from history
            val_acc_list = history.get("val_acc")
            val_acc = val_acc_list[-1] if val_acc_list else 0
            test_acc = metrics.get("test_accuracy", 0)

            val_accs.append(val_acc)
            test_accs.append(test_acc)
            model_labels.append(model)

        x = np.arange(len(model_labels))
        width = 0.35

        bars1 = axes[0].bar(x - width/2, val_accs, width, label="Validation Acc", color="#2196F3", alpha=0.8)
        bars2 = axes[0].bar(x + width/2, test_accs, width, label="Test Acc", color="#FF5722", alpha=0.8)

        for bar, val, test in zip(bars1, val_accs, test_accs):
            axes[0].annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for bar, val, test in zip(bars2, val_accs, test_accs):
            axes[0].annotate(f"{test:.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        axes[0].set_xlabel("Model", fontsize=12)
        axes[0].set_ylabel("Accuracy", fontsize=12)
        axes[0].set_title(f"{dataset}: Validation vs Test Accuracy", fontsize=14)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_labels, rotation=45, ha="right")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].set_ylim(0, 1.1)

        # Right: Training curves - Val Accuracy over epochs
        for model in models_with_history:
            history = models_data[model].get("history", {})
            train_acc = history.get("train_acc", [])
            val_acc = history.get("val_acc", [])

            if train_acc:
                axes[1].plot(range(1, len(train_acc) + 1), train_acc, linewidth=2, label=f"{model} (train)")
            if val_acc:
                val_indices = np.linspace(0, len(train_acc) - 1, len(val_acc)).astype(int)
                axes[1].plot(range(1, len(train_acc) + 1),
                           np.interp(range(len(train_acc)), val_indices, val_acc),
                           linewidth=2, linestyle="--", label=f"{model} (val)")

        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Accuracy", fontsize=12)
        axes[1].set_title(f"{dataset}: Accuracy Training Curves", fontsize=14)
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        acc_path = os.path.join(output_dir, f"accuracy_{dataset}_{timestamp}.png")
        plt.savefig(acc_path, dpi=150, bbox_inches="tight")
        saved_plots[f"accuracy_{dataset}"] = acc_path
        print(f"Accuracy plot saved to: {acc_path}")
        if show:
            plt.show()
        plt.close()

        # ---- F1 Score: Validation vs Testing ----
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        val_f1s = []
        test_f1s = []

        for model in models_with_history:
            metrics = models_data[model]
            history = metrics.get("history") or {}

            val_f1_list = history.get("val_f1")
            val_f1 = val_f1_list[-1] if val_f1_list else 0
            test_f1 = metrics.get("test_f1_score", 0)

            val_f1s.append(val_f1)
            test_f1s.append(test_f1)

        bars1 = axes[0].bar(x - width/2, val_f1s, width, label="Validation F1", color="#4CAF50", alpha=0.8)
        bars2 = axes[0].bar(x + width/2, test_f1s, width, label="Test F1", color="#9C27B0", alpha=0.8)

        for bar, val, test in zip(bars1, val_f1s, test_f1s):
            axes[0].annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for bar, val, test in zip(bars2, val_f1s, test_f1s):
            axes[0].annotate(f"{test:.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        axes[0].set_xlabel("Model", fontsize=12)
        axes[0].set_ylabel("F1 Score", fontsize=12)
        axes[0].set_title(f"{dataset}: Validation vs Test F1 Score", fontsize=14)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_labels, rotation=45, ha="right")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].set_ylim(0, 1.1)

        # Right: F1 Training curves
        for model in models_with_history:
            history = models_data[model].get("history", {})
            train_f1 = history.get("train_f1", [])
            val_f1 = history.get("val_f1", [])

            if train_f1:
                axes[1].plot(range(1, len(train_f1) + 1), train_f1, linewidth=2, label=f"{model} (train)")
            if val_f1:
                val_indices = np.linspace(0, len(train_f1) - 1, len(val_f1)).astype(int)
                axes[1].plot(range(1, len(train_f1) + 1),
                           np.interp(range(len(train_f1)), val_indices, val_f1),
                           linewidth=2, linestyle="--", label=f"{model} (val)")

        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("F1 Score", fontsize=12)
        axes[1].set_title(f"{dataset}: F1 Score Training Curves", fontsize=14)
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        f1_path = os.path.join(output_dir, f"f1score_{dataset}_{timestamp}.png")
        plt.savefig(f1_path, dpi=150, bbox_inches="tight")
        saved_plots[f"f1score_{dataset}"] = f1_path
        print(f"F1 score plot saved to: {f1_path}")
        if show:
            plt.show()
        plt.close()

    # ========================================================================
    # Cross-dataset summary: Test Accuracy and F1 only
    # ========================================================================
    if len(datasets) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        n_datasets = len(datasets)
        n_models = len(models)
        x = np.arange(n_datasets)
        width = 0.8 / n_models

        # Test Accuracy summary
        for i, model in enumerate(models):
            test_accs = []
            for ds in datasets:
                acc = all_results[ds].get(model, {}).get("test_accuracy", 0)
                test_accs.append(acc)
            offset = (i - n_models / 2 + 0.5) * width
            axes[0].bar(x + offset, test_accs, width, label=model, alpha=0.8)

        axes[0].set_xlabel("Dataset", fontsize=12)
        axes[0].set_ylabel("Test Accuracy", fontsize=12)
        axes[0].set_title("Test Accuracy Summary", fontsize=14)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(datasets, rotation=45, ha="right")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].set_ylim(0, 1.1)

        # Test F1 summary
        for i, model in enumerate(models):
            test_f1s = []
            for ds in datasets:
                f1 = all_results[ds].get(model, {}).get("test_f1_score", 0)
                test_f1s.append(f1)
            offset = (i - n_models / 2 + 0.5) * width
            axes[1].bar(x + offset, test_f1s, width, label=model, alpha=0.8)

        axes[1].set_xlabel("Dataset", fontsize=12)
        axes[1].set_ylabel("Test F1 Score", fontsize=12)
        axes[1].set_title("Test F1 Score Summary", fontsize=14)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(datasets, rotation=45, ha="right")
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].set_ylim(0, 1.1)

        plt.tight_layout()
        summary_path = os.path.join(output_dir, f"summary_{timestamp}.png")
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        saved_plots["summary"] = summary_path
        print(f"Summary plot saved to: {summary_path}")
        if show:
            plt.show()
        plt.close()

    return saved_plots


def plot_individual_curves(
    all_results: Dict[str, Dict],
    output_dir: str,
    timestamp: str,
    show: bool = False,
) -> List[str]:
    """
    Generate individual learning curve plots for each dataset-model combination.

    Returns:
        List of saved plot paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = []

    for dataset, models_data in all_results.items():
        for model, metrics in models_data.items():
            if "history" not in metrics:
                continue

            history = metrics["history"]
            if not history:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Accuracy curves
            train_acc = history.get("train_acc", [])
            val_acc = history.get("val_acc", [])
            if train_acc:
                axes[0].plot(range(1, len(train_acc) + 1), train_acc, 'b-', label='Train', linewidth=2)
            if val_acc:
                val_freq = max(1, len(train_acc) // len(val_acc)) if val_acc else 1
                axes[0].plot(range(1, len(val_acc) * val_freq + 1, val_freq), val_acc, 'r--', label='Val', linewidth=2)
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Accuracy")
            axes[0].set_title("Accuracy")
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            # Loss curves
            train_loss = history.get("train_loss", [])
            val_loss = history.get("val_loss", [])
            if train_loss:
                axes[1].plot(range(1, len(train_loss) + 1), train_loss, 'b-', label='Train', linewidth=2)
            if val_loss:
                val_freq = max(1, len(train_loss) // len(val_loss)) if val_loss else 1
                axes[1].plot(range(1, len(val_loss) * val_freq + 1, val_freq), val_loss, 'r--', label='Val', linewidth=2)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            axes[1].set_title("Loss")
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            # F1 curves
            train_f1 = history.get("train_f1", [])
            val_f1 = history.get("val_f1", [])
            if train_f1:
                axes[2].plot(range(1, len(train_f1) + 1), train_f1, 'b-', label='Train', linewidth=2)
            if val_f1:
                val_freq = max(1, len(train_f1) // len(val_f1)) if val_f1 else 1
                axes[2].plot(range(1, len(val_f1) * val_freq + 1, val_freq), val_f1, 'r--', label='Val', linewidth=2)
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("F1 Score")
            axes[2].set_title("F1 Score")
            axes[2].legend()
            axes[2].grid(alpha=0.3)

            plt.suptitle(f"{model.upper()} on {dataset.upper()}", fontsize=14, fontweight='bold')
            plt.tight_layout()

            plot_path = os.path.join(output_dir, f"curves_{dataset}_{model}_{timestamp}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            saved_plots.append(plot_path)
            print(f"Individual curves saved to: {plot_path}")

            if show:
                plt.show()
            plt.close()

    return saved_plots


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Tuning Training for GNNs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset options
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=["amazon", "dblp", "email"],
        default=["amazon", "dblp", "email"],
        help="Datasets to use for hyperparameter tuning",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Data directory",
    )

    # Email feature flags (only used if email dataset is selected)
    parser.add_argument("--email-use-degree", action="store_true", default=False)
    parser.add_argument("--email-use-centrality", action="store_true", default=False)
    parser.add_argument("--email-use-local", action="store_true", default=False)
    parser.add_argument("--email-use-original", action="store_true", default=True)

    # Model options
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["gcn", "gat", "sage", "ppnp", "appnp",
                 "residual_gcn", "residual_gat", "residual_sage", "residual_appnp"],
        default=["gcn", "gat", "sage"],
        help="Models to tune and benchmark",
    )

    # Training options
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs per trial",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=20,
        help="Number of Bayesian optimization iterations per model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu). Auto-detected if not specified.",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/hyperparam_tuning",
        help="Output directory for results and plots",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=True,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--plot-individual",
        action="store_true",
        default=False,
        help="Generate individual learning curve plots for each model-dataset combo",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        default=False,
        help="Display plots interactively",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed progress",
    )

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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Store all results
    all_results = {}  # {dataset: {model: {metrics}}}

    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING TRAINING")
    print("=" * 80)
    print(f"Datasets: {args.datasets}")
    print(f"Models: {args.models}")
    print(f"Iterations per model: {args.n_iterations}")
    print(f"Epochs per trial: {args.epochs}")
    print("=" * 80)

    # Run hyperparameter tuning for each dataset-model combination
    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Loading dataset: {dataset_name.upper()}")
        print(f"{'='*60}")

        # Load dataset
        _, data = load_dataset(dataset_name, args.data_dir, email_features)
        data = create_masks(data) if not hasattr(data, 'train_mask') else data

        # Get dataset statistics
        num_nodes = data.num_nodes
        num_edges = data.edge_index.shape[1]
        num_features = data.x.shape[1] if data.x is not None else 0
        num_classes = int(data.y.max().item() + 1)
        print(f"Nodes: {num_nodes}, Edges: {num_edges}, Features: {num_features}, Classes: {num_classes}")

        all_results[dataset_name] = {}

        for model_name in args.models:
            print(f"\n--- Tuning {model_name.upper()} on {dataset_name.upper()} ---")

            # Perform hyperparameter tuning
            best_config, best_test_metrics, best_history, all_trials = hyperparameter_tuning(
                model_name=model_name,
                dataset_name=dataset_name,
                data=data,
                n_iterations=args.n_iterations,
                epochs=args.epochs,
                device=device,
                verbose=args.verbose,
            )

            # Store results
            all_results[dataset_name][model_name] = {
                "best_config": best_config,
                "test_accuracy": best_test_metrics.get("accuracy", 0) if best_test_metrics else 0,
                "test_f1_score": best_test_metrics.get("f1_score", 0) if best_test_metrics else 0,
                "history": best_history,
                "all_trials": all_trials,
            }

    # Generate comparison plots
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 60)

    saved_plots = plot_comparison(
        all_results=all_results,
        output_dir=args.output_dir,
        timestamp=timestamp,
        show=args.show_plots,
    )

    # Generate individual curves if requested
    if args.plot_individual:
        print("\nGenerating individual learning curves...")
        individual_plots = plot_individual_curves(
            all_results=all_results,
            output_dir=args.output_dir,
            timestamp=timestamp,
            show=args.show_plots,
        )
        saved_plots["individual"] = individual_plots

    # Save results to JSON
    if args.save_json:
        json_path = os.path.join(args.output_dir, f"results_{timestamp}.json")

        # Create serializable copy (exclude non-serializable objects)
        serializable_results = {}
        for ds, models_data in all_results.items():
            serializable_results[ds] = {}
            for model, metrics in models_data.items():
                serializable_results[ds][model] = {
                    "best_config": metrics.get("best_config", {}),
                    "test_accuracy": metrics.get("test_accuracy", 0),
                    "test_f1_score": metrics.get("test_f1_score", 0),
                    "all_trials": metrics.get("all_trials", []),
                    "history_summary": {
                        "final_train_acc": metrics.get("history", {}).get("train_acc", [])[-1:] if metrics.get("history") else [],
                        "final_val_acc": metrics.get("history", {}).get("val_acc", [])[-1:] if metrics.get("history") else [],
                        "final_train_loss": metrics.get("history", {}).get("train_loss", [])[-1:] if metrics.get("history") else [],
                        "final_val_loss": metrics.get("history", {}).get("val_loss", [])[-1:] if metrics.get("history") else [],
                    } if metrics.get("history") else {},
                }

        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nResults saved to: {json_path}")

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    for ds, models_data in all_results.items():
        print(f"\n[{ds.upper()}]")
        for model, metrics in models_data.items():
            print(f"  {model.upper()}:")
            print(f"    Test Accuracy: {metrics.get('test_accuracy', 0):.4f}")
            print(f"    Test F1 Score: {metrics.get('test_f1_score', 0):.4f}")
            if metrics.get("best_config"):
                cfg = metrics["best_config"]
                print(f"    Best Config: hid={cfg.get('hidden_channels', '?')}, "
                      f"layers={cfg.get('num_layers', '?')}, "
                      f"lr={cfg.get('lr', '?'):.2e}")

    print("\n" + "=" * 80)
    print("SAVED PLOTS")
    print("=" * 80)
    for plot_type, path in saved_plots.items():
        if isinstance(path, list):
            for p in path:
                print(f"  {p}")
        else:
            print(f"  {plot_type}: {path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

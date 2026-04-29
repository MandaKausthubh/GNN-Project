import os
import json
import argparse
import random
from itertools import product

from gnn_datasets.amazon import AmazonPhotos
from gnn_datasets.dblp import DBLP
from gnn_datasets.email import EmailEuCore
from utils import *
from utils.trainer import Trainer
from models.gcn import GCNWrapper
from models.gat import GATWrapper
from models.sage import SAGEWrapper
from models.ppnp import APPNPWrapper, PPNPWrapper

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.seed import seed_everything

import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any

from utils.bayes_hp import (
    get_bayesian_optimizer_for_model,
    convert_bayesian_params_to_trainable,
)

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

# ================= Useful stuff =================
def pretty_print_dicts(data, padding: int = 0):
    indent = " " * padding
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{indent}{key}:", end="")
            pretty_print_dicts(value, padding + 4)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            print(f"{indent}[{i}]:")
            pretty_print_dicts(item, padding + 4)
    else:
        print(f"{indent}{data}")



#  ================ DATASETS ================

def load_dataset(name, args):
    """Load a dataset by name."""
    if 'amazon' == name:
        dataset = AmazonPhotos(root=os.path.join(args.data_dir, "AmazonPhotos"))
        data = dataset[0]
        return (dataset, data)

    if 'dblp' == name:
        dataset = DBLP(root=os.path.join(args.data_dir, "DBLP"))
        data = dataset.get_homograph_apa()
        for mask_name in ['train_mask', 'val_mask', 'test_mask']:
            mask = getattr(dataset, mask_name, None)
            if mask is not None and mask.dim() == 2:
                setattr(dataset, mask_name, mask[:, 0])
        return dataset, data

    if 'email' == name:
        email_features = {
            "use_degree": args.email_use_degree,
            "use_centrality": args.email_use_centrality,
            "use_local": args.email_use_local,
            "use_original": args.email_use_original,
        }
        dataset = EmailEuCore(
            root=os.path.join(args.data_dir, "EmailEuCore"),
            **email_features
        )
        data = dataset[0]
        return (dataset, data)

    else:
        raise ValueError(f"Unknown dataset: {name}")




def get_dataset_stats(data: Data) -> Dict[str, Any]:
    """Get statistics about a dataset."""
    assert data is not None, "Data object cannot be None"
    stats = {
        "num_nodes": data.num_nodes,
        "num_edges": data.edge_index.shape[1],   # type: ignore
        "num_features": data.x.shape[1] if data.x is not None else 0,    # type: ignore
        "num_classes": int(data.y.max().item() + 1) if data.y is not None else 0,    # type: ignore
        "avg_degree": float(data.edge_index.shape[1] / data.num_nodes) if data.num_nodes > 0 else 0,    # type: ignore
    }
    return stats



def create_masks(
    data: Data,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Data:
    """Create train/val/test masks if they don't exist."""
    if hasattr(data, 'train_mask') and data.train_mask is not None:
        return data

    num_nodes = data.num_nodes
    seed_everything(seed)
    perm = torch.randperm(num_nodes)    # type: ignore

    train_size = int(num_nodes * train_ratio)    # type: ignore
    val_size = int(num_nodes * val_ratio)    # type: ignore

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)    # type: ignore
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)    # type: ignore
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)    # type: ignore

    data.train_mask[perm[:train_size]] = True
    data.val_mask[perm[train_size:train_size + val_size]] = True
    data.test_mask[perm[train_size + val_size:]] = True

    return data





# ================ MODELS ================

def create_model(name, in_channels, out_channels, hyperparams):
    """Create a model by name."""
    common_kwargs = {
        "in_channels": in_channels,
        "hidden_channels": hyperparams.get("hidden_channels", 128),
        "num_layers": hyperparams.get("num_layers", 2),
        "out_channels": out_channels,
        "dropout": hyperparams.get("dropout", 0.5),
        "norm": hyperparams.get("norm", "layer"),
    }

    if 'gcn' == name:
        return GCNWrapper(**common_kwargs)

    elif 'gat' == name:

        gat_kwargs = {
            "heads": hyperparams.get("gat_heads", 4),
            "concat": False if common_kwargs['num_layers'] > 1 else True
        }

        return GATWrapper(**common_kwargs, **gat_kwargs)

    elif 'graphsage' == name:
        return SAGEWrapper(**common_kwargs)

    elif 'appnp' == name:
        appnp_kwargs = {
            "K": hyperparams.get("K", 10),
            "alpha": hyperparams.get("alpha", 0.1)
        }
        return APPNPWrapper(**common_kwargs, **appnp_kwargs)

    elif 'ppnp' == name:
        ppnp_kwargs = {
            "alpha": hyperparams.get("alpha", 0.1)
        }
        return PPNPWrapper(**common_kwargs, **ppnp_kwargs)

    else:
        raise ValueError(f"Unknown model: {name}")



def train_single_model_dataset_config(
    model_name: str,
    dataset_name: str,
    hyperparams: Dict[str, Any],
    epochs: int = 100,
    device: torch.device = torch.device("cpu"),
    args: Optional[argparse.Namespace] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, List[float]], Dict[str, float], nn.Module]:
    """Train a single model on a single dataset configuration."""
    _, data = load_dataset(dataset_name, args)
    data = create_masks(data)

    stats = get_dataset_stats(data)
    if verbose:
        print(f"Dataset '{dataset_name}' stats:")
        pretty_print_dicts(stats)

    model = create_model(model_name, stats['num_features'], stats['num_classes'], hyperparams).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams.get("lr", 0.01),
        weight_decay=hyperparams.get("weight_decay", 5e-4)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model, optimizer, criterion, device,
        scheduler=scheduler,   # type: ignore
        gradient_clip_val=hyperparams.get("gradient_clip", 1.0),
        early_stopping_patience=hyperparams.get("early_stopping_patience", 20)
    )

    history = trainer.train(
        data, epochs=epochs, val_every=1, verbose=verbose
    )

    results = trainer.predict(data, "test_mask")
    test_metrics = {
        "accuracy": results["accuracy"],
        "f1": results["f1"],
    }

    return history, test_metrics, model



def hyperparameter_search(
    model_name: str,
    dataset_name: str,
    data: Data,
    search_type: str = "bayesian",
    n_trials: int = 20,
    epochs: int = 300,
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
        search_type: "grid", "random", or "bayesian".
        n_trials: Number of trials (for random/bayesian search).
        epochs: Training epochs per trial.
        val_metric: Metric to optimize ("accuracy" or "loss").
        device: Device to train on.
        verbose: Print progress.

    Returns:
        Tuple of (best_hyperparams, best_test_metrics, all_trial_results).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")   # type: ignore
    param_grid = HYPERPARAM_GRID.get(model_name, HYPERPARAM_GRID["gcn"])
    print("Starting hyperparameter search...")

    if search_type == "grid":
        # Generate all combinations
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        all_configs = [dict(zip(keys, v)) for v in product(*values)]
        if verbose:
            print(f"Grid search: {len(all_configs)} configurations")
    elif search_type == "random":
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
    elif search_type == "bayesian":
        # Bayesian optimization - will generate configs iteratively
        all_configs = None  # Generated on-the-fly
        if verbose:
            print(f"Bayesian optimization: {n_trials} trials")
    else:
        raise ValueError(f"Unknown search_type: {search_type}")

    best_val_score = -float("inf") if val_metric == "accuracy" else float("inf")
    best_config = None
    best_test_metrics = None
    all_results = []

    # Bayesian optimizer (only used if search_type == "bayesian")
    bayes_opt = None
    if search_type == "bayesian":
        bayes_opt = get_bayesian_optimizer_for_model(model_name)

    # Iterative search for Bayesian, or sequential for grid/random
    n_configs = n_trials if search_type in ("bayesian", "random") else len(all_configs)

    for i in range(n_configs):
        if search_type == "bayesian":
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
        else:
            config = all_configs[i]

        config_str = f"hid={config.get('hidden_channels', '?')},layers={config.get('num_layers', '?')},lr={config.get('lr', '?')}"
        try:
            _, val_metrics, _ = train_single_model_dataset_config(
                model_name=model_name,
                dataset_name=dataset_name,
                hyperparams=config,
                epochs=epochs,
                device=device,   # type: ignore
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

            # For Bayesian optimization, observe the result
            if search_type == "bayesian":
                bayes_opt.observe(raw_config, val_score)

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
                # pbar.set_postfix_str(f"Cfg-{i+1}: {config_str} | {val_metric}={val_score:.4f} (BEST)")
            else:
                # pbar.set_postfix_str(f"Cfg-{i+1}: {config_str} | {val_metric}={val_score:.4f}")
                pass

        except Exception as e:
            # pbar.set_postfix_str(f"Cfg-{i+1}: {config_str} | FAILED: {str(e)[:30]}")
            if verbose:
                print(f"  -> Failed: {e}")
            continue
        finally:
            # pbar.update(1)
            pass

    # pbar.set_postfix_str(f"Best: {val_metric}={best_val_score:.4f} | Config: {best_config}")
    # pbar.close()

    # Print final results summary
    print("\n" + "=" * 60)
    print(f"[FINAL RESULTS - Hyperparameter Search: {model_name} on {dataset_name} after {n_configs} trials]")
    print("=" * 60)
    print(f"Best Validation {val_metric}: {best_val_score:.4f}")
    print(f"Best Config: {best_config}")
    if best_test_metrics:
        print(f"Test Accuracy: {best_test_metrics.get('accuracy', 0):.4f}")
        print(f"Test F1 Score: {best_test_metrics.get('f1_score', 0):.4f}")
    print("=" * 60)

    return best_config, best_test_metrics, {"trials": all_results}







def run_baseline_evaluations_on_dataset(
    model_names: List[str],
    dataset_name: str,
    hyperparams: Dict[str, Any],
    epochs: int = 100,
    device: torch.device = torch.device("cpu"),
    args: Optional[argparse.Namespace] = None,
) -> Dict[str, Dict[str, float]]:
    """Run baseline evaluations for multiple models on a dataset."""
    results = {}
    for model_name in model_names:
        print(f"\nEvaluating {model_name} on {dataset_name} with hyperparameters...")
        pretty_print_dicts(hyperparams, padding=0)
        print('\n'+"-" * 50)
        print(f"Dataset stats:")
        pretty_print_dicts(get_dataset_stats(load_dataset(dataset_name, args)[1]), padding=0)
        print("-" * 50 + "\n")
        print(f"Training {model_name}...")

        # Find best hyperparameters for this model using Bayesian and train with those hyperparameters
        best_config, test_metrics, result_of_tuning = hyperparameter_search(
            model_name=model_name,
            dataset_name=dataset_name,
            data=load_dataset(dataset_name, args)[1],
            search_type="bayesian",
            n_trials=hyperparams.get("n_hyperparam_trials", 20),
            epochs=epochs,
            val_metric="accuracy",
            device=device,   # type: ignore
            verbose=True,
        )

        print(f"Best hyperparameters for {model_name}:")
        pretty_print_dicts(best_config, padding=4)
        print(f"Test metrics with best hyperparameters: {test_metrics}")
        results[model_name] = {
            "best_config": best_config,
            "test_metrics": test_metrics,
            "tuning_results": result_of_tuning,
        }

    # Plot the results:
    # Training Curves for best configs: Accuracy, F1 Score, Loss vs Steps
    # Validation Curves for all configs: Accuracy, F1 Score, Loss vs Steps
    # Plot should be a 3 x 2 subfigure grid: 3 rows for Accuracy, F1 Score, Loss; 2 columns for Training and Validation

    _, axes = plt.subplots(3, 2, figsize=(15, 12))
    for model_name, result in results.items():
        tuning_results = result["tuning_results"]["trials"]
        for trial in tuning_results:
            val_metrics = trial["val_metrics"]
            val_acc = val_metrics.get("accuracy", [])
            val_f1 = val_metrics.get("f1", [])
            val_loss = val_metrics.get("loss", [])
            steps = list(range(1, len(val_acc) + 1))

            axes[0, 1].plot(steps, val_acc, label=f"{model_name} Trial {trial['trial']}")
            axes[1, 1].plot(steps, val_f1, label=f"{model_name} Trial {trial['trial']}")
            axes[2, 1].plot(steps, val_loss, label=f"{model_name} Trial {trial['trial']}")

    axes[0, 1].set_title("Validation Accuracy")
    axes[1, 1].set_title("Validation F1 Score")
    axes[2, 1].set_title("Validation Loss")

    axes[0, 0].set_title("Training Accuracy")
    axes[1, 0].set_title("Training F1 Score")
    axes[2, 0].set_title("Training Loss")

    for ax in axes[:, 1]:
        ax.set_xlabel("Epochs")
        ax.legend()

    axes.set_title(f"Model Performance Curves: {dataset_name}")

    return results







def run_efficiency_plots(args):
    """Run efficiency plots for different models on a dataset."""
    print(f"Running efficiency plots for models: {args.models} on datasets: {args.datasets}")
    for dataset_name in args.datasets:
        results = run_baseline_evaluations_on_dataset(
            model_names=args.models,
            dataset_name=dataset_name,
            hyperparams={
                "hidden_channels": args.hidden_channels,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "norm": args.norm,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "gat_heads": args.gat_heads,
                "K": args.K,
                "alpha": args.alpha,
                "n_hyperparam_trials": args.n_hyperparam_trials,
            },
            epochs=args.epochs,
            device=torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")),
            args=args,
        )

        # Store results as JSON files:
        output_dir = os.path.join("results", "efficiency_plots", dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        for model_name, result in results.items():
            output_path = os.path.join(output_dir, f"{model_name}_results.json")
            with open(output_path, "w") as f:
                json.dump(result, f, indent=4)

    # Save or show the plot
    plt.tight_layout()
    plt.show()


















def main():
    parser = argparse.ArgumentParser(description='Plot efficiency curves for different models.')

    # parser.add_argument("--dataset", type=str, choices=["amazon", "dblp", "email"], default="email", help="Dataset to use")
    parser.add_argument("--datasets", type=str, nargs="+", default=["email"], help="List of datasets to use")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory",)

    # Email feature flags
    parser.add_argument("--email-use-degree", action="store_true", default=False)
    parser.add_argument("--email-use-centrality", action="store_true", default=False)
    parser.add_argument("--email-use-local", action="store_true", default=False)
    parser.add_argument("--email-use-original", action="store_true", default=True)

    # Model options
    parser.add_argument( "--models", type=str, nargs="+", default=None, help="List of models for benchmarking",)

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
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--n-hyperparam-trials", type=int, default=20, help="Number of hyperparameter trials for tuning")
    parser.add_argument("--verbose", action="store_true", default=False, help="Verbose output during training and tuning")

    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device} for experiments")

    run_efficiency_plots(args)



if __name__ == "__main__": 
    main()

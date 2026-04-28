"""
Plotting utilities for benchmark results.

Usage:
    python plot_benchmark.py --results-path ./outputs/email_features_benchmark_20260428_123456.json
    python plot_benchmark.py --results-dir ./outputs --latest
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def find_latest_results(results_dir: str, pattern: str = "email_features_benchmark_*.json") -> Optional[str]:
    """Find the most recent benchmark results file."""
    dir_path = Path(results_dir)
    files = sorted(dir_path.glob(pattern))
    if files:
        return str(files[-1])
    return None


def plot_accuracy_comparison(results: Dict[str, Any], save_path: str, show: bool = False):
    """
    Plot accuracy comparison across different configurations.

    For email features benchmark: plots accuracy by feature combination.
    For general benchmark: plots accuracy by model.
    """
    # Determine structure
    first_key = list(results.keys())[0]
    first_value = results[first_key]

    # Check if this is email features benchmark (nested structure)
    is_email_features = isinstance(first_value, dict) and any(
        isinstance(v, dict) and "accuracy_mean" in v
        for v in first_value.values()
    )

    if is_email_features:
        # Email features benchmark: combo_name -> model_name -> metrics
        plot_email_features_accuracy(results, save_path, show)
    else:
        # Simple benchmark: model_name -> metrics
        plot_simple_accuracy(results, save_path, show)


def plot_email_features_accuracy(results: Dict[str, Any], save_path: str, show: bool = False):
    """Plot accuracy for email features benchmark."""
    models = set()
    combos = list(results.keys())

    for combo_data in results.values():
        models.update(combo_data.keys())

    models = sorted(list(models))

    x = np.arange(len(combos))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, model in enumerate(models):
        accuracies = []
        errors = []
        for combo in combos:
            if model in results[combo]:
                acc = results[combo][model].get("accuracy_mean", 0)
                std = results[combo][model].get("accuracy_std", 0)
                accuracies.append(acc)
                errors.append(std)
            else:
                accuracies.append(0)
                errors.append(0)

        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, accuracies, width, yerr=errors if any(errors) else None,
                     label=model, capsize=3, alpha=0.8)

    ax.set_xlabel("Feature Combination", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Email-Eu-Core Benchmark: Accuracy by Feature Combination", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(combos, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Accuracy comparison plot saved to: {save_path}")

    if show:
        plt.show()
    plt.close()


def plot_simple_accuracy(results: Dict[str, Any], save_path: str, show: bool = False):
    """Plot accuracy for simple benchmark (single dataset)."""
    models = list(results.keys())
    accuracies = [results[m].get("accuracy_mean", 0) for m in models]
    stds = [results[m].get("accuracy_std", 0) for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(models, accuracies, yerr=stds, capsize=5, alpha=0.8)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Benchmark Results: Accuracy by Model", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Add value labels on bars
    for bar, acc, std in zip(bars, accuracies, stds):
        height = bar.get_height()
        ax.annotate(f"{acc:.4f}\n(+/- {std:.4f})",
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Accuracy comparison plot saved to: {save_path}")

    if show:
        plt.show()
    plt.close()


def plot_feature_importance(results: Dict[str, Any], save_path: str, show: bool = False):
    """
    Plot the impact of adding features on accuracy.
    Shows how accuracy changes as more features are added.
    """
    # Only applicable for email features benchmark
    first_key = list(results.keys())[0]
    if not isinstance(results[first_key], dict):
        print("Feature importance plot only applicable for email features benchmark.")
        return

    models = set()
    for combo_data in results.values():
        models.update(combo_data.keys())
    models = sorted(list(models))

    # Sort combinations by number of features
    def count_features(combo_name):
        return combo_name.count("_") + 1

    combos = sorted(results.keys(), key=count_features)

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 6), sharey=True)

    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        accuracies = []
        for combo in combos:
            if model in results[combo]:
                acc = results[combo][model].get("accuracy_mean", 0)
                std = results[combo][model].get("accuracy_std", 0)
                accuracies.append((acc, std))
            else:
                accuracies.append((0, 0))

        acc_values = [a[0] for a in accuracies]
        std_values = [a[1] for a in accuracies]

        ax.plot(range(len(combos)), acc_values, marker="o", linewidth=2, markersize=8)
        ax.fill_between(range(len(combos)),
                       [a - s for a, s in zip(acc_values, std_values)],
                       [a + s for a, s in zip(acc_values, std_values)],
                       alpha=0.3)

        ax.set_xlabel("Feature Combination (sorted by count)", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{model.upper()}", fontsize=12)
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels(combos, rotation=45, ha="right", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.0)

    plt.suptitle("Impact of Feature Combinations on Accuracy", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Feature importance plot saved to: {save_path}")

    if show:
        plt.show()
    plt.close()


def plot_all_metrics(results: Dict[str, Any], save_path: str, show: bool = False):
    """Plot both accuracy and F1 score side by side."""
    first_key = list(results.keys())[0]
    is_email_features = isinstance(results[first_key], dict)

    if is_email_features:
        models = set()
        for combo_data in results.values():
            models.update(combo_data.keys())
        models = sorted(list(models))
        combos = list(results.keys())

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Accuracy plot
        x = np.arange(len(combos))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            accuracies = [results[combo][model].get("accuracy_mean", 0) if model in results[combo] else 0
                         for combo in combos]
            offset = (i - len(models) / 2 + 0.5) * width
            axes[0].bar(x + offset, accuracies, width, label=model, alpha=0.8)

        axes[0].set_xlabel("Feature Combination")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Accuracy Comparison")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(combos, rotation=45, ha="right")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].set_ylim(0, 1.0)

        # F1 score plot
        for i, model in enumerate(models):
            f1_scores = [results[combo][model].get("f1_mean", 0) if model in results[combo] else 0
                        for combo in combos]
            offset = (i - len(models) / 2 + 0.5) * width
            axes[1].bar(x + offset, f1_scores, width, label=model, alpha=0.8)

        axes[1].set_xlabel("Feature Combination")
        axes[1].set_ylabel("F1 Score")
        axes[1].set_title("F1 Score Comparison")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(combos, rotation=45, ha="right")
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].set_ylim(0, 1.0)

        plt.suptitle("Benchmark Results: Accuracy and F1 Score", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"All metrics plot saved to: {save_path}")

        if show:
            plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--results-path", type=str, default=None,
                       help="Path to benchmark results JSON file")
    parser.add_argument("--results-dir", type=str, default="./outputs",
                       help="Directory containing results (used with --latest)")
    parser.add_argument("--latest", action="store_true",
                       help="Use the latest results file in results-dir")
    parser.add_argument("--output-dir", type=str, default="./plots",
                       help="Directory to save plots")
    parser.add_argument("--show", action="store_true",
                       help="Display plots interactively")
    parser.add_argument("--all", action="store_true",
                       help="Generate all plot types")

    args = parser.parse_args()

    # Find results file
    if args.results_path:
        results_path = args.results_path
    elif args.latest:
        results_path = find_latest_results(args.results_dir)
        if not results_path:
            print(f"No benchmark results found in {args.results_dir}")
            return
    else:
        print("Please specify --results-path or use --latest to find the most recent results")
        return

    print(f"Loading results from: {results_path}")
    results = load_results(results_path)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = Path(results_path).stem

    # Generate plots
    plot_accuracy_comparison(
        results,
        save_path=os.path.join(args.output_dir, f"{base_name}_accuracy.png"),
        show=args.show
    )

    plot_all_metrics(
        results,
        save_path=os.path.join(args.output_dir, f"{base_name}_all_metrics.png"),
        show=args.show
    )

    if args.all:
        plot_feature_importance(
            results,
            save_path=os.path.join(args.output_dir, f"{base_name}_feature_importance.png"),
            show=args.show
        )

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

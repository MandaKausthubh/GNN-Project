"""
Training time analysis utilities for GNN training.

Generates bar charts comparing average training time per epoch across models.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_avg_epoch_time_comparison(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    save_path: Optional[str] = None,
    show: bool = False,
    dpi: int = 150,
    figsize: Tuple[int, int] = (16, 6),
) -> None:
    """
    Plot average training time per epoch as grouped bar chart across models and datasets.

    Args:
        all_results: Benchmark results dict with structure:
            {dataset_name: {"benchmark": {model_name: {"individual_runs": {
                "histories": [{"epoch_times": [...]}, ...]
            }}}}}
        save_path: Path to save the figure. If None, figure is not saved.
        show: Whether to display the plot.
        dpi: DPI for saved figure.
        figsize: Figure size tuple (width, height).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed.")
        return

    datasets = list(all_results.keys())
    n_models = None
    all_model_names = []

    for dataset_name in datasets:
        benchmark = all_results[dataset_name].get("benchmark", all_results[dataset_name])
        model_names = list(benchmark.keys())
        if n_models is None:
            n_models = len(model_names)
            all_model_names = model_names
        elif set(model_names) != set(all_model_names):
            all_model_names = list(set(all_model_names) | set(model_names))

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_models or 1, 1)))

    for ax, dataset_name in zip(axes, datasets):
        benchmark = all_results[dataset_name].get("benchmark", all_results[dataset_name])
        model_names = list(benchmark.keys())

        avg_times = []
        std_times = []

        for model_name in model_names:
            model_res = benchmark.get(model_name, {})
            individual_runs = model_res.get("individual_runs", {})
            histories = individual_runs.get("histories", [])

            if histories:
                all_epoch_times = []
                for history in histories:
                    if isinstance(history, dict) and "epoch_times" in history:
                        all_epoch_times.extend(history["epoch_times"])
                    elif isinstance(history, list):
                        for h in history:
                            if isinstance(h, dict) and "epoch_times" in h:
                                all_epoch_times.extend(h["epoch_times"])

                if all_epoch_times:
                    avg_times.append(np.mean(all_epoch_times))
                    std_times.append(np.std(all_epoch_times))
                else:
                    avg_times.append(0.0)
                    std_times.append(0.0)
            else:
                avg_times.append(0.0)
                std_times.append(0.0)

        x_pos = np.arange(len(model_names))
        bars = ax.bar(x_pos, avg_times, yerr=std_times, capsize=4,
                      color=colors[:len(model_names)], alpha=0.8, edgecolor='black')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Avg Time per Epoch (seconds)', fontsize=12)
        ax.set_title(f'{dataset_name.upper()}', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.upper() for m in model_names], rotation=45, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        for bar, avg, std in zip(bars, avg_times, std_times):
            if avg > 0:
                ax.annotate(f'{avg:.3f}s',
                            xy=(bar.get_x() + bar.get_width() / 2, avg + std),
                            xytext=(0, 3),
                            textcoords='offset points',
                            ha='center', va='bottom', fontsize=8)

    plt.suptitle('Average Training Time per Epoch: Model Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Avg epoch time comparison plot saved to: {save_path}")

    if show:
        plt.show()

    plt.close()

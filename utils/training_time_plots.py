"""
Training time analysis utilities for GNN training.

Generates bar charts comparing average training time per epoch across models.
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_avg_epoch_time_comparison(
    avg_times: Dict[str, Dict[str, float]],
    std_times: Optional[Dict[str, Dict[str, float]]] = None,
    save_dir: Optional[str] = None,
    show: bool = False,
    dpi: int = 150,
) -> None:
    """
    Generate one bar chart per dataset showing avg training time per epoch across models.

    Args:
        avg_times: Dict mapping dataset_name -> model_name -> avg time per epoch (seconds).
        std_times: Optional dict of same shape with std deviations.
        save_dir: Directory to save plots. If None, plots are not saved.
        show: Whether to display plots.
        dpi: DPI for saved figures.
    """
    datasets = sorted(avg_times.keys())

    for dataset_name in datasets:
        model_names = sorted(avg_times[dataset_name].keys())
        means = [avg_times[dataset_name][m] for m in model_names]
        stds = [std_times[dataset_name][m] if std_times else 0.0 for m in model_names]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
        x_pos = list(range(len(model_names)))

        bars = ax.bar(
            x_pos,
            means,
            yerr=stds,
            capsize=5,
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )

        ax.set_xlabel("Model", fontsize=13)
        ax.set_ylabel("Avg Time per Epoch (seconds)", fontsize=13)
        ax.set_title(f"Avg Training Time per Epoch — {dataset_name.upper()}", fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.upper() for m in model_names], fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        for bar, mean in zip(bars, means):
            ax.annotate(
                f"{mean:.3f}s",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()

        if save_dir:
            out_path = os.path.join(save_dir, f"{dataset_name}_model_time_comparison.png")
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved: {out_path}")

        if show:
            plt.show()

        plt.close()

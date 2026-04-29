"""
Training time analysis and ablation comparison utilities for GNN training.

Generates bar charts for training time comparisons and ablation studies.
"""

import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Generic Avg Epoch Time (used by train.py benchmark_all_datasets)
# =============================================================================

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
        ax.set_title(
            f"Avg Training Time per Epoch — {dataset_name.upper()}", fontsize=14
        )
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
            out_path = os.path.join(
                save_dir, f"{dataset_name}_model_time_comparison.png"
            )
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved: {out_path}")

        if show:
            plt.show()

        plt.close()


# =============================================================================
# Ablation: Residual vs Non-Residual
# =============================================================================

def plot_residual_comparison(
    all_results: Dict[str, Dict[str, Any]],
    save_dir: Optional[str] = None,
    show: bool = False,
    dpi: int = 150,
) -> None:
    """
    Generate grouped bar charts comparing residual vs non-residual models.

    Produces two figures:
      1. Accuracy comparison per dataset (base vs +Res side by side)
      2. Improvement (delta) bar chart per dataset
      3. F1 comparison per dataset

    Args:
        all_results: Results dict from benchmark_residual_vs_base.
        save_dir: Directory to save plots.
        show: Whether to display plots.
        dpi: DPI for saved figures.
    """
    model_pairs = {
        "gcn": ("GCN", "GCN+Res"),
        "gat": ("GAT", "GAT+Res"),
        "sage": ("GraphSAGE", "GraphSAGE+Res"),
        "appnp": ("APPNP", "APPNP+Res"),
    }

    datasets = sorted(all_results.keys())
    base_models = [k for k in model_pairs.keys()]

    # ── Figure 1: Accuracy Comparison ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, dataset_name in zip(axes, datasets):
        comparison = all_results[dataset_name].get("comparison", {})
        base_accs, res_accs, base_stds, res_stds = [], [], [], []
        labels = []

        for base_key in base_models:
            if base_key not in comparison:
                continue
            res_key = "residual_" + base_key
            base_accs.append(comparison[base_key]["accuracy_mean"])
            res_accs.append(comparison[res_key]["accuracy_mean"])
            base_stds.append(comparison[base_key]["accuracy_std"])
            res_stds.append(comparison[res_key]["accuracy_std"])
            labels.append(model_pairs[base_key][0])

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width / 2, base_accs, width, label="Base", color="#4C72B0", alpha=0.85, edgecolor="black")
        bars2 = ax.bar(x + width / 2, res_accs, width, label="+Residual", color="#DD8452", alpha=0.85, edgecolor="black")

        ax.errorbar(x - width / 2, base_accs, yerr=base_stds, fmt="none", color="black", capsize=4)
        ax.errorbar(x + width / 2, res_accs, yerr=res_stds, fmt="none", color="black", capsize=4)

        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"{dataset_name.upper()}", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=10)

        for bars in [bars1, bars2]:
            for bar in bars:
                ax.annotate(
                    f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    fig.suptitle("Residual vs Non-Residual: Accuracy Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_dir:
        out_path = os.path.join(save_dir, "ablation_residual_accuracy.png")
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if show:
        plt.show()
    plt.close()

    # ── Figure 2: Accuracy Improvement (Delta) ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets))
    width = 0.25
    colors_imp = ["#55A868", "#8172B3", "#CCB974", "#E15759"]

    for i, base_key in enumerate(base_models):
        deltas = []
        for dataset_name in datasets:
            comparison = all_results[dataset_name].get("comparison", {})
            if base_key not in comparison:
                deltas.append(0.0)
                continue
            res_key = "residual_" + base_key
            delta = comparison[res_key]["accuracy_mean"] - comparison[base_key]["accuracy_mean"]
            deltas.append(delta)

        bars = ax.bar(x + (i - 1) * width, deltas, width, label=model_pairs[base_key][0],
                      color=colors_imp[i], alpha=0.85, edgecolor="black")

        for bar, delta in zip(bars, deltas):
            sign = "+" if delta >= 0 else ""
            ax.annotate(
                f"{sign}{delta:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Accuracy Improvement (Δ)", fontsize=12)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_title("Residual Gain: Accuracy Improvement over Base Model", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets], fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()

    if save_dir:
        out_path = os.path.join(save_dir, "ablation_residual_improvement.png")
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if show:
        plt.show()
    plt.close()

    # ── Figure 3: F1 Score Comparison ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, dataset_name in zip(axes, datasets):
        comparison = all_results[dataset_name].get("comparison", {})
        base_f1s, res_f1s, base_stds, res_stds = [], [], [], []
        labels = []

        for base_key in base_models:
            if base_key not in comparison:
                continue
            res_key = "residual_" + base_key
            base_f1s.append(comparison[base_key]["f1_mean"])
            res_f1s.append(comparison[res_key]["f1_mean"])
            base_stds.append(comparison[base_key]["f1_std"])
            res_stds.append(comparison[res_key]["f1_std"])
            labels.append(model_pairs[base_key][0])

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width / 2, base_f1s, width, label="Base", color="#4C72B0", alpha=0.85, edgecolor="black")
        bars2 = ax.bar(x + width / 2, res_f1s, width, label="+Residual", color="#DD8452", alpha=0.85, edgecolor="black")

        ax.errorbar(x - width / 2, base_f1s, yerr=base_stds, fmt="none", color="black", capsize=4)
        ax.errorbar(x + width / 2, res_f1s, yerr=res_stds, fmt="none", color="black", capsize=4)

        ax.set_ylabel("F1 Score", fontsize=12)
        ax.set_title(f"{dataset_name.upper()}", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=10)

        for bars in [bars1, bars2]:
            for bar in bars:
                ax.annotate(
                    f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    fig.suptitle("Residual vs Non-Residual: F1 Score Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_dir:
        out_path = os.path.join(save_dir, "ablation_residual_f1.png")
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if show:
        plt.show()
    plt.close()


# =============================================================================
# Ablation: Oversmoothing (Layer Depth)
# =============================================================================

def plot_oversmoothing_comparison(
    all_results: Dict[str, Dict[str, Any]],
    layer_counts: Optional[List[int]] = None,
    models: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    show: bool = False,
    dpi: int = 150,
) -> None:
    """
    Generate line plots and heatmaps showing how accuracy changes with depth.

    Produces:
      1. Line plot: Accuracy vs Layers per model, per dataset (one figure)
      2. Heatmap: Accuracy across models x layers, per dataset

    Args:
        all_results: Results dict from benchmark_layers.
        layer_counts: List of layer counts used.
        models: List of model names.
        save_dir: Directory to save plots.
        show: Whether to display plots.
        dpi: DPI for saved figures.
    """
    layer_counts = layer_counts or [2, 4, 8]
    datasets = sorted(all_results.keys())

    model_display = {
        "gcn": "GCN",
        "residual_gcn": "GCN+Res",
        "gat": "GAT",
        "residual_gat": "GAT+Res",
        "sage": "GraphSAGE",
        "residual_sage": "GraphSAGE+Res",
        "appnp": "APPNP",
        "residual_appnp": "APPNP+Res",
    }
    model_colors = {
        "gcn": "#4C72B0",
        "residual_gcn": "#4C72B0",
        "gat": "#55A868",
        "residual_gat": "#55A868",
        "sage": "#CCB974",
        "residual_sage": "#CCB974",
        "appnp": "#E15759",
        "residual_appnp": "#E15759",
    }
    model_linestyle = {
        "gcn": "-",
        "residual_gcn": "--",
        "gat": "-",
        "residual_gat": "--",
        "sage": "-",
        "residual_sage": "--",
        "appnp": "-",
        "residual_appnp": "--",
    }

    # ── Figure 1: Line Plot — Accuracy vs Layers ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, dataset_name in zip(axes, datasets):
        layers_data = all_results[dataset_name].get("layers", {})

        for model_name in models:
            if model_name not in layers_data:
                continue
            accs = []
            stds = []
            for lc in layer_counts:
                entry = layers_data[model_name].get(lc, {})
                accs.append(entry.get("accuracy_mean", 0.0))
                stds.append(entry.get("accuracy_std", 0.0))

            color = model_colors.get(model_name, "gray")
            ls = model_linestyle.get(model_name, "-")
            label = model_display.get(model_name, model_name)

            ax.plot(
                layer_counts,
                accs,
                marker="o",
                linewidth=2.5,
                linestyle=ls,
                label=label,
                color=color,
                alpha=0.9,
            )
            ax.fill_between(
                layer_counts,
                [a - s for a, s in zip(accs, stds)],
                [a + s for a, s in zip(accs, stds)],
                alpha=0.15,
                color=color,
            )

        ax.set_xlabel("Number of Layers", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"{dataset_name.upper()}", fontsize=14)
        ax.set_xticks(layer_counts)
        ax.set_xticklabels([str(lc) for lc in layer_counts], fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        "Oversmoothing Analysis: Accuracy vs Number of Layers",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_dir:
        out_path = os.path.join(save_dir, "ablation_oversmoothing_line.png")
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if show:
        plt.show()
    plt.close()

    # ── Figure 2: Heatmap — Accuracy across Models × Layers ─────────────────
    for dataset_name in datasets:
        fig, ax = plt.subplots(figsize=(10, 6))

        layers_data = all_results[dataset_name].get("layers", {})
        model_names_filtered = [m for m in models if m in layers_data]

        matrix = []
        for model_name in model_names_filtered:
            row = []
            for lc in layer_counts:
                entry = layers_data[model_name].get(lc, {})
                row.append(entry.get("accuracy_mean", 0.0))
            matrix.append(row)

        matrix = np.array(matrix)

        im = ax.imshow(
            matrix,
            cmap="RdYlGn",
            aspect="auto",
            vmin=0.4,
            vmax=1.0,
        )

        ax.set_xticks(range(len(layer_counts)))
        ax.set_xticklabels([f"L{lc}" for lc in layer_counts], fontsize=12)
        ax.set_yticks(range(len(model_names_filtered)))
        ax.set_yticklabels(
            [model_display.get(m, m.upper()) for m in model_names_filtered], fontsize=11
        )

        for i in range(len(model_names_filtered)):
            for j in range(len(layer_counts)):
                text = ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=11,
                    fontweight="bold",
                )

        ax.set_xlabel("Number of Layers", fontsize=13)
        ax.set_ylabel("Model", fontsize=13)
        ax.set_title(
            f"Oversmoothing Heatmap — {dataset_name.upper()}",
            fontsize=14,
            fontweight="bold",
        )

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Accuracy", fontsize=11)

        plt.tight_layout()

        if save_dir:
            out_path = os.path.join(
                save_dir, f"ablation_oversmoothing_heatmap_{dataset_name}.png"
            )
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved: {out_path}")

        if show:
            plt.show()
        plt.close()

    # ── Figure 3: Bar chart — Accuracy drop from L2 to L8 ─────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    drop_data = {}

    for model_name in models:
        l2_accs = []
        l8_accs = []
        for dataset_name in datasets:
            layers_data = all_results[dataset_name].get("layers", {})
            if model_name not in layers_data:
                continue
            l2 = layers_data[model_name].get(2, {}).get("accuracy_mean", 0.0)
            l8 = layers_data[model_name].get(8, {}).get("accuracy_mean", 0.0)
            l2_accs.append(l2)
            l8_accs.append(l8)

        if l2_accs:
            drop_data[model_name] = np.mean(l8_accs) - np.mean(l2_accs)

    model_names = list(drop_data.keys())
    drops = [drop_data[m] for m in model_names]
    colors = [
        (
            "#E15759" if d < -0.05
            else "#59A14F" if d > 0.01
            else "#F1E457"
        )
        for d in drops
    ]
    labels = [model_display.get(m, m.upper()) for m in model_names]

    x = np.arange(len(labels))
    bars = ax.bar(x, drops, color=colors, alpha=0.85, edgecolor="black")

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Accuracy Change (L8 − L2)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(
        "Oversmoothing Impact: Accuracy Change from 2 to 8 Layers\n(negative = oversmoothing)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for bar, drop in zip(bars, drops):
        sign = "+" if drop >= 0 else ""
        ax.annotate(
            f"{sign}{drop:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E15759", alpha=0.85, label="Significant Drop (> 0.05)"),
        Patch(facecolor="#F1E457", alpha=0.85, label="Minimal Change"),
        Patch(facecolor="#59A14F", alpha=0.85, label="Improved/Stable"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)

    plt.tight_layout()

    if save_dir:
        out_path = os.path.join(save_dir, "ablation_oversmoothing_drop.png")
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if show:
        plt.show()
    plt.close()

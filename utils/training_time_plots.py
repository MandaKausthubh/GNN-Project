"""
Training time analysis utilities for GNN training.

Generates plots showing training time per epoch and cumulative training time.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_training_time_per_epoch(epoch_times: List[float],
                                  save_path: Optional[str] = None,
                                  show: bool = False,
                                  dpi: int = 150,
                                  figsize: Tuple[int, int] = (12, 5),
                                  title: Optional[str] = None,
                                  color: str = '#2E86AB') -> None:
    """
    Plot training time per epoch.

    Args:
        epoch_times: List of training times for each epoch (in seconds).
        save_path: Path to save the figure. If None, figure is not saved.
        show: Whether to display the plot.
        dpi: DPI for saved figure.
        figsize: Figure size tuple (width, height).
        title: Optional plot title.
        color: Color for the bars/lines.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    epochs = range(1, len(epoch_times) + 1)
    avg_time = np.mean(epoch_times)
    std_time = np.std(epoch_times)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Bar plot of time per epoch
    axes[0].bar(epochs, epoch_times, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[0].axhline(y=avg_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_time:.3f}s')
    axes[0].fill_between(epochs,
                         [avg_time - std_time] * len(epochs),
                         [avg_time + std_time] * len(epochs),
                         color='red', alpha=0.1, label=f'Std: {std_time:.3f}s')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Time (seconds)', fontsize=12)
    axes[0].set_title(title or 'Training Time per Epoch', fontsize=14)
    axes[0].legend(loc='upper right')
    axes[0].grid(axis='y', alpha=0.3)

    # Cumulative time plot
    cumulative_times = np.cumsum(epoch_times)
    axes[1].plot(epochs, cumulative_times, color=color, linewidth=2, marker='o', markersize=3)
    axes[1].fill_between(epochs, cumulative_times, alpha=0.3, color=color)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Cumulative Time (seconds)', fontsize=12)
    axes[1].set_title('Cumulative Training Time', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    # Add total time annotation
    total_time = cumulative_times[-1]
    axes[1].annotate(f'Total: {total_time:.2f}s',
                     xy=(len(epochs), total_time),
                     xytext=(-60, -20),
                     textcoords='offset points',
                     fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Training time plot saved to: {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_training_time_comparison(models_times: Dict[str, List[float]],
                                    save_path: Optional[str] = None,
                                    show: bool = False,
                                    dpi: int = 150,
                                    figsize: Tuple[int, int] = (14, 6)) -> None:
    """
    Plot training time comparison across multiple models.

    Args:
        models_times: Dict mapping model names to lists of epoch times.
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

    n_models = len(models_times)
    model_names = list(models_times.keys())

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Prepare data for box plot
    times_data = [models_times[name] for name in model_names]
    avg_times = [np.mean(times) for times in times_data]
    std_times = [np.std(times) for times in times_data]
    total_times = [np.sum(times) for times in times_data]

    # Colors for each model
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    # Box plot of time per epoch
    bp = axes[0].boxplot(times_data, labels=model_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylabel('Time per Epoch (seconds)', fontsize=12)
    axes[0].set_title('Training Time Distribution by Model', fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)

    # Bar plot of average time per epoch
    x_pos = np.arange(len(model_names))
    bars = axes[1].bar(x_pos, avg_times, yerr=std_times, capsize=5,
                       color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Model', fontsize=12)
    axes[1].set_ylabel('Average Time per Epoch (seconds)', fontsize=12)
    axes[1].set_title('Average Training Time Comparison', fontsize=14)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, avg, total in zip(bars, avg_times, total_times):
        height = bar.get_height()
        axes[1].annotate(f'{avg:.3f}s\n(total: {total:.1f}s)',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords='offset points',
                         ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Training time comparison plot saved to: {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_training_time_summary(epoch_times: List[float],
                                model_name: str,
                                dataset_name: str,
                                save_path: Optional[str] = None,
                                show: bool = False,
                                dpi: int = 150) -> None:
    """
    Generate a comprehensive training time summary for a single model run.

    Args:
        epoch_times: List of training times for each epoch (in seconds).
        model_name: Name of the model.
        dataset_name: Name of the dataset.
        save_path: Path to save the figure. If None, figure is not saved.
        show: Whether to display the plot.
        dpi: DPI for saved figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed.")
        return

    epochs = range(1, len(epoch_times) + 1)
    avg_time = np.mean(epoch_times)
    std_time = np.std(epoch_times)
    min_time = np.min(epoch_times)
    max_time = np.max(epoch_times)
    total_time = np.sum(epoch_times)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle(f'Training Time Analysis: {model_name.upper()} on {dataset_name.upper()}',
                 fontsize=16, fontweight='bold')

    # Time per epoch (main plot)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.bar(epochs, epoch_times, color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=avg_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_time:.3f}s')
    ax1.fill_between(epochs,
                     [avg_time - std_time] * len(epochs),
                     [avg_time + std_time] * len(epochs),
                     color='red', alpha=0.1)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Time (seconds)', fontsize=11)
    ax1.set_title('Training Time per Epoch', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)

    # Cumulative time
    ax2 = fig.add_subplot(gs[1, 0])
    cumulative_times = np.cumsum(epoch_times)
    ax2.plot(epochs, cumulative_times, color='#2E86AB', linewidth=2, marker='o', markersize=3)
    ax2.fill_between(epochs, cumulative_times, alpha=0.3, color='#2E86AB')
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Cumulative Time (s)', fontsize=10)
    ax2.set_title('Cumulative Time', fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Histogram of epoch times
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(epoch_times, bins=20, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax3.axvline(x=avg_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_time:.3f}s')
    ax3.set_xlabel('Time (seconds)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Time Distribution', fontsize=11)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    # Statistics box
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    stats_text = (
        f"Training Time Statistics\n"
        f"{'='*30}\n\n"
        f"Total Time: {total_time:.2f}s\n"
        f"({total_time/60:.2f} min)\n\n"
        f"Average: {avg_time:.3f}s\n"
        f"Std Dev: {std_time:.3f}s\n"
        f"Min: {min_time:.3f}s\n"
        f"Max: {max_time:.3f}s\n"
        f"Epochs: {len(epoch_times)}"
    )
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Efficiency metric (time per 100 epochs projected)
    ax5 = fig.add_subplot(gs[2, 0])
    time_per_100 = avg_time * 100
    ax5.bar(['Time/100 epochs'], [time_per_100], color='#A23B72', alpha=0.8, edgecolor='black')
    ax5.set_ylabel('Time (seconds)', fontsize=10)
    ax5.set_title('Estimated Time per 100 Epochs', fontsize=11)
    ax5.grid(axis='y', alpha=0.3)
    for i, v in enumerate([time_per_100]):
        ax5.text(i, v + time_per_100*0.02, f'{v:.1f}s', ha='center', fontsize=10)

    # Relative time (if multiple runs would be compared)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    ax6.text(0.5, 0.5, 'Per-Epoch Efficiency', transform=ax6.transAxes, fontsize=12,
             ha='center', va='center', style='italic', color='gray')

    # Summary
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Training time summary plot saved to: {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_avg_epoch_time_comparison(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    save_path: Optional[str] = None,
    show: bool = False,
    dpi: int = 150,
    figsize: Tuple[int, int] = (16, 6),
) -> None:
    """
    Plot average training time per epoch comparison across models and datasets.

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

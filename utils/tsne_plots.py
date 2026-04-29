"""
t-SNE visualization utilities for GNN embeddings.

Generates t-SNE plots of node embeddings colored by class labels.
"""

import os
from typing import Optional, Tuple


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.seed import seed_everything
from sklearn.manifold import TSNE


def get_node_embeddings(model: torch.nn.Module, data: Data,
                         device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract node embeddings from a trained GNN model.

    Args:
        model: Trained GNN model.
        data: PyG Data object.
        device: Device to run inference on.

    Returns:
        Tuple of (embeddings, labels) as numpy arrays.
    """
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
        # If embeddings is a tuple (logits, embeddings), take the first element
        if isinstance(embeddings, tuple):
            embeddings = embeddings[0]

    embeddings = embeddings.cpu().numpy()
    labels = data.y.cpu().numpy()

    return embeddings, labels


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray,
              save_path: Optional[str] = None, show: bool = False,
              perplexity: float = 30.0, n_iter: int = 1000,
              dpi: int = 150, figsize: Tuple[int, int] = (10, 8),
              title: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Generate and optionally save a t-SNE visualization of node embeddings.

    Args:
        embeddings: Node embeddings of shape (num_nodes, embedding_dim).
        labels: Node labels of shape (num_nodes,).
        save_path: Path to save the figure. If None, figure is not saved.
        show: Whether to display the plot.
        perplexity: t-SNE perplexity parameter.
        n_iter: t-SNE number of iterations.
        dpi: DPI for saved figure.
        figsize: Figure size tuple (width, height).
        title: Optional plot title.

    Returns:
        The matplotlib Figure object, or None if not applicable.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return None

    num_classes = int(labels.max() + 1)
    unique_labels = np.unique(labels)

    # Generate t-SNE embedding
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                random_state=42, learning_rate='auto', init='pca')
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Generate distinct colors for each class
    colors = plt.cm.get_cmap('tab10', num_classes)
    if num_classes > 10:
        colors = plt.cm.get_cmap('hsv', num_classes)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[colors(label)], label=f'Class {label}',
                   alpha=0.7, s=20, edgecolors='none')

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(title or 't-SNE Visualization of Node Embeddings', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"t-SNE plot saved to: {save_path}")

    if show:
        plt.show()

    plt.close()

    return None


def plot_tsne_from_model(model: torch.nn.Module, data: Data,
                          device: torch.device, save_path: Optional[str] = None,
                          show: bool = False, perplexity: float = 30.0,
                          n_iter: int = 1000, dpi: int = 150,
                          title: Optional[str] = None):
    """
    Generate t-SNE plot directly from a trained model and data.

    Args:
        model: Trained GNN model.
        data: PyG Data object.
        device: Device to run inference on.
        save_path: Path to save the figure. If None, figure is not saved.
        show: Whether to display the plot.
        perplexity: t-SNE perplexity parameter.
        n_iter: t-SNE number of iterations.
        dpi: DPI for saved figure.
        title: Optional plot title.
    """
    embeddings, labels = get_node_embeddings(model, data, device)
    plot_tsne(embeddings, labels, save_path=save_path, show=show,
              perplexity=perplexity, n_iter=n_iter, dpi=dpi, title=title)


def plot_tsne_comparison(model_results: dict, data: Data,
                          device: torch.device, save_path: Optional[str] = None,
                          show: bool = False, perplexity: float = 30.0,
                          dpi: int = 150, figsize: Tuple[int, int] = (18, 6)):
    """
    Generate t-SNE plots for multiple models side by side for comparison.

    Args:
        model_results: Dict mapping model names to trained models.
        data: PyG Data object.
        device: Device to run inference on.
        save_path: Path to save the figure. If None, figure is not saved.
        show: Whether to display the plot.
        perplexity: t-SNE perplexity parameter.
        dpi: DPI for saved figure.
        figsize: Figure size tuple (width, height).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed.")
        return

    n_models = len(model_results)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, model) in zip(axes, model_results.items()):
        embeddings, labels = get_node_embeddings(model, data, device)

        num_classes = int(labels.max() + 1)
        unique_labels = np.unique(labels)

        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000,
                    random_state=42, learning_rate='auto', init='pca')
        embeddings_2d = tsne.fit_transform(embeddings)

        colors = plt.cm.get_cmap('tab10', num_classes)
        if num_classes > 10:
            colors = plt.cm.get_cmap('hsv', num_classes)

        for label in unique_labels:
            mask = labels == label
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=[colors(label)], label=f'Class {label}',
                       alpha=0.7, s=15, edgecolors='none')

        ax.set_xlabel('t-SNE Dim 1', fontsize=10)
        ax.set_ylabel('t-SNE Dim 2', fontsize=10)
        ax.set_title(f'{model_name.upper()}', fontsize=12)
        ax.grid(True, alpha=0.3)

    # Add legend to the last subplot
    handles, labels_legend = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='upper right', bbox_to_anchor=(0.99, 0.99),
               fontsize=9, title='Classes')

    plt.suptitle('t-SNE Comparison of Model Embeddings', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"t-SNE comparison plot saved to: {save_path}")

    if show:
        plt.show()

    plt.close()

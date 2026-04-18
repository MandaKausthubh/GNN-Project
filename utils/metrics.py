"""
Metric computation utilities.
"""

from typing import Optional

import torch
from torch import Tensor


def compute_accuracy(pred: Tensor, y: Tensor, mask: Optional[Tensor] = None) -> float:
    """
    Compute classification accuracy.

    Args:
        pred: Predicted class labels.
        y: Ground truth labels.
        mask: Optional mask to compute accuracy on specific nodes.

    Returns:
        Accuracy as a float.
    """
    if mask is not None:
        pred = pred[mask]
        y = y[mask]

    if len(pred) == 0:
        return 0.0

    return (pred == y).float().mean().item()


def compute_f1(
    pred: Tensor,
    y: Tensor,
    average: str = "macro",
    zero_division: float = 0.0,
) -> float:
    """
    Compute F1 score.

    Args:
        pred: Predicted class labels.
        y: Ground truth labels.
        average: Averaging method ("macro", "micro", "weighted").
        zero_division: Value to return when there is no intersection.

    Returns:
        F1 score as a float.
    """
    from sklearn.metrics import f1_score

    pred_np = pred.cpu().numpy() if pred.is_cuda else pred.numpy()
    y_np = y.cpu().numpy() if y.is_cuda else y.numpy()

    return f1_score(y_np, pred_np, average=average, zero_division=zero_division)


def compute_roc_auc(
    pred_probs: Tensor,
    y: Tensor,
    multi_class: str = "ovr",
) -> float:
    """
    Compute ROC-AUC score.

    Args:
        pred_probs: Predicted probabilities (class probabilities).
        y: Ground truth labels.
        multi_class: Strategy for multi-class ("ovr", "ovo").

    Returns:
        ROC-AUC score as a float.
    """
    from sklearn.metrics import roc_auc_score

    if pred_probs.dim() == 1:
        # Binary case
        return roc_auc_score(y.cpu().numpy(), pred_probs.cpu().numpy())

    # Multi-class case
    pred_probs_np = pred_probs.cpu().detach().numpy()
    y_np = y.cpu().numpy()

    return roc_auc_score(y_np, pred_probs_np, multi_class=multi_class)


def compute_confusion_matrix(
    pred: Tensor,
    y: Tensor,
    num_classes: int = None,
) -> torch.Tensor:
    """
    Compute confusion matrix.

    Args:
        pred: Predicted class labels.
        y: Ground truth labels.
        num_classes: Number of classes. If None, inferred from max label.

    Returns:
        Confusion matrix as a tensor.
    """
    from sklearn.metrics import confusion_matrix

    if num_classes is None:
        num_classes = max(pred.max().item(), y.max().item()) + 1

    pred_np = pred.cpu().numpy()
    y_np = y.cpu().numpy()

    cm = confusion_matrix(y_np, pred_np, labels=list(range(num_classes)))
    return torch.tensor(cm, dtype=torch.long)


__all__ = [
    "compute_accuracy",
    "compute_f1",
    "compute_roc_auc",
    "compute_confusion_matrix",
]
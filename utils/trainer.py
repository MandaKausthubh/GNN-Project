"""
Training and inference utilities with full wandb support.
"""

import os
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from tqdm import tqdm


class Trainer:
    """
    Trainer class for GNN models with comprehensive training and inference cycles.

    Features:
        - Full wandb logging support
        - Normalization options (LayerNorm, BatchNorm)
        - Dropout regularization
        - Residual connections
        - Learning rate scheduling
        - Early stopping
        - Gradient clipping
        - Mixed precision training

    Args:
        model: The GNN model to train.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Device to train on.
        use_wandb: Whether to use wandb for logging.
        wandb_kwargs: Keyword arguments for wandb init.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module = None,
        device: Optional[Union[str, torch.device]] = None,
        use_wandb: bool = False,
        wandb_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping_patience: int = 0,
        early_stopping_delta: float = 0.0,
        gradient_clip_val: Optional[float] = None,
        mixed_precision: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model = self.model.to(self.device)

        # Training settings
        self.gradient_clip_val = gradient_clip_val
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        # Wandb setup
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
                if wandb_kwargs:
                    wandb.init(**wandb_kwargs)
                    wandb.watch(model)
            except ImportError:
                print("wandb not installed. Install with: pip install wandb")
                self.use_wandb = False

        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_f1": [],
            "val_f1": [],
            "learning_rate": [],
        }

    def train_epoch(self, data: Data) -> Tuple[float, float, float]:
        """
        Train for one epoch.

        Args:
            data: PyG Data object with train_mask.

        Returns:
            Tuple of (loss, accuracy, f1_score).
        """
        self.model.train()
        total_loss = 0.0
        num_correct = 0
        num_examples = 0

        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                out = self.model(data.x, data.edge_index)
                loss = self.criterion(out[data.train_mask], data.y[data.train_mask])

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            if self.gradient_clip_val:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad.clip_grad_value_(
                    self.model.parameters(), self.gradient_clip_val
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()

            if self.gradient_clip_val:
                torch.nn.utils.clip_grad.clip_grad_value_(
                    self.model.parameters(), self.gradient_clip_val
                )
            self.optimizer.step()

        total_loss = loss.item()

        # Compute metrics on training set
        pred = out.argmax(dim=1)
        train_mask = data.train_mask
        if train_mask is not None:
            num_correct = (pred[train_mask] == data.y[train_mask]).sum().item()
            num_examples = train_mask.sum().item()

        accuracy = num_correct / num_examples if num_examples > 0 else 0.0
        f1 = self._compute_f1(pred[train_mask], data.y[train_mask]) if num_examples > 0 else 0.0

        return total_loss, accuracy, f1

    @torch.no_grad()
    def validate(self, data: Data, mask_name: str = "val_mask") -> Tuple[float, float, float]:
        """
        Validate the model.

        Args:
            data: PyG Data object.
            mask_name: Name of the mask to use ("val_mask" or "test_mask").

        Returns:
            Tuple of (loss, accuracy, f1_score).
        """
        self.model.eval()
        total_loss = 0.0
        num_correct = 0
        num_examples = 0

        mask = getattr(data, mask_name, None)
        if mask is None:
            return 0.0, 0.0, 0.0

        out = self.model(data.x, data.edge_index)
        loss = self.criterion(out[mask], data.y[mask])
        total_loss = loss.item()

        pred = out.argmax(dim=1)
        num_correct = (pred[mask] == data.y[mask]).sum().item()
        num_examples = mask.sum().item()

        accuracy = num_correct / num_examples if num_examples > 0 else 0.0
        f1 = self._compute_f1(pred[mask], data.y[mask]) if num_examples > 0 else 0.0

        return total_loss, accuracy, f1

    def _compute_f1(self, pred: Tensor, y: Tensor) -> float:
        """Compute macro F1 score."""
        from sklearn.metrics import f1_score
        return f1_score(y.cpu().numpy(), pred.cpu().numpy(), average="macro", zero_division=0)

    def train(
        self,
        data: Data,
        epochs: int,
        val_every: int = 1,
        print_every: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            data: PyG Data object with train_mask, val_mask, and optionally test_mask.
            epochs: Number of training epochs.
            val_every: Validate every N epochs.
            print_every: Print progress every N epochs.

        Returns:
            Training history dictionary.
        """
        for epoch in range(1, epochs + 1):
            train_loss, train_acc, train_f1 = self.train_epoch(data)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["train_f1"].append(train_f1)
            self.history["learning_rate"].append(current_lr)

            # Validation
            val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0
            should_validate = epoch % val_every == 0 or epoch == epochs
            if should_validate:
                val_loss, val_acc, val_f1 = self.validate(data, "val_mask")
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                self.history["val_f1"].append(val_f1)

                # Early stopping check
                if self.early_stopping_patience > 0:
                    improved = val_loss < self.best_val_loss - self.early_stopping_delta
                    if improved:
                        self.best_val_loss = val_loss
                        self.epochs_without_improvement = 0
                    else:
                        self.epochs_without_improvement += 1

                    if self.epochs_without_improvement >= self.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

            # Scheduler step
            if self.scheduler is not None:
                # ReduceLROnPlateau requires a metric, others don't
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Logging
            if self.use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "train/f1": train_f1,
                    "lr": current_lr,
                }
                if should_validate:
                    log_dict["val/loss"] = val_loss
                    log_dict["val/accuracy"] = val_acc
                    log_dict["val/f1"] = val_f1
                self._wandb.log(log_dict)

            # Print progress
            if epoch % print_every == 0 or epoch == 1:
                val_str = f"| Val: {val_acc:.4f}" if should_validate else ""
                print(f"Epoch {epoch:3d}/{epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} {val_str}")

        return self.history

    @torch.no_grad()
    def inference(
        self,
        data: Data,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """
        Run inference on the full graph.

        Args:
            data: PyG Data object.
            batch_size: Batch size for mini-batch inference.

        Returns:
            Dictionary with predictions and embeddings.
        """
        self.model.eval()
        out = self.model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        return {
            "predictions": pred,
            "logits": out,
            "embeddings": self.model.emb,
        }

    @torch.no_grad()
    def predict(
        self,
        data: Data,
        mask_name: str = "test_mask",
    ) -> Dict[str, Any]:
        """
        Predict on a specific mask (train/val/test).

        Args:
            data: PyG Data object.
            mask_name: Name of the mask to predict on.

        Returns:
            Dictionary with predictions, ground truth, and metrics.
        """
        self.model.eval()
        mask = getattr(data, mask_name, None)

        if mask is None:
            return {}

        out = self.model(data.x, data.edge_index)
        pred = out.argmax(dim=1)[mask]
        y_true = data.y[mask]

        accuracy = (pred == y_true).float().mean().item()
        f1 = self._compute_f1(pred, y_true)

        return {
            "predictions": pred,
            "ground_truth": y_true,
            "accuracy": accuracy,
            "f1_score": f1,
        }

    def save_checkpoint(self, path: str, epoch: int = None):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", {})
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint.get("epoch", 0)

    def __repr__(self) -> str:
        return (
            f"Trainer(\n"
            f"  model={self.model.__class__.__name__},\n"
            f"  device={self.device},\n"
            f"  wandb={'enabled' if self.use_wandb else 'disabled'},\n"
            f"  mixed_precision={'enabled' if self.mixed_precision else 'disabled'}\n"
            f")"
        )
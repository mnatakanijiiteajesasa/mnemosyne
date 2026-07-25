"""
gnn_engine/trainer.py

Trainer for MemoryGNN with checkpoint management.
"""

import torch
from pathlib import Path
from typing import List, Dict, Optional, Any
from torch_geometric.data import Data

from .model import MemoryGNN


class GNNTrainer:
    """
    Trainer for MemoryGNN with checkpoint management.
    """

    def __init__(self, model: MemoryGNN, device: str = "cpu",
                 checkpoint_dir: str = "/tmp/gnn_checkpoints"):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)

    def train_epoch(self, data_list: List[Data], alpha: float = 0.5):
        """
        Train for one epoch over a batch of graphs.

        Args:
            data_list: list of torch_geometric.Data objects with:
                - x: node features [N, INPUT_DIM]
                - edge_index: edges [2, E]
                - y_relevance: binary labels [N]
                - y_cluster: categorical labels [N]
            alpha: balance factor for losses

        Returns:
            avg_loss, avg_rel_loss, avg_cluster_loss
        """
        self.model.train()
        total_loss = 0.0
        total_rel_loss = 0.0
        total_cluster_loss = 0.0

        for data in data_list:
            data = data.to(self.device)
            self.optimizer.zero_grad()

            # Forward
            h, r, c = self.model(data.x, data.edge_index)

            # Loss
            loss, rel_loss, cluster_loss = self.model.loss(
                h, r, c,
                data.y_relevance, data.y_cluster,
                alpha=alpha
            )

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_rel_loss += rel_loss.item()
            total_cluster_loss += cluster_loss.item()

        self.scheduler.step()

        n = len(data_list)
        return (
            total_loss / n,
            total_rel_loss / n,
            total_cluster_loss / n,
        )

    @torch.no_grad()
    def evaluate(self, data_list: List[Data], alpha: float = 0.5):
        """
        Evaluate on a batch of graphs (no gradient updates).

        Returns:
            avg_loss, avg_rel_loss, avg_cluster_loss, rel_acc, cluster_acc
        """
        self.model.eval()
        total_loss = 0.0
        total_rel_loss = 0.0
        total_cluster_loss = 0.0
        total_rel_correct = 0
        total_cluster_correct = 0
        total_samples = 0

        for data in data_list:
            data = data.to(self.device)

            # Forward
            h, r, c = self.model(data.x, data.edge_index)

            # Loss
            loss, rel_loss, cluster_loss = self.model.loss(
                h, r, c,
                data.y_relevance, data.y_cluster,
                alpha=alpha
            )

            total_loss += loss.item()
            total_rel_loss += rel_loss.item()
            total_cluster_loss += cluster_loss.item()

            # Accuracies
            rel_pred = (r.squeeze() > 0.5).long()
            cluster_pred = c.argmax(dim=1)

            total_rel_correct += (rel_pred == data.y_relevance).sum().item()
            total_cluster_correct += (cluster_pred == data.y_cluster).sum().item()
            total_samples += data.x.size(0)

        n = len(data_list)
        rel_acc = total_rel_correct / total_samples if total_samples > 0 else 0.0
        cluster_acc = total_cluster_correct / total_samples if total_samples > 0 else 0.0

        return (
            total_loss / n,
            total_rel_loss / n,
            total_cluster_loss / n,
            rel_acc,
            cluster_acc,
        )

    def save_checkpoint(self, epoch: int, metrics: Dict[str, Any] = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics or {},
        }
        path = self.checkpoint_dir / f"ckpt_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        return str(path)

    def load_checkpoint(self, epoch: int):
        """Load model from checkpoint."""
        path = self.checkpoint_dir / f"ckpt_epoch_{epoch}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint.get('metrics', {})

    def load_best_model(self, metric_name: str = "val_loss"):
        """Find and load the checkpoint with best metric."""
        checkpoints = sorted(self.checkpoint_dir.glob("ckpt_epoch_*.pt"))
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found")

        best_epoch = None
        best_value = float('inf') if 'loss' in metric_name else 0.0

        for ckpt in checkpoints:
            checkpoint = torch.load(ckpt, map_location=self.device)
            metrics = checkpoint.get('metrics', {})
            value = metrics.get(metric_name, best_value)

            if 'loss' in metric_name:
                if value < best_value:
                    best_value = value
                    best_epoch = checkpoint['epoch']
            else:
                if value > best_value:
                    best_value = value
                    best_epoch = checkpoint['epoch']

        if best_epoch is not None:
            self.load_checkpoint(best_epoch)
            return best_epoch, best_value
        return None, None
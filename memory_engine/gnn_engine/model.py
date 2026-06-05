"""
gnn_engine/model.py

GraphSAGE model for memory relevance scoring and training.

Takes node feature vectors and graph edges as input.
Outputs enriched node embeddings (h_i') and a relevance score (r_i)
for each memory node.

Input node features (per memory):
  - embedding:       384-dim sentence embedding
  - type_onehot:     4-dim  (preference, fact, episode, rule)
  - importance:      1-dim
  - normalised_age:  1-dim
  - access_count:    1-dim
Total input dim: 391

Training:
  - Relevance labels: Did the memory get accessed in the next 5 turns? (0/1)
  - Cluster labels: Which memory type cluster? (4-way classification)
  - Loss: relevance_loss + 0.5 * cluster_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import os
from pathlib import Path


INPUT_DIM  = 391   # embedding(384) + type(4) + importance(1) + age(1) + access(1)
HIDDEN_DIM = 256
OUTPUT_DIM = 128   # enriched embedding dim
NUM_CLUSTERS = 4   # 4 memory types: preference, fact, episode, rule


class MemoryGNN(nn.Module):
    """
    GraphSAGE-based GNN for memory relevance and cluster prediction.
    """

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM,
                 output_dim: int = OUTPUT_DIM, num_clusters: int = NUM_CLUSTERS):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_clusters = num_clusters

        # Two GraphSAGE layers for message passing
        self.conv1 = SAGEConv(input_dim, hidden_dim, aggr='mean')
        self.conv2 = SAGEConv(hidden_dim, output_dim, aggr='mean')

        # Relevance head: scalar score per node (0-1)
        # Predicts: "Is this memory relevant for the next query?"
        self.relevance_head = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Cluster head: 4-way classification for memory type
        # Soft-predicts memory type based on neighborhood aggregation
        self.cluster_head = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_clusters),
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Forward pass: aggregate neighborhood info and produce scores.

        Args:
            x:          Node feature matrix  [N, INPUT_DIM]
            edge_index: Edge index tensor     [2, E]

        Returns:
            h:  Enriched node embeddings     [N, OUTPUT_DIM]
            r:  Relevance scores             [N, 1]
            c:  Cluster logits               [N, NUM_CLUSTERS]
        """
        # Layer 1: initial aggregation
        h = self.conv1(x, edge_index)
        h = self.relu(h)
        h = self.dropout(h)

        # Layer 2: refined aggregation
        h = self.conv2(h, edge_index)
        h = self.relu(h)

        # Dual heads for dual supervision
        r = self.relevance_head(h)      # [N, 1]
        c = self.cluster_head(h)        # [N, NUM_CLUSTERS]

        return h, r, c

    def loss(self, h: torch.Tensor, r: torch.Tensor, c: torch.Tensor,
             relevance_labels: torch.Tensor, cluster_labels: torch.Tensor,
             alpha: float = 0.5):
        """
        Compute combined loss for dual supervision.

        Args:
            h: enriched embeddings [N, OUTPUT_DIM]
            r: relevance scores [N, 1]
            c: cluster logits [N, NUM_CLUSTERS]
            relevance_labels: binary labels [N]
            cluster_labels: categorical labels [N]
            alpha: weight for cluster loss

        Returns:
            total_loss, rel_loss, cluster_loss
        """
        # Relevance: binary cross-entropy
        relevance_loss = F.binary_cross_entropy(
            r.squeeze(), relevance_labels.float()
        )

        # Cluster: cross-entropy for type prediction
        cluster_loss = F.cross_entropy(c, cluster_labels)

        # Combined
        total_loss = relevance_loss + alpha * cluster_loss

        return total_loss, relevance_loss, cluster_loss


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

    def train_epoch(self, data_list: list[Data], alpha: float = 0.5):
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
    def evaluate(self, data_list: list[Data], alpha: float = 0.5):
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

    def save_checkpoint(self, epoch: int, metrics: dict = None):
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
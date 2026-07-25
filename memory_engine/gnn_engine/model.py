"""
gnn_engine/model.py

GraphSAGE model for memory relevance scoring and training.

Takes node feature vectors and graph edges as input.
Outputs enriched node embeddings (h_i') and a relevance score (r_i)
for each memory node.

Input node features (per memory):
  - embedding:       384-dim sentence embedding
  - type_onehot:     5-dim  (preference, fact, episode, rule, planning)
  - importance:      1-dim
  - normalised_age:  1-dim
  - access_count:    1-dim
Total input dim: 392

Training:
  - Relevance labels: Did the memory get accessed in the next 5 turns? (0/1)
  - Cluster labels: Which memory type cluster? (5-way classification)
  - Loss: relevance_loss + 0.5 * cluster_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import os
from pathlib import Path

from .trainer import GNNTrainer


INPUT_DIM  = 392   # embedding(384) + type(5) + importance(1) + age(1) + access(1)
HIDDEN_DIM = 256
OUTPUT_DIM = 128   # enriched embedding dim
NUM_CLUSTERS = 5   # 5 memory types: preference, fact, episode, rule, planning


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

        # Cluster head: 5-way classification for memory type
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
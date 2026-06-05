"""
gnn_engine/model.py

GraphSAGE model for memory relevance scoring.

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
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


INPUT_DIM  = 391   # embedding(384) + type(4) + importance(1) + age(1) + access(1)
HIDDEN_DIM = 256
OUTPUT_DIM = 128   # enriched embedding dim


class MemoryGNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Two GraphSAGE layers for message passing
        self.conv1 = SAGEConv(INPUT_DIM,  HIDDEN_DIM)
        self.conv2 = SAGEConv(HIDDEN_DIM, OUTPUT_DIM)

        # Relevance head: scalar score per node
        self.relevance_head = nn.Sequential(
            nn.Linear(OUTPUT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Args:
            x:          Node feature matrix  [N, INPUT_DIM]
            edge_index: Edge index tensor     [2, E]

        Returns:
            h:  Enriched node embeddings     [N, OUTPUT_DIM]
            r:  Relevance scores             [N, 1]
        """
        h = self.conv1(x, edge_index)
        h = self.relu(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)
        h = self.relu(h)

        r = self.relevance_head(h)

        return h, r
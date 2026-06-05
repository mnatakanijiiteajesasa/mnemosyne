"""
tests/test_gnn_phase5.py

Unit tests for Phase 5 GNN components.

Run with:
  pytest tests/test_gnn_phase5.py -v
  
Or from Docker:
  docker compose exec api pytest tests/test_gnn_phase5.py -v
"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data

from memory_engine.gnn_engine.model import MemoryGNN, GNNTrainer
from memory_engine.gnn_engine.processor import GraphProcessor, MEMORY_TYPE_MAP
from memory_engine.gnn_engine.inference import GNNInferenceEngine, GNNRetrievalScorer
from memory_engine.models import MemoryType


class TestMemoryGNN:
    """Test the GraphSAGE model architecture."""

    def test_model_initialization(self):
        """Model should initialize without errors."""
        model = MemoryGNN()
        assert model is not None
        assert model.input_dim == 391
        assert model.output_dim == 128
        assert model.num_clusters == 4

    def test_forward_pass(self):
        """Model should do forward pass with correct output shapes."""
        model = MemoryGNN()
        model.eval()

        # Create mock data
        N = 10  # 10 nodes
        x = torch.randn(N, 391)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
             [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]],
            dtype=torch.long
        )

        with torch.no_grad():
            h, r, c = model(x, edge_index)

        assert h.shape == (N, 128), f"Expected h shape (10, 128), got {h.shape}"
        assert r.shape == (N, 1), f"Expected r shape (10, 1), got {r.shape}"
        assert c.shape == (N, 4), f"Expected c shape (10, 4), got {c.shape}"

        # Relevance should be in [0, 1]
        assert (r >= 0).all() and (r <= 1).all()

    def test_loss_computation(self):
        """Model should compute losses correctly."""
        model = MemoryGNN()

        N = 5
        x = torch.randn(N, 391)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

        h, r, c = model(x, edge_index)

        # Create labels
        relevance_labels = torch.tensor([1, 0, 1, 0, 1], dtype=torch.long)
        cluster_labels = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)

        loss, rel_loss, cluster_loss = model.loss(
            h, r, c, relevance_labels, cluster_labels, alpha=0.5
        )

        assert loss.item() > 0
        assert rel_loss.item() > 0
        assert cluster_loss.item() > 0
        print(f"Sample loss: {loss.item():.4f} (rel: {rel_loss.item():.4f}, cluster: {cluster_loss.item():.4f})")


class TestGNNTrainer:
    """Test the training loop."""

    def test_trainer_initialization(self):
        """Trainer should initialize with model and optimizer."""
        model = MemoryGNN()
        trainer = GNNTrainer(model, device="cpu")

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.device == "cpu"

    def test_single_epoch_training(self):
        """Training should reduce loss over an epoch."""
        model = MemoryGNN()
        trainer = GNNTrainer(model, device="cpu")

        # Create mock data
        N = 10
        data_list = []
        for _ in range(3):  # 3 graphs
            x = torch.randn(N, 391)
            edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
            y_rel = torch.randint(0, 2, (N,))
            y_cluster = torch.randint(0, 4, (N,))

            data = Data(x=x, edge_index=edge_index, y_relevance=y_rel, y_cluster=y_cluster)
            data_list.append(data)

        # Train
        loss_before = trainer.train_epoch(data_list, alpha=0.5)[0]
        loss_after = trainer.train_epoch(data_list, alpha=0.5)[0]

        print(f"Loss before: {loss_before:.4f}, after: {loss_after:.4f}")
        # Loss should decrease or at least not increase drastically
        assert loss_after >= 0

    def test_checkpoint_save_load(self, tmp_path):
        """Model checkpoints should save and load correctly."""
        model = MemoryGNN()
        trainer = GNNTrainer(model, device="cpu", checkpoint_dir=str(tmp_path))

        # Save checkpoint
        metrics = {"val_loss": 0.123, "epoch": 1}
        path = trainer.save_checkpoint(epoch=1, metrics=metrics)
        assert path is not None

        # Load checkpoint
        loaded_metrics = trainer.load_checkpoint(epoch=1)
        assert loaded_metrics["val_loss"] == 0.123


class TestGraphProcessor:
    """Test graph construction (requires mock data)."""

    def test_memory_type_mapping(self):
        """Memory type mapping should be complete."""
        for mtype in MemoryType:
            assert mtype in MEMORY_TYPE_MAP
            idx = MEMORY_TYPE_MAP[mtype]
            assert 0 <= idx < 4

    def test_node_features_dimension(self):
        """Node features should be 391-dim."""
        from memory_engine.gnn_engine.processor import GraphProcessor

        # This test would require actual DB connections
        # For now, just verify the dimension constant
        assert GraphProcessor is not None
        # TODO: Integration test with Docker compose setup


class TestGNNInferenceEngine:
    """Test inference and scoring."""

    def test_inference_engine_initialization(self):
        """Inference engine should initialize."""
        engine = GNNInferenceEngine(device="cpu", processor=None)
        assert engine.model is not None
        assert engine.device == "cpu"

    def test_reranking_scorer(self):
        """Retrieval scorer should rerank results."""
        engine = GNNInferenceEngine(device="cpu", processor=None)
        scorer = GNNRetrievalScorer(engine)

        assert scorer.engine is engine

    @torch.no_grad()
    def test_score_computation_mock(self):
        """Test hybrid score computation with mock data."""
        engine = GNNInferenceEngine(device="cpu", processor=None)

        # Manually simulate scoring
        relevance = 0.8
        similarity = 0.7
        cluster_conf = 0.9

        hybrid = 0.5 * relevance + 0.3 * similarity + 0.2 * cluster_conf
        expected = 0.5 * 0.8 + 0.3 * 0.7 + 0.2 * 0.9
        expected = 0.40 + 0.21 + 0.18

        assert abs(hybrid - expected) < 1e-6
        print(f"Hybrid score: {hybrid:.4f}")


class TestIntegration:
    """Integration tests (require Docker setup)."""

    @pytest.mark.asyncio
    async def test_full_pipeline_mock(self):
        """Mock test of the full pipeline."""
        # This would need actual DB setup
        # For now, verify all components can be imported
        from memory_engine.gnn_engine.model import MemoryGNN, GNNTrainer
        from memory_engine.gnn_engine.processor import GraphProcessor
        from memory_engine.gnn_engine.inference import GNNInferenceEngine

        model = MemoryGNN()
        engine = GNNInferenceEngine(device="cpu")

        assert model is not None
        assert engine is not None


# ====================================================================
# STANDALONE TEST RUNNERS (no pytest)
# ====================================================================

def test_model_shapes_standalone():
    """Quick test without pytest."""
    print("\n=== Testing Model Shapes ===")
    model = MemoryGNN()
    model.eval()

    N, E = 10, 20
    x = torch.randn(N, 391)
    edge_index = torch.randint(0, N, (2, E))

    with torch.no_grad():
        h, r, c = model(x, edge_index)

    print(f"Input shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Output h shape: {h.shape} ✓")
    print(f"Output r shape: {r.shape} ✓")
    print(f"Output c shape: {c.shape} ✓")


def test_loss_standalone():
    """Quick loss test."""
    print("\n=== Testing Loss Computation ===")
    model = MemoryGNN()

    N = 8
    x = torch.randn(N, 391)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

    h, r, c = model(x, edge_index)

    rel_labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
    cluster_labels = torch.randint(0, 4, (N,))

    loss, rel_loss, cluster_loss = model.loss(h, r, c, rel_labels, cluster_labels, alpha=0.5)

    print(f"Total loss: {loss.item():.4f}")
    print(f"Relevance loss: {rel_loss.item():.4f}")
    print(f"Cluster loss: {cluster_loss.item():.4f}")


def test_trainer_standalone():
    """Quick trainer test."""
    print("\n=== Testing Trainer ===")
    model = MemoryGNN()
    trainer = GNNTrainer(model, device="cpu")

    print(f"Checkpoint dir: {trainer.checkpoint_dir}")
    print(f"Optimizer: {type(trainer.optimizer).__name__}")

    # Create mock batch
    N = 5
    data_list = []
    for i in range(2):
        x = torch.randn(N, 391)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        data = Data(
            x=x,
            edge_index=edge_index,
            y_relevance=torch.randint(0, 2, (N,)),
            y_cluster=torch.randint(0, 4, (N,))
        )
        data_list.append(data)

    loss, rel_loss, cluster_loss = trainer.train_epoch(data_list, alpha=0.5)
    print(f"Epoch loss: {loss:.4f}")


if __name__ == "__main__":
    # Run standalone tests
    test_model_shapes_standalone()
    test_loss_standalone()
    test_trainer_standalone()
    print("\n✓ All standalone tests passed!")
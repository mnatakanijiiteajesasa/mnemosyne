import asyncio
  import torch
  from pathlib import Path
 
  from memory_engine.gnn_engine.model import GNNTrainer, MemoryGNN
  from memory_engine.gnn_engine.processor import GraphProcessor
 
 
  async def train():
      device = "cuda" if torch.cuda.is_available() else "cpu"
      print(f"Training on device: {device}")
 
      mongo_url  = "mongodb://agent:agent@mongo:27017/memories?authSource=admin"
      qdrant_url = "http://qdrant:6333"
 
      processor = GraphProcessor(mongo_url, qdrant_url)
      model = MemoryGNN().to(device)
      trainer = GNNTrainer(model, device=device)
 
      # Training loop (mock data for now)
      # TODO: In Phase 8, fetch real training data from access logs
      print("GNN model initialized. Ready for training.")
 
      # Save initial checkpoint
      trainer.save_checkpoint(0, metrics={"status": "initialized"})
      print(f"Checkpoint saved to {trainer.checkpoint_dir}")
 
 
  if __name__ == "__main__":
      asyncio.run(train())
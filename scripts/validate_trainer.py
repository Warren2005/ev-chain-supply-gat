"""
Validation script for ST-GAT Trainer

Quick test with real data to verify training loop works.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from models.st_gat import STGAT
from utils.temporal_dataset import TemporalGraphDataset
from utils.graph_dataloader import TemporalGraphDataLoader
from utils.trainer import STGATTrainer


def main():
    """Validate trainer with real data"""
    print("="*70)
    print("ST-GAT TRAINER - VALIDATION TEST")
    print("="*70)
    print()
    
    # Load datasets
    data_dir = project_root / "data" / "processed"
    
    edge_index = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 4, 4, 4, 0], 
                               [6, 2, 3, 4, 1, 2, 3, 6, 6, 2, 3, 1]], 
                              dtype=torch.long)
    
    print("Loading datasets...")
    train_dataset = TemporalGraphDataset(
        data_path=str(data_dir / "train_features_filtered.parquet"),
        edge_index=edge_index,
        window_size=20
    )
    
    val_dataset = TemporalGraphDataset(
        data_path=str(data_dir / "val_features_filtered.parquet"),
        edge_index=edge_index,
        window_size=20
    )
    
    print(f"✓ Train: {len(train_dataset)} samples")
    print(f"✓ Val: {len(val_dataset)} samples")
    print()
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, _ = TemporalGraphDataLoader.create_train_val_test_loaders(
        train_dataset,
        val_dataset,
        val_dataset,
        batch_size=16,
        num_workers=0
    )
    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Val loader: {len(val_loader)} batches")
    print()
    
    # Create model
    print("Initializing ST-GAT model...")
    model = STGAT(
        num_nodes=7,
        num_edges=12,
        input_features=15,
        gat_hidden_dim=128,
        gat_heads=8,
        gat_layers=2,
        lstm_hidden_dim=128,
        lstm_layers=2,
        temporal_window=20,
        device='cpu'
    )
    print("✓ Model created")
    print()
    
    # Create trainer
    print("Initializing trainer...")
    trainer = STGATTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        weight_decay=1e-5,
        gradient_clip_val=1.0,
        early_stopping_patience=10,
        checkpoint_dir="checkpoints/validation_test",
        device='cpu'
    )
    print("✓ Trainer initialized")
    print()
    
    # Train for 3 epochs (quick test)
    print("="*70)
    print("TRAINING TEST (3 epochs)")
    print("="*70)
    
    history = trainer.train(num_epochs=3, verbose=True)
    
    print()
    print("="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    summary = trainer.get_training_summary()
    
    print(f"\nTraining completed successfully!")
    print(f"  Total epochs: {summary['total_epochs']}")
    print(f"  Best val loss: {summary['best_val_loss']:.6f}")
    print(f"  Final train loss: {summary['final_train_loss']:.6f}")
    print(f"  Final val loss: {summary['final_val_loss']:.6f}")
    print(f"  Avg epoch time: {summary['avg_epoch_time']:.2f}s")
    print()
    
    print("✓ ALL TRAINER TESTS PASSED!")
    print()
    print("="*70)
    print("✓ PHASE 4 STEP 3 COMPLETE!")
    print("="*70)
    print()
    print("Ready for Phase 4 Step 4: Full Training Run")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
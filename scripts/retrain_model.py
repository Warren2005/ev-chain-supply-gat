"""
Retrain ST-GAT with Fixed Data and Better Hyperparameters

This script retrains the model with:
1. Fixed preprocessed data (outliers clipped, log-transformed)
2. Lower learning rate (1e-4 instead of 1e-3)
3. Learning rate scheduler (ReduceLROnPlateau)
4. Better monitoring to prevent collapse
5. UPDATED: 15 edges with SQM properly connected

Usage:
    python scripts/retrain_model.py

Author: EV Supply Chain GAT Team
Date: December 2024
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import json
from datetime import datetime

from models.st_gat import STGAT
from utils.temporal_dataset import TemporalGraphDataset
from utils.graph_dataloader import TemporalGraphDataLoader
from utils.trainer import STGATTrainer


def plot_comparison(
    old_history: dict,
    new_history: dict,
    save_path: Path
):
    """
    Plot comparison between old (collapsed) and new (fixed) training.
    
    Args:
        old_history: Training history from collapsed model
        new_history: Training history from retrained model
        save_path: Path to save comparison plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss comparison
    ax = axes[0, 0]
    old_epochs = range(1, len(old_history['train_losses']) + 1)
    new_epochs = range(1, len(new_history['train_losses']) + 1)
    
    ax.plot(old_epochs, old_history['train_losses'], 'r-', 
            label='Old (Collapsed)', linewidth=2, alpha=0.7)
    ax.plot(new_epochs, new_history['train_losses'], 'b-', 
            label='New (Fixed)', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation loss comparison
    ax = axes[0, 1]
    ax.plot(old_epochs, old_history['val_losses'], 'r-', 
            label='Old (Collapsed)', linewidth=2, alpha=0.7)
    ax.plot(new_epochs, new_history['val_losses'], 'b-', 
            label='New (Fixed)', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss reduction per epoch (new model)
    ax = axes[1, 0]
    train_changes = [new_history['train_losses'][0] - loss 
                     for loss in new_history['train_losses']]
    val_changes = [new_history['val_losses'][0] - loss 
                   for loss in new_history['val_losses']]
    
    ax.plot(new_epochs, train_changes, 'b-', label='Train Reduction', linewidth=2)
    ax.plot(new_epochs, val_changes, 'g-', label='Val Reduction', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Reduction from Initial')
    ax.set_title('New Model: Learning Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Comparison stats
    ax = axes[1, 1]
    ax.axis('off')
    
    old_train_reduction = (old_history['train_losses'][0] - old_history['train_losses'][-1])/old_history['train_losses'][0]*100
    new_train_reduction = (new_history['train_losses'][0] - new_history['train_losses'][-1])/new_history['train_losses'][0]*100
    
    stats_text = f"""
    TRAINING COMPARISON
    
    Old Model (Collapsed):
    • Initial train loss: {old_history['train_losses'][0]:.4f}
    • Final train loss: {old_history['train_losses'][-1]:.4f}
    • Reduction: {old_train_reduction:.1f}%
    • Best val loss: {min(old_history['val_losses']):.4f}
    
    New Model (Fixed):
    • Initial train loss: {new_history['train_losses'][0]:.4f}
    • Final train loss: {new_history['train_losses'][-1]:.4f}
    • Reduction: {new_train_reduction:.1f}%
    • Best val loss: {min(new_history['val_losses']):.4f}
    
    Improvement:
    • Train loss reduction: {new_train_reduction - old_train_reduction:.1f}% better
    • Best val loss: {(min(old_history['val_losses']) - min(new_history['val_losses'])):.4f} better
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {save_path}")


def main():
    """Main retraining function"""
    print("="*70)
    print("RETRAINING ST-GAT WITH FIXED DATA")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4  # Lower than before!
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP = 5.0  # Increased from 1.0
    PATIENCE = 25  # Increased patience
    MAX_EPOCHS = 150
    DEVICE = 'cpu'
    
    # Checkpoint and results directories
    checkpoint_dir = Path("checkpoints/retrained")
    results_dir = Path("results/retrained")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Stock order and edge index (UPDATED: 15 edges with SQM connected)
    stock_names = ['ALB', 'APTV', 'F', 'GM', 'MGA', 'SQM', 'TSLA']
    edge_index = torch.tensor([
        [0, 0, 0, 4, 4, 4, 1, 1, 1, 0, 0, 4, 5, 5, 5],
        [6, 2, 3, 6, 2, 3, 2, 3, 6, 4, 1, 1, 6, 2, 3]
    ], dtype=torch.long)
    
    print(f"Graph structure: 7 nodes, 15 edges (SQM now connected)")
    print()
    
    # Load FIXED datasets
    print("Loading fixed datasets...")
    data_dir = Path("data/processed")
    
    train_dataset = TemporalGraphDataset(
        data_path=str(data_dir / "train_features_fixed.parquet"),
        edge_index=edge_index,
        window_size=20
    )
    
    val_dataset = TemporalGraphDataset(
        data_path=str(data_dir / "val_features_fixed.parquet"),
        edge_index=edge_index,
        window_size=20
    )
    
    test_dataset = TemporalGraphDataset(
        data_path=str(data_dir / "test_features_fixed.parquet"),
        edge_index=edge_index,
        window_size=20
    )
    
    print(f"✓ Train: {len(train_dataset)} samples")
    print(f"✓ Val: {len(val_dataset)} samples")
    print(f"✓ Test: {len(test_dataset)} samples")
    print()
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = TemporalGraphDataLoader.create_train_val_test_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    
    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Val loader: {len(val_loader)} batches")
    print()
    
    # Create fresh model
    print("Initializing ST-GAT model...")
    model = STGAT(
        num_nodes=7,
        num_edges=15,  # UPDATED from 12 to 15
        input_features=15,
        gat_hidden_dim=128,
        gat_heads=8,
        gat_layers=2,
        lstm_hidden_dim=128,
        lstm_layers=2,
        temporal_window=20,
        device=DEVICE
    )
    
    print(f"✓ Model created with {model.model_stats['total_parameters']:,} parameters")
    print()
    
    # Create trainer with BETTER hyperparameters
    print("Initializing trainer with improved settings...")
    print(f"  Learning rate: {LEARNING_RATE} (was 1e-3)")
    print(f"  Gradient clip: {GRADIENT_CLIP} (was 1.0)")
    print(f"  Patience: {PATIENCE} (was 20)")
    print()
    
    trainer = STGATTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        gradient_clip_val=GRADIENT_CLIP,
        early_stopping_patience=PATIENCE,
        checkpoint_dir=str(checkpoint_dir),
        device=DEVICE
    )
    
    # Add learning rate scheduler
    scheduler = ReduceLROnPlateau(
        trainer.optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )
    
    print("✓ Added ReduceLROnPlateau scheduler")
    print()
    
    # Custom training loop with scheduler
    print("="*70)
    print(f"STARTING RETRAINING - Up to {MAX_EPOCHS} epochs")
    print("="*70)
    print()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    epoch_times = []
    learning_rates = []
    
    import time
    
    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        
        # Train one epoch
        train_loss = trainer.train_epoch()
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_metrics = trainer.validate_epoch()
        val_losses.append(val_loss)
        
        # Track epoch time
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = trainer.optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Check for best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            trainer.epochs_without_improvement = 0
        else:
            trainer.epochs_without_improvement += 1
        
        # Save checkpoint
        trainer.save_checkpoint(epoch + 1, val_loss, is_best)
        
        # Print progress
        print(f"Epoch {epoch+1}/{MAX_EPOCHS}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Val RMSE:   {val_metrics['rmse']:.6f}")
        print(f"  Val R²:     {val_metrics['r2']:.4f}")
        print(f"  LR:         {current_lr:.2e}")
        print(f"  Time:       {epoch_time:.2f}s")
        if is_best:
            print("  ✓ New best model!")
        print()
        
        # Early stopping check
        if trainer.epochs_without_improvement >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            print(f"No improvement for {PATIENCE} epochs")
            break
    
    # Training complete
    print("="*70)
    print("RETRAINING COMPLETE")
    print("="*70)
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Total epochs: {len(train_losses)}")
    print(f"Final learning rate: {learning_rates[-1]:.2e}")
    print()
    
    # Save training history
    new_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epoch_times': epoch_times,
        'learning_rates': learning_rates,
        'best_val_loss': best_val_loss
    }
    
    with open(results_dir / 'retrain_history.json', 'w') as f:
        json.dump({k: [float(x) for x in v] if isinstance(v, list) else float(v)
                   for k, v in new_history.items()}, f, indent=2)
    
    print(f"✓ Training history saved to: {results_dir / 'retrain_history.json'}")
    
    # Load old training history for comparison
    with open('results/training/training_report.json', 'r') as f:
        old_report = json.load(f)
    
    old_history = {
        'train_losses': old_report['history']['train_losses'],
        'val_losses': old_report['history']['val_losses']
    }
    
    # Plot comparison
    print("\nGenerating comparison visualizations...")
    plot_comparison(old_history, new_history, results_dir / 'training_comparison.png')
    
    print()
    print("="*70)
    print("✓ RETRAINING SUCCESSFUL!")
    print("="*70)
    print(f"\nResults saved to: {results_dir}")
    print(f"Best model: {checkpoint_dir / 'best_model.pt'}")
    print()
    print("Next steps:")
    print("  1. Evaluate retrained model")
    print("  2. Compare old vs new performance")
    print("  3. Generate improved 3D visualizations")
    print()


if __name__ == "__main__":
    main()
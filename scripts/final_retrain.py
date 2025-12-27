"""
Final Retraining with Simplified Architecture and Robust Loss

This is the definitive training attempt with:
1. Simplified ST-GAT (64 hidden, 1 layer each)
2. Huber loss (robust to outliers)
3. Higher regularization
4. Lower learning rate
5. Residual connections

Author: EV Supply Chain GAT Team
Date: December 2024
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import json
from datetime import datetime

from models.simplified_st_gat import SimplifiedSTGAT
from utils.temporal_dataset import TemporalGraphDataset
from utils.graph_dataloader import TemporalGraphDataLoader
from utils.robust_trainer import RobustTrainer


def plot_all_training_attempts(save_path: Path):
    """
    Plot comparison of all three training attempts.
    
    Shows progression:
    1. Original (collapsed completely)
    2. First fix (still collapsed but better)
    3. Final (simplified architecture + Huber loss)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Load all training histories
    with open('results/training/training_report.json', 'r') as f:
        attempt1 = json.load(f)
    
    with open('results/retrained/retrain_history.json', 'r') as f:
        attempt2 = json.load(f)
    
    with open('results/final_retrain/train_history.json', 'r') as f:
        attempt3 = json.load(f)
    
    # Training loss comparison
    ax = axes[0, 0]
    ax.plot(attempt1['history']['train_losses'], 'r-', label='Attempt 1 (Collapsed)', alpha=0.7)
    ax.plot(attempt2['train_losses'], 'orange', label='Attempt 2 (Better)', alpha=0.7)
    ax.plot(attempt3['train_losses'], 'b-', label='Attempt 3 (Final)', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss - All Attempts')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation loss comparison
    ax = axes[0, 1]
    ax.plot(attempt1['history']['val_losses'], 'r-', label='Attempt 1', alpha=0.7)
    ax.plot(attempt2['val_losses'], 'orange', label='Attempt 2', alpha=0.7)
    ax.plot(attempt3['val_losses'], 'b-', label='Attempt 3', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss - All Attempts')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss reduction percentage
    ax = axes[0, 2]
    reductions = [
        (attempt1['history']['train_losses'][0] - attempt1['history']['train_losses'][-1]) / attempt1['history']['train_losses'][0] * 100,
        (attempt2['train_losses'][0] - attempt2['train_losses'][-1]) / attempt2['train_losses'][0] * 100,
        (attempt3['train_losses'][0] - attempt3['train_losses'][-1]) / attempt3['train_losses'][0] * 100
    ]
    ax.bar(['Attempt 1', 'Attempt 2', 'Attempt 3'], reductions, color=['red', 'orange', 'blue'])
    ax.set_ylabel('Training Loss Reduction (%)')
    ax.set_title('Learning Effectiveness')
    ax.axhline(y=5, color='g', linestyle='--', label='5% threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Best validation loss comparison
    ax = axes[1, 0]
    best_vals = [
        min(attempt1['history']['val_losses']),
        min(attempt2['val_losses']),
        min(attempt3['val_losses'])
    ]
    ax.bar(['Attempt 1', 'Attempt 2', 'Attempt 3'], best_vals, color=['red', 'orange', 'blue'])
    ax.set_ylabel('Best Validation Loss')
    ax.set_title('Best Performance Achieved')
    ax.grid(True, alpha=0.3)
    
    # Learning rate evolution (attempt 3)
    ax = axes[1, 1]
    if 'learning_rates' in attempt3:
        ax.plot(attempt3['learning_rates'], 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Attempt 3: Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"""
    TRAINING EVOLUTION SUMMARY
    
    Attempt 1 (Original - MSE Loss):
    • Architecture: 128 hidden, 2 layers
    • Train reduction: {reductions[0]:.1f}%
    • Best val loss: {best_vals[0]:.4f}
    • Result: Complete collapse
    
    Attempt 2 (Fixed Data):
    • Architecture: 128 hidden, 2 layers
    • Train reduction: {reductions[1]:.1f}%
    • Best val loss: {best_vals[1]:.4f}
    • Result: Partial collapse
    
    Attempt 3 (Simplified + Huber):
    • Architecture: 64 hidden, 1 layer
    • Train reduction: {reductions[2]:.1f}%
    • Best val loss: {best_vals[2]:.4f}
    • Result: TBD
    
    Improvement from Attempt 1 to 3:
    • Train learning: +{reductions[2] - reductions[0]:.1f}%
    • Val loss: {(best_vals[0] - best_vals[2]) / best_vals[0] * 100:.1f}% better
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {save_path}")


def main():
    """Final retraining with simplified architecture"""
    print("="*70)
    print("FINAL RETRAINING - SIMPLIFIED ST-GAT WITH HUBER LOSS")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-5  # Even lower!
    WEIGHT_DECAY = 1e-4   # Higher regularization
    GRADIENT_CLIP = 1.0
    PATIENCE = 30
    MAX_EPOCHS = 100
    HUBER_DELTA = 1.0
    DEVICE = 'cpu'
    
    # Directories
    checkpoint_dir = Path("checkpoints/final_retrain")
    results_dir = Path("results/final_retrain")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Stock info
    stock_names = ['ALB', 'APTV', 'F', 'GM', 'MGA', 'SQM', 'TSLA']
    edge_index = torch.tensor([
        [0, 0, 0, 4, 4, 4, 1, 1, 1, 0, 0, 4, 5, 5, 5],
        [6, 2, 3, 6, 2, 3, 2, 3, 6, 4, 1, 1, 6, 2, 3]
    ], dtype=torch.long)
    
    print("Key Changes from Previous Attempts:")
    print("  • Simplified architecture: 64 hidden (was 128)")
    print("  • Single GAT layer (was 2)")
    print("  • Single LSTM layer (was 2)")
    print("  • 4 attention heads (was 8)")
    print("  • Huber loss (was MSE)")
    print("  • Residual connections added")
    print("  • Batch normalization added")
    print("  • Higher dropout: 0.3 (was 0.2)")
    print("  • Lower LR: 5e-5 (was 1e-4)")
    print("  • Higher weight decay: 1e-4 (was 1e-5)")
    print()
    
    # Load datasets
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
        train_dataset, val_dataset, test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    print(f"✓ Train: {len(train_loader)} batches")
    print(f"✓ Val: {len(val_loader)} batches")
    print()
    
    # Create SIMPLIFIED model
    print("Initializing Simplified ST-GAT model...")
    model = SimplifiedSTGAT(
        num_nodes=7,
        num_edges=15,
        input_features=15,
        gat_hidden_dim=64,  # Reduced!
        gat_heads=4,        # Reduced!
        lstm_hidden_dim=64, # Reduced!
        temporal_window=20,
        output_dim=1,
        dropout=0.3,        # Increased!
        device=DEVICE
    )
    print()
    
    # Create robust trainer
    print("Initializing Robust Trainer with Huber Loss...")
    trainer = RobustTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        gradient_clip_val=GRADIENT_CLIP,
        early_stopping_patience=PATIENCE,
        checkpoint_dir=str(checkpoint_dir),
        device=DEVICE,
        huber_delta=HUBER_DELTA
    )
    print()
    
    # Add LR scheduler
    scheduler = ReduceLROnPlateau(
        trainer.optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )
    print("✓ Added ReduceLROnPlateau scheduler")
    print()
    
    # Train with custom loop (for scheduler)
    print("="*70)
    print(f"STARTING FINAL TRAINING - Up to {MAX_EPOCHS} epochs")
    print("="*70)
    print()
    
    import time
    
    train_losses = []
    val_losses = []
    epoch_times = []
    learning_rates = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_loss = trainer.train_epoch()
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_metrics = trainer.validate_epoch()
        val_losses.append(val_loss)
        
        # Track time and LR
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        scheduler.step(val_loss)
        current_lr = trainer.optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Check for best
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Save
        trainer.save_checkpoint(epoch + 1, val_loss, is_best)
        
        # Print
        print(f"Epoch {epoch+1}/{MAX_EPOCHS}")
        print(f"  Train Loss:  {train_loss:.6f}")
        print(f"  Val Loss:    {val_loss:.6f}")
        print(f"  Val RMSE:    {val_metrics['rmse']:.6f}")
        print(f"  Val R²:      {val_metrics['r2']:.4f}")
        print(f"  Val Dir Acc: {val_metrics['directional_accuracy']*100:.2f}%")
        print(f"  LR:          {current_lr:.2e}")
        print(f"  Time:        {epoch_time:.2f}s")
        if is_best:
            print("  ✓ New best model!")
        print()
        
        # Early stopping
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Training complete
    print("="*70)
    print("FINAL TRAINING COMPLETE")
    print("="*70)
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Total epochs: {len(train_losses)}")
    print(f"Train loss reduction: {(train_losses[0] - train_losses[-1]) / train_losses[0] * 100:.1f}%")
    print()
    
    # Save history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epoch_times': epoch_times,
        'learning_rates': learning_rates,
        'best_val_loss': best_val_loss
    }
    
    with open(results_dir / 'train_history.json', 'w') as f:
        json.dump({k: [float(x) for x in v] if isinstance(v, list) else float(v)
                   for k, v in history.items()}, f, indent=2)
    
    print(f"✓ History saved: {results_dir / 'train_history.json'}")
    
    # Plot comparison with all attempts
    print("\nGenerating comprehensive comparison plots...")
    plot_all_training_attempts(results_dir / 'all_attempts_comparison.png')
    
    print()
    print("="*70)
    print("✓ FINAL RETRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest model: {checkpoint_dir / 'best_model.pt'}")
    print(f"Results: {results_dir}")
    print()
    print("Next step:")
    print("  python scripts/final_evaluation.py")
    print()


if __name__ == "__main__":
    main()
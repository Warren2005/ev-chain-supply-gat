"""
Production Training Script for ST-GAT Model

This script trains the complete ST-GAT model on the full EV supply chain dataset.
It includes comprehensive logging, checkpointing, and visualization.

Usage:
    python scripts/train_st_gat.py --epochs 100 --batch_size 32

Author: EV Supply Chain GAT Team
Date: December 2024
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

from models.st_gat import STGAT
from utils.temporal_dataset import TemporalGraphDataset
from utils.graph_dataloader import TemporalGraphDataLoader
from utils.trainer import STGATTrainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train ST-GAT Model')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for L2 regularization (default: 1e-5)')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping value (default: 1.0)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    
    # Model parameters
    parser.add_argument('--gat_hidden', type=int, default=128,
                        help='GAT hidden dimension (default: 128)')
    parser.add_argument('--gat_heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--gat_layers', type=int, default=2,
                        help='Number of GAT layers (default: 2)')
    parser.add_argument('--lstm_hidden', type=int, default=128,
                        help='LSTM hidden dimension (default: 128)')
    parser.add_argument('--lstm_layers', type=int, default=2,
                        help='Number of LSTM layers (default: 2)')
    
    # Other settings
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to train on (default: cpu)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/production',
                        help='Directory to save checkpoints')
    parser.add_argument('--results_dir', type=str, default='results/training',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def plot_training_curves(history: dict, save_path: Path):
    """
    Plot training and validation loss curves.
    
    Args:
        history: Dictionary with train_losses and val_losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    epochs = range(1, len(history['train_losses']) + 1)
    plt.plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, history['val_losses'], 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Training Progress', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Best epoch marker
    best_epoch = history['val_losses'].index(min(history['val_losses'])) + 1
    plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    
    # Epoch time plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['epoch_times'], 'g-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Training Speed', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training curves saved to: {save_path}")


def save_training_report(args, history: dict, summary: dict, save_path: Path):
    """
    Save comprehensive training report.
    
    Args:
        args: Command line arguments
        history: Training history
        summary: Training summary
        save_path: Path to save the report
    """
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'gradient_clip': args.gradient_clip,
            'early_stopping_patience': args.patience,
            'gat_hidden_dim': args.gat_hidden,
            'gat_heads': args.gat_heads,
            'gat_layers': args.gat_layers,
            'lstm_hidden_dim': args.lstm_hidden,
            'lstm_layers': args.lstm_layers,
            'device': args.device,
            'random_seed': args.seed
        },
        'results': {
            'total_epochs_trained': summary['total_epochs'],
            'best_val_loss': float(summary['best_val_loss']),
            'final_train_loss': float(summary['final_train_loss']),
            'final_val_loss': float(summary['final_val_loss']),
            'avg_epoch_time': float(summary['avg_epoch_time']),
            'total_training_time': float(summary['total_training_time'])
        },
        'history': {
            'train_losses': [float(x) for x in history['train_losses']],
            'val_losses': [float(x) for x in history['val_losses']],
            'epoch_times': [float(x) for x in history['epoch_times']]
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Training report saved to: {save_path}")


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    results_dir = Path(args.results_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("ST-GAT PRODUCTION TRAINING")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {args.device}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Define edge index for 7 stocks (excluding RIVN)
    # Stock order: 0:ALB, 1:APTV, 2:F, 3:GM, 4:MGA, 5:SQM, 6:TSLA
    edge_index = torch.tensor([
        [0, 0, 0, 0, 0, 1, 1, 1, 4, 4, 4, 0],
        [6, 2, 3, 4, 1, 2, 3, 6, 6, 2, 3, 1]
    ], dtype=torch.long)
    
    # Load datasets
    print("Loading datasets...")
    data_dir = project_root / "data" / "processed"
    
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
    
    test_dataset = TemporalGraphDataset(
        data_path=str(data_dir / "test_features_filtered.parquet"),
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
        batch_size=args.batch_size,
        num_workers=0
    )
    
    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Val loader: {len(val_loader)} batches")
    print(f"✓ Test loader: {len(test_loader)} batches")
    print()
    
    # Create model
    print("Initializing ST-GAT model...")
    model = STGAT(
        num_nodes=7,
        num_edges=12,
        input_features=15,
        gat_hidden_dim=args.gat_hidden,
        gat_heads=args.gat_heads,
        gat_layers=args.gat_layers,
        lstm_hidden_dim=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        temporal_window=20,
        device=args.device
    )
    
    print(f"✓ Model created with {model.model_stats['total_parameters']:,} parameters")
    print()
    
    # Create trainer
    print("Initializing trainer...")
    trainer = STGATTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_val=args.gradient_clip,
        early_stopping_patience=args.patience,
        checkpoint_dir=str(checkpoint_dir),
        device=args.device
    )
    
    print("✓ Trainer initialized")
    print()
    
    # Train model
    print("="*70)
    print(f"STARTING TRAINING - {args.epochs} EPOCHS")
    print("="*70)
    print()
    
    history = trainer.train(num_epochs=args.epochs, verbose=True)
    
    # Get training summary
    summary = trainer.get_training_summary()
    
    # Print final results
    print()
    print("="*70)
    print("TRAINING COMPLETE - FINAL RESULTS")
    print("="*70)
    print(f"\nTotal epochs: {summary['total_epochs']}")
    print(f"Best validation loss: {summary['best_val_loss']:.6f}")
    print(f"Final train loss: {summary['final_train_loss']:.6f}")
    print(f"Final val loss: {summary['final_val_loss']:.6f}")
    print(f"Average epoch time: {summary['avg_epoch_time']:.2f}s")
    print(f"Total training time: {summary['total_training_time']/60:.2f} minutes")
    print()
    
    # Save training curves
    print("Generating training visualizations...")
    plot_path = results_dir / "training_curves.png"
    plot_training_curves(history, plot_path)
    
    # Save training report
    report_path = results_dir / "training_report.json"
    save_training_report(args, history, summary, report_path)
    
    print()
    print("="*70)
    print("✓ PRODUCTION TRAINING COMPLETE!")
    print("="*70)
    print(f"\nCheckpoints saved to: {checkpoint_dir}")
    print(f"Results saved to: {results_dir}")
    print(f"\nBest model: {checkpoint_dir / 'best_model.pt'}")
    print()
    print("Next steps:")
    print("  1. Evaluate on test set (Phase 5)")
    print("  2. Analyze attention weights")
    print("  3. Compare with baselines")
    print()


if __name__ == "__main__":
    main()
"""
Optimize Model for Better R² Scores

Systematic optimization to improve R² while maintaining
directional accuracy through:
1. Better loss functions
2. Target preprocessing
3. Architecture tweaks
4. Training strategies

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
import time

from models.simplified_st_gat import SimplifiedSTGAT
from utils.temporal_dataset import TemporalGraphDataset
from utils.graph_dataloader import TemporalGraphDataLoader
from utils.advanced_loss_functions import (
    StockWeightedHuberLoss,
    MultiTaskLoss,
    VariancePreservingLoss,
    AdaptiveHuberLoss,
    compute_stock_weights_from_baseline
)


class ImprovedTrainer:
    """
    Enhanced trainer with advanced loss functions and R² optimization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        loss_fn: nn.Module,
        learning_rate: float = 5e-5,
        weight_decay: float = 1e-4,
        gradient_clip: float = 1.0,
        patience: int = 30,
        device: str = 'cpu'
    ):
        """Initialize improved trainer"""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        self.gradient_clip = gradient_clip
        self.patience = patience
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        self.train_losses = []
        self.val_losses = []
        self.val_r2_scores = []
        self.val_dir_accs = []
    
    def train_epoch(self) -> float:
        """Train one epoch"""
        self.model.train()
        epoch_losses = []
        
        for features, edge_index, targets in self.train_loader:
            features = features.to(self.device)
            edge_index = edge_index.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions = self.model(features, edge_index).squeeze(-1)
            loss = self.loss_fn(predictions, targets)
            
            loss.backward()
            
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.optimizer.step()
            epoch_losses.append(loss.item())
        
        return np.mean(epoch_losses)
    
    def validate_epoch(self):
        """Validate one epoch"""
        self.model.eval()
        epoch_losses = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for features, edge_index, targets in self.val_loader:
                features = features.to(self.device)
                edge_index = edge_index.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(features, edge_index).squeeze(-1)
                loss = self.loss_fn(predictions, targets)
                
                epoch_losses.append(loss.item())
                all_preds.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        metrics = self.compute_metrics(all_preds, all_targets)
        metrics['loss'] = np.mean(epoch_losses)
        
        return metrics
    
    def compute_metrics(self, predictions, targets):
        """Compute evaluation metrics"""
        # R²
        ss_res = ((targets - predictions) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot)
        
        # Directional accuracy
        dir_acc = (torch.sign(predictions) == torch.sign(targets)).float().mean()
        
        # RMSE
        rmse = torch.sqrt(((predictions - targets) ** 2).mean())
        
        return {
            'r2': r2.item(),
            'directional_accuracy': dir_acc.item(),
            'rmse': rmse.item()
        }
    
    def train(self, num_epochs: int, checkpoint_dir: Path):
        """Train the model"""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Loss function: {self.loss_fn.__class__.__name__}")
        print()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_metrics = self.validate_epoch()
            val_loss = val_metrics['loss']
            self.val_losses.append(val_loss)
            self.val_r2_scores.append(val_metrics['r2'])
            self.val_dir_accs.append(val_metrics['directional_accuracy'])
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Check for best
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_r2': val_metrics['r2'],
                    'val_dir_acc': val_metrics['directional_accuracy']
                }, checkpoint_dir / 'best_model.pt')
            else:
                self.epochs_without_improvement += 1
            
            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val R²:     {val_metrics['r2']:.4f}")
            print(f"  Val Dir Acc: {val_metrics['directional_accuracy']*100:.1f}%")
            print(f"  Val RMSE:   {val_metrics['rmse']:.4f}")
            print(f"  LR:         {current_lr:.2e}")
            print(f"  Time:       {epoch_time:.2f}s")
            if is_best:
                print("  ✓ New best model!")
            print()
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_r2_scores': self.val_r2_scores,
            'val_dir_accs': self.val_dir_accs,
            'best_val_loss': self.best_val_loss
        }


def main():
    """Main optimization"""
    print("="*70)
    print("R² SCORE OPTIMIZATION - SYSTEMATIC APPROACH")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup
    stock_names = ['ALB', 'APTV', 'F', 'GM', 'MGA', 'SQM', 'TSLA']
    edge_index = torch.tensor([
        [0, 0, 0, 4, 4, 4, 1, 1, 1, 0, 0, 4, 5, 5, 5],
        [6, 2, 3, 6, 2, 3, 2, 3, 6, 4, 1, 1, 6, 2, 3]
    ], dtype=torch.long)
    
    # Load datasets
    print("Loading datasets...")
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
    
    train_loader, val_loader, test_loader = TemporalGraphDataLoader.create_train_val_test_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=32, num_workers=0
    )
    
    print(f"✓ Train: {len(train_dataset)} samples")
    print(f"✓ Val: {len(val_dataset)} samples")
    print(f"✓ Test: {len(test_dataset)} samples")
    print()
    
    # Compute stock weights
    print("Computing stock-specific parameters...")
    stock_weights = compute_stock_weights_from_baseline(train_dataset, stock_names)
    
    # Compute target statistics for adaptive loss
    all_targets = []
    for _, _, target in train_dataset:
        all_targets.append(target)
    all_targets = torch.stack(all_targets)
    target_stds = all_targets.std(dim=0)
    
    print("Target standard deviations per stock:")
    for stock, std in zip(stock_names, target_stds):
        print(f"  {stock}: {std:.4f}")
    print()
    
    # Try multiple loss functions
    loss_functions = {
        'MultiTask': MultiTaskLoss(direction_weight=0.3, magnitude_weight=0.7),
        'StockWeighted': StockWeightedHuberLoss(stock_weights, delta=1.0),
        'VariancePreserving': VariancePreservingLoss(
            nn.HuberLoss(delta=1.0),
            variance_weight=0.2
        ),
        'AdaptiveHuber': AdaptiveHuberLoss(target_stds)
    }
    
    print("="*70)
    print("TESTING MULTIPLE LOSS FUNCTIONS")
    print("="*70)
    print()
    
    results = {}
    
    for loss_name, loss_fn in loss_functions.items():
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {loss_name} Loss")
        print(f"{'='*70}")
        
        # Create fresh model
        model = SimplifiedSTGAT(
            num_nodes=7,
            num_edges=15,
            input_features=15,
            gat_hidden_dim=64,
            gat_heads=4,
            lstm_hidden_dim=64,
            temporal_window=20,
            output_dim=1,
            dropout=0.3,
            device='cpu'
        )
        
        # Create trainer
        checkpoint_dir = Path(f"checkpoints/r2_optimization/{loss_name}")
        
        trainer = ImprovedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            learning_rate=5e-5,
            weight_decay=1e-4,
            patience=30,
            device='cpu'
        )
        
        # Train
        history = trainer.train(num_epochs=100, checkpoint_dir=checkpoint_dir)
        
        # Store results
        results[loss_name] = {
            'best_val_loss': history['best_val_loss'],
            'best_val_r2': max(history['val_r2_scores']),
            'final_dir_acc': history['val_dir_accs'][-1],
            'history': history
        }
        
        print(f"\n{loss_name} Results:")
        print(f"  Best Val Loss: {history['best_val_loss']:.6f}")
        print(f"  Best Val R²: {max(history['val_r2_scores']):.4f}")
        print(f"  Final Dir Acc: {history['val_dir_accs'][-1]*100:.1f}%")
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON OF ALL LOSS FUNCTIONS")
    print("="*70)
    print()
    
    comparison_df = pd.DataFrame([
        {
            'Loss Function': name,
            'Best Val R²': res['best_val_r2'],
            'Final Dir Acc (%)': res['final_dir_acc'] * 100,
            'Best Val Loss': res['best_val_loss']
        }
        for name, res in results.items()
    ])
    
    print(comparison_df.to_string(index=False))
    print()
    
    # Find best
    best_loss = max(results.items(), key=lambda x: x[1]['best_val_r2'])
    print(f"🏆 BEST PERFORMER: {best_loss[0]}")
    print(f"   R²: {best_loss[1]['best_val_r2']:.4f}")
    print(f"   Dir Acc: {best_loss[1]['final_dir_acc']*100:.1f}%")
    print()
    
    # Save comparison
    results_dir = Path("results/r2_optimization")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_df.to_csv(results_dir / 'loss_comparison.csv', index=False)
    
    with open(results_dir / 'all_results.json', 'w') as f:
        json.dump({
            name: {
                'best_val_r2': float(res['best_val_r2']),
                'final_dir_acc': float(res['final_dir_acc']),
                'best_val_loss': float(res['best_val_loss'])
            }
            for name, res in results.items()
        }, f, indent=2)
    
    print(f"✓ Results saved to: {results_dir}")
    print()
    print("Next: Evaluate best model on test set")
    print()


if __name__ == "__main__":
    main()
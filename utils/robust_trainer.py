"""
Robust Trainer with Huber Loss for Financial Data

Uses Huber loss which is less sensitive to outliers than MSE.
Better suited for noisy financial volatility prediction.

Author: EV Supply Chain GAT Team
Date: December 2024
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np


class RobustTrainer:
    """
    Trainer with Huber loss for robust financial predictions.
    
    Huber loss combines advantages of MSE and MAE:
    - Acts like MSE for small errors (smooth gradients)
    - Acts like MAE for large errors (robust to outliers)
    
    This is crucial for financial data with occasional extreme values.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 5e-5,  # Lower than before
        weight_decay: float = 1e-4,   # Higher regularization
        gradient_clip_val: float = 1.0,
        early_stopping_patience: int = 30,
        checkpoint_dir: str = "checkpoints",
        device: str = "cpu",
        huber_delta: float = 1.0  # Huber loss threshold
    ):
        """Initialize robust trainer"""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Huber loss (robust to outliers)
        self.criterion = nn.HuberLoss(delta=huber_delta)
        
        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training configuration
        self.gradient_clip_val = gradient_clip_val
        self.early_stopping_patience = early_stopping_patience
        
        # Checkpoint management
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.epoch_times = []
        
        # Setup logging
        self.logger = self._setup_logger()
        self._log_config(learning_rate, weight_decay, huber_delta)
    
    def _setup_logger(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger(f"{__name__}.RobustTrainer")
        logger.setLevel(logging.INFO)
        
        if logger.handlers:
            logger.handlers.clear()
        
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _log_config(self, lr: float, wd: float, delta: float):
        """Log training configuration"""
        self.logger.info("Robust Trainer Configuration:")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Loss: Huber (delta={delta})")
        self.logger.info(f"  Learning rate: {lr}")
        self.logger.info(f"  Weight decay: {wd}")
        self.logger.info(f"  Gradient clip: {self.gradient_clip_val}")
        self.logger.info(f"  Early stopping patience: {self.early_stopping_patience}")
        self.logger.info(f"  Train batches: {len(self.train_loader)}")
        self.logger.info(f"  Val batches: {len(self.val_loader)}")
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        for batch_idx, (features, edge_index, targets) in enumerate(self.train_loader):
            # Move to device
            features = features.to(self.device)
            edge_index = edge_index.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(features, edge_index)
            predictions = predictions.squeeze(-1)
            
            # Compute Huber loss
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_val
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Track loss
            epoch_losses.append(loss.item())
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                self.logger.info(
                    f"  Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.6f}"
                )
        
        return np.mean(epoch_losses)
    
    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, edge_index, targets in self.val_loader:
                features = features.to(self.device)
                edge_index = edge_index.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(features, edge_index)
                predictions = predictions.squeeze(-1)
                
                # Compute loss
                loss = self.criterion(predictions, targets)
                epoch_losses.append(loss.item())
                
                # Store for metrics
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Compute average loss
        avg_loss = np.mean(epoch_losses)
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self._compute_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss
        
        return avg_loss, metrics
    
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute evaluation metrics"""
        # MSE (for comparison)
        mse = torch.mean((predictions - targets) ** 2).item()
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # MAE
        mae = torch.mean(torch.abs(predictions - targets)).item()
        
        # R²
        ss_res = torch.sum((targets - predictions) ** 2).item()
        ss_tot = torch.sum((targets - targets.mean()) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Directional accuracy
        correct_direction = (
            torch.sign(predictions) == torch.sign(targets)
        ).float().mean().item()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': correct_direction
        }
    
    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
    
    def train(
        self,
        num_epochs: int,
        verbose: bool = True
    ) -> Dict[str, list]:
        """Train the model"""
        self.logger.info("="*70)
        self.logger.info("STARTING ROBUST TRAINING")
        self.logger.info("="*70)
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch + 1
            
            if verbose:
                print(f"\nEpoch {self.current_epoch}/{num_epochs}")
                print("-" * 50)
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch()
            self.val_losses.append(val_loss)
            
            # Track time
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            
            # Check if best
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(self.current_epoch, val_loss, is_best)
            
            # Print progress
            if verbose:
                print(f"Train Loss: {train_loss:.6f}")
                print(f"Val Loss:   {val_loss:.6f}")
                print(f"Val RMSE:   {val_metrics['rmse']:.6f}")
                print(f"Val R²:     {val_metrics['r2']:.4f}")
                print(f"Val Dir Acc: {val_metrics['directional_accuracy']*100:.2f}%")
                print(f"Time:       {epoch_time:.2f}s")
                if is_best:
                    print("✓ New best model!")
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                self.logger.info(
                    f"\nEarly stopping triggered after {self.current_epoch} epochs"
                )
                break
        
        self.logger.info("="*70)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info("="*70)
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epoch_times': self.epoch_times,
            'best_val_loss': self.best_val_loss
        }
"""
Training Loop for ST-GAT Model

This module implements the training infrastructure including:
- Training and validation loops
- Loss computation and optimization
- Early stopping
- Model checkpointing
- Metrics tracking and logging

Author: EV Supply Chain GAT Team
Date: December 2024
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
import numpy as np


class STGATTrainer:
    """
    Trainer class for ST-GAT model.
    
    Handles the complete training pipeline including:
    - Forward/backward passes
    - Loss computation (MSE for volatility prediction)
    - Optimization with Adam
    - Early stopping based on validation loss
    - Model checkpointing
    - Training metrics tracking
    
    The trainer follows best practices:
    - Separate train/validation loops
    - Gradient clipping to prevent exploding gradients
    - Learning rate scheduling (optional)
    - Comprehensive logging
    - Automatic device handling (CPU/GPU)
    
    Attributes:
        model (nn.Module): ST-GAT model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        optimizer (Optimizer): Optimizer (Adam)
        criterion (nn.Module): Loss function (MSE)
        device (str): Device to train on ('cpu' or 'cuda')
        checkpoint_dir (Path): Directory for saving checkpoints
        early_stopping_patience (int): Epochs to wait before early stopping
        gradient_clip_val (float): Max gradient norm for clipping
        logger (logging.Logger): Logger instance
    """
    
    # Default hyperparameters
    DEFAULT_LEARNING_RATE = 1e-3
    DEFAULT_WEIGHT_DECAY = 1e-5
    DEFAULT_GRADIENT_CLIP = 1.0
    DEFAULT_EARLY_STOPPING_PATIENCE = 15
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        gradient_clip_val: float = DEFAULT_GRADIENT_CLIP,
        early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
        checkpoint_dir: str = "checkpoints",
        device: str = "cpu"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: ST-GAT model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            learning_rate: Learning rate for optimizer (default: 1e-3)
            weight_decay: L2 regularization weight (default: 1e-5)
            gradient_clip_val: Max gradient norm for clipping (default: 1.0)
            early_stopping_patience: Epochs to wait before early stopping (default: 15)
            checkpoint_dir: Directory to save model checkpoints
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function - MSE for volatility prediction
        self.criterion = nn.MSELoss()
        
        # Optimizer - Adam with weight decay
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
        
        self.logger.info("Trainer initialized")
        self._log_config(learning_rate, weight_decay)
    
    def _setup_logger(self) -> logging.Logger:
        """
        Configure logging for trainer.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"{__name__}.STGATTrainer")
        logger.setLevel(logging.INFO)
        
        if logger.handlers:
            logger.handlers.clear()
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _log_config(self, lr: float, wd: float) -> None:
        """Log training configuration."""
        self.logger.info("Training Configuration:")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Learning rate: {lr}")
        self.logger.info(f"  Weight decay: {wd}")
        self.logger.info(f"  Gradient clip: {self.gradient_clip_val}")
        self.logger.info(f"  Early stopping patience: {self.early_stopping_patience}")
        self.logger.info(f"  Train batches: {len(self.train_loader)}")
        self.logger.info(f"  Val batches: {len(self.val_loader)}")
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        epoch_losses = []
        
        for batch_idx, (features, edge_index, targets) in enumerate(self.train_loader):
            # Move data to device
            features = features.to(self.device)
            edge_index = edge_index.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(features, edge_index)
            
            # Reshape for loss computation
            # predictions: [batch, num_nodes, 1] -> [batch, num_nodes]
            # targets: [batch, num_nodes]
            predictions = predictions.squeeze(-1)
            
            # Compute loss
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
            
            # Log progress every 20 batches
            if (batch_idx + 1) % 20 == 0:
                self.logger.info(
                    f"  Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.6f}"
                )
        
        avg_loss = np.mean(epoch_losses)
        return avg_loss
    
    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch.
        
        Returns:
            Tuple of (average validation loss, metrics dict)
        """
        self.model.eval()
        epoch_losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, edge_index, targets in self.val_loader:
                # Move data to device
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
        
        # Compute additional metrics
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
        """
        Compute evaluation metrics.
        
        Args:
            predictions: Model predictions [total_samples, num_nodes]
            targets: Ground truth targets [total_samples, num_nodes]
        
        Returns:
            Dictionary of metrics
        """
        # MSE
        mse = torch.mean((predictions - targets) ** 2).item()
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # MAE
        mae = torch.mean(torch.abs(predictions - targets)).item()
        
        # R² score
        ss_res = torch.sum((targets - predictions) ** 2).item()
        ss_tot = torch.sum((targets - targets.mean()) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            val_loss: Validation loss at this epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"  Saved best model (val_loss: {val_loss:.6f})")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(
            checkpoint_path, 
            map_location=self.device,
            weights_only=False  # Added: Allow loading optimizer state
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(
        self,
        num_epochs: int,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            verbose: Whether to print progress
        
        Returns:
            Dictionary with training history
        """
        self.logger.info("="*70)
        self.logger.info("STARTING TRAINING")
        self.logger.info("="*70)
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch + 1
            
            if verbose:
                print(f"\nEpoch {self.current_epoch}/{num_epochs}")
                print("-" * 50)
            
            # Training phase
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation phase
            val_loss, val_metrics = self.validate_epoch()
            self.val_losses.append(val_loss)
            
            # Track epoch time
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(self.current_epoch, val_loss, is_best)
            
            # Logging
            if verbose:
                print(f"Train Loss: {train_loss:.6f}")
                print(f"Val Loss:   {val_loss:.6f}")
                print(f"Val RMSE:   {val_metrics['rmse']:.6f}")
                print(f"Val MAE:    {val_metrics['mae']:.6f}")
                print(f"Val R²:     {val_metrics['r2']:.4f}")
                print(f"Time:       {epoch_time:.2f}s")
                
                if is_best:
                    print("✓ New best model!")
            
            self.logger.info(
                f"Epoch {self.current_epoch}: "
                f"Train={train_loss:.6f}, Val={val_loss:.6f}, "
                f"RMSE={val_metrics['rmse']:.6f}, "
                f"Time={epoch_time:.2f}s"
            )
            
            # Early stopping check
            if self.epochs_without_improvement >= self.early_stopping_patience:
                self.logger.info(
                    f"\nEarly stopping triggered after {self.current_epoch} epochs"
                )
                self.logger.info(
                    f"No improvement for {self.early_stopping_patience} epochs"
                )
                break
        
        # Training complete
        self.logger.info("="*70)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info("="*70)
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        self.logger.info(f"Total epochs: {self.current_epoch}")
        self.logger.info(f"Average epoch time: {np.mean(self.epoch_times):.2f}s")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epoch_times': self.epoch_times,
            'best_val_loss': self.best_val_loss
        }
    
    def get_training_summary(self) -> Dict:
        """
        Get summary of training statistics.
        
        Returns:
            Dictionary with training summary
        """
        return {
            'total_epochs': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else None,
            'total_training_time': np.sum(self.epoch_times) if self.epoch_times else None
        }
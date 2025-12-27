"""
Advanced Loss Functions for Volatility Prediction

Implements specialized loss functions to improve R² while
maintaining directional accuracy:
1. Stock-weighted loss
2. Multi-task loss (direction + magnitude)
3. Quantile-robust loss
4. Variance-preserving loss

Author: EV Supply Chain GAT Team
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StockWeightedHuberLoss(nn.Module):
    """
    Huber loss with per-stock weighting.
    
    Weights stocks by their baseline R² to focus on learnable patterns.
    Raw materials get higher weight than OEMs.
    """
    
    def __init__(
        self,
        stock_weights: torch.Tensor,
        delta: float = 1.0
    ):
        """
        Initialize stock-weighted Huber loss.
        
        Args:
            stock_weights: Weight per stock [num_stocks]
            delta: Huber loss threshold
        """
        super(StockWeightedHuberLoss, self).__init__()
        self.stock_weights = stock_weights
        self.delta = delta
        self.huber = nn.HuberLoss(delta=delta, reduction='none')
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted loss.
        
        Args:
            predictions: [batch, num_stocks]
            targets: [batch, num_stocks]
        
        Returns:
            Scalar loss
        """
        # Per-sample, per-stock loss
        losses = self.huber(predictions, targets)  # [batch, num_stocks]
        
        # Weight by stock
        weighted_losses = losses * self.stock_weights.unsqueeze(0)
        
        # Average
        return weighted_losses.mean()


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss: Direction classification + Magnitude regression.
    
    Explicitly optimizes for both:
    - Direction: Binary classification (positive/negative)
    - Magnitude: Huber regression on absolute values
    
    This leverages the model's strength in direction prediction.
    """
    
    def __init__(
        self,
        direction_weight: float = 0.3,
        magnitude_weight: float = 0.7,
        huber_delta: float = 1.0
    ):
        """
        Initialize multi-task loss.
        
        Args:
            direction_weight: Weight for direction loss
            magnitude_weight: Weight for magnitude loss
            huber_delta: Huber threshold for magnitude
        """
        super(MultiTaskLoss, self).__init__()
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        self.huber = nn.HuberLoss(delta=huber_delta)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-task loss.
        
        Args:
            predictions: [batch, num_stocks]
            targets: [batch, num_stocks]
        
        Returns:
            Combined loss
        """
        # Direction loss (binary cross-entropy on signs)
        pred_signs = torch.sign(predictions)
        target_signs = torch.sign(targets)
        
        # Convert to probabilities for BCE
        # Sign: -1 or 1 → probability: 0 or 1
        pred_probs = (pred_signs + 1) / 2  # [-1,1] → [0,1]
        target_probs = (target_signs + 1) / 2
        
        direction_loss = F.binary_cross_entropy(
            pred_probs.clamp(0.01, 0.99),  # Avoid log(0)
            target_probs,
            reduction='mean'
        )
        
        # Magnitude loss (Huber on absolute values)
        magnitude_loss = self.huber(
            torch.abs(predictions),
            torch.abs(targets)
        )
        
        # Combine
        total_loss = (
            self.direction_weight * direction_loss +
            self.magnitude_weight * magnitude_loss
        )
        
        return total_loss


class VariancePreservingLoss(nn.Module):
    """
    Loss that explicitly penalizes variance mismatch.
    
    Standard losses can lead to predictions with lower variance
    than targets (conservative predictions → poor R²).
    
    This loss adds a term to match prediction variance to target variance.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        variance_weight: float = 0.1
    ):
        """
        Initialize variance-preserving loss.
        
        Args:
            base_loss: Base loss function (e.g., Huber)
            variance_weight: Weight for variance matching term
        """
        super(VariancePreservingLoss, self).__init__()
        self.base_loss = base_loss
        self.variance_weight = variance_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with variance matching.
        
        Args:
            predictions: [batch, num_stocks]
            targets: [batch, num_stocks]
        
        Returns:
            Combined loss
        """
        # Base loss (e.g., Huber)
        base = self.base_loss(predictions, targets)
        
        # Variance matching per stock
        pred_var = predictions.var(dim=0)  # [num_stocks]
        target_var = targets.var(dim=0)    # [num_stocks]
        
        variance_loss = F.mse_loss(pred_var, target_var)
        
        # Combine
        total_loss = base + self.variance_weight * variance_loss
        
        return total_loss


class AdaptiveHuberLoss(nn.Module):
    """
    Huber loss with adaptive delta per stock.
    
    Delta adapts based on target distribution statistics.
    Stocks with higher variance get higher delta (more robust).
    """
    
    def __init__(self, target_stds: torch.Tensor):
        """
        Initialize adaptive Huber loss.
        
        Args:
            target_stds: Standard deviation per stock [num_stocks]
        """
        super(AdaptiveHuberLoss, self).__init__()
        
        # Delta = 0.5 * std (adaptive threshold)
        self.deltas = 0.5 * target_stds
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive Huber loss.
        
        Args:
            predictions: [batch, num_stocks]
            targets: [batch, num_stocks]
        
        Returns:
            Loss
        """
        errors = predictions - targets
        
        # Huber loss per stock with adaptive delta
        losses = []
        for stock_idx in range(predictions.shape[1]):
            delta = self.deltas[stock_idx]
            stock_errors = errors[:, stock_idx]
            
            # Huber formula
            is_small_error = torch.abs(stock_errors) <= delta
            squared_loss = 0.5 * stock_errors ** 2
            linear_loss = delta * (torch.abs(stock_errors) - 0.5 * delta)
            
            stock_loss = torch.where(is_small_error, squared_loss, linear_loss)
            losses.append(stock_loss.mean())
        
        return torch.stack(losses).mean()


class QuantileRegressionLoss(nn.Module):
    """
    Quantile regression loss for robust prediction.
    
    Instead of predicting mean (MSE/Huber), predict median (quantile=0.5).
    More robust to outliers and may improve R².
    """
    
    def __init__(self, quantile: float = 0.5):
        """
        Initialize quantile loss.
        
        Args:
            quantile: Quantile to predict (0.5 = median)
        """
        super(QuantileRegressionLoss, self).__init__()
        self.quantile = quantile
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile loss.
        
        Args:
            predictions: [batch, num_stocks]
            targets: [batch, num_stocks]
        
        Returns:
            Loss
        """
        errors = targets - predictions
        
        loss = torch.where(
            errors >= 0,
            self.quantile * errors,
            (self.quantile - 1) * errors
        )
        
        return loss.mean()


def compute_stock_weights_from_baseline(
    train_dataset,
    stock_names: list
) -> torch.Tensor:
    """
    Compute stock weights based on baseline R² scores.
    
    Stocks easier to predict get higher weight.
    
    Args:
        train_dataset: Training dataset
        stock_names: List of stock tickers
    
    Returns:
        Weights tensor [num_stocks]
    """
    # Collect all targets
    all_targets = []
    for _, _, target in train_dataset:
        all_targets.append(target)
    
    all_targets = torch.stack(all_targets)  # [num_samples, num_stocks]
    
    # Compute baseline R² (predict mean)
    baseline_r2 = []
    for stock_idx in range(len(stock_names)):
        targets = all_targets[:, stock_idx]
        mean_pred = targets.mean()
        
        ss_res = ((targets - mean_pred) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        
        r2 = 1 - (ss_res / ss_tot)
        baseline_r2.append(max(r2.item(), 0.0))  # Clip to 0
    
    baseline_r2 = torch.tensor(baseline_r2)
    
    # Weight = softmax(baseline_r2) to emphasize learnable stocks
    weights = F.softmax(baseline_r2 * 5, dim=0)  # Temperature = 5
    
    print("\nStock weights (based on learnability):")
    for stock, weight, r2 in zip(stock_names, weights, baseline_r2):
        print(f"  {stock}: weight={weight:.3f}, baseline_r2={r2:.3f}")
    print()
    
    return weights
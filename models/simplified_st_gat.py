"""
Simplified ST-GAT for Financial Volatility Prediction

Key improvements:
- Simpler architecture (less overfitting on noisy data)
- Residual connections
- Batch normalization
- Designed specifically for financial time series

Author: EV Supply Chain GAT Team
Date: December 2024
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multi_head_gat import MultiHeadGATLayer


class SimplifiedSTGAT(nn.Module):
    """
    Simplified Spatio-Temporal Graph Attention Network for Finance.
    
    Designed specifically for noisy financial data:
    - Single GAT layer (avoid overparameterization)
    - Single LSTM layer (simpler temporal modeling)
    - Batch normalization (stabilize training)
    - Residual connections (learn deviations)
    - Smaller hidden dimensions (prevent overfitting)
    
    Architecture:
        Input → [GAT] → BatchNorm → [LSTM] → BatchNorm → [Linear] → Output
                 ↓                                                      ↑
                 └──────────────── Residual ──────────────────────────┘
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        input_features: int,
        gat_hidden_dim: int = 64,
        gat_heads: int = 4,
        lstm_hidden_dim: int = 64,
        temporal_window: int = 20,
        output_dim: int = 1,
        dropout: float = 0.2,
        device: str = "cpu"
    ):
        """
        Initialize Simplified ST-GAT.
        
        Args:
            num_nodes: Number of nodes (stocks) in graph
            num_edges: Number of edges (supply chain relationships)
            input_features: Number of input features per node
            gat_hidden_dim: Hidden dimension for GAT (default: 64, was 128)
            gat_heads: Number of attention heads (default: 4, was 8)
            lstm_hidden_dim: Hidden dimension for LSTM (default: 64, was 128)
            temporal_window: Length of temporal sequence
            output_dim: Output dimension (default: 1 for volatility)
            dropout: Dropout rate (default: 0.2)
            device: Device to run on
        """
        super(SimplifiedSTGAT, self).__init__()
        
        # Store configuration
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.input_features = input_features
        self.gat_hidden_dim = gat_hidden_dim
        self.gat_heads = gat_heads
        self.lstm_hidden_dim = lstm_hidden_dim
        self.temporal_window = temporal_window
        self.output_dim = output_dim
        self.dropout = dropout
        self.device = device
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Single GAT layer (multi-head)
        self.gat = MultiHeadGATLayer(
            in_features=input_features,
            out_features_per_head=gat_hidden_dim // gat_heads,
            num_heads=gat_heads,
            dropout=dropout,
            concat=True
        )
        
        # Batch normalization after GAT
        self.bn_gat = nn.BatchNorm1d(gat_hidden_dim)
        
        # Single LSTM layer
        self.lstm = nn.LSTM(
            input_size=gat_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=False,
            dropout=0.0,  # No dropout in single layer LSTM
            bidirectional=False
        )
        
        # Batch normalization after LSTM
        self.bn_lstm = nn.BatchNorm1d(lstm_hidden_dim)
        
        # Residual projection (if dimensions don't match)
        self.residual_proj = None
        if input_features != lstm_hidden_dim:
            self.residual_proj = nn.Linear(input_features, lstm_hidden_dim)
        
        # Output layer
        self.output_layer = nn.Linear(lstm_hidden_dim, output_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Move to device
        self.to(device)
        
        # Log model info
        self._log_model_info()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(f"{__name__}.SimplifiedSTGAT")
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
    
    def _log_model_info(self):
        """Log model configuration"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        self.logger.info("="*70)
        self.logger.info("SIMPLIFIED ST-GAT MODEL CONFIGURATION")
        self.logger.info("="*70)
        self.logger.info(f"Nodes: {self.num_nodes}, Edges: {self.num_edges}")
        self.logger.info(f"GAT: {self.input_features} → {self.gat_hidden_dim} ({self.gat_heads} heads)")
        self.logger.info(f"LSTM: {self.gat_hidden_dim} → {self.lstm_hidden_dim} (1 layer)")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info("="*70)
    
    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: [batch_size, num_nodes, seq_len, input_features]
            edge_index: [2, num_edges]
        
        Returns:
            predictions: [batch_size, num_nodes, output_dim]
        """
        batch_size, num_nodes, seq_len, feat_dim = features.shape
        
        # Process each timestep through GAT
        gat_outputs = []
        
        for t in range(seq_len):
            # Get features at time t: [batch, num_nodes, feat_dim]
            x_t = features[:, :, t, :]
            
            # Reshape for GAT: [batch * num_nodes, feat_dim]
            x_t_flat = x_t.reshape(batch_size * num_nodes, feat_dim)
            
            # Apply GAT
            h_t = self.gat(x_t_flat, edge_index)
            h_t = self.dropout_layer(h_t)
            
            # Batch norm
            h_t = self.bn_gat(h_t)
            
            # Reshape back: [batch, num_nodes, gat_hidden_dim]
            h_t = h_t.reshape(batch_size, num_nodes, self.gat_hidden_dim)
            
            gat_outputs.append(h_t)
        
        # Stack temporal outputs: [batch, num_nodes, seq_len, gat_hidden_dim]
        gat_output = torch.stack(gat_outputs, dim=2)
        
        # Reshape for LSTM: [num_nodes, batch, seq_len, gat_hidden_dim]
        gat_output = gat_output.permute(1, 0, 2, 3)
        
        # Process each node through LSTM
        lstm_outputs = []
        
        for node_idx in range(num_nodes):
            # Get sequence for this node: [batch, seq_len, gat_hidden_dim]
            node_seq = gat_output[node_idx]
            
            # Transpose for LSTM: [seq_len, batch, gat_hidden_dim]
            node_seq = node_seq.transpose(0, 1)
            
            # LSTM
            lstm_out, _ = self.lstm(node_seq)
            
            # Take last timestep: [batch, lstm_hidden_dim]
            lstm_last = lstm_out[-1]
            
            # Batch norm
            lstm_last = self.bn_lstm(lstm_last)
            
            lstm_outputs.append(lstm_last)
        
        # Stack: [num_nodes, batch, lstm_hidden_dim]
        lstm_output = torch.stack(lstm_outputs, dim=0)
        
        # Transpose: [batch, num_nodes, lstm_hidden_dim]
        lstm_output = lstm_output.transpose(0, 1)
        
        # Residual connection
        # Use last timestep of original features as residual
        residual = features[:, :, -1, :]  # [batch, num_nodes, input_features]
        
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        # Add residual
        output = lstm_output + residual
        
        # Dropout
        output = self.dropout_layer(output)
        
        # Final prediction
        predictions = self.output_layer(output)
        
        return predictions


def create_simplified_model(
    num_nodes: int = 7,
    num_edges: int = 15,
    input_features: int = 15,
    device: str = "cpu"
) -> SimplifiedSTGAT:
    """
    Create simplified ST-GAT model with good defaults for finance.
    
    Args:
        num_nodes: Number of stocks
        num_edges: Number of supply chain edges
        input_features: Number of features per node
        device: Device to run on
    
    Returns:
        Initialized SimplifiedSTGAT model
    """
    model = SimplifiedSTGAT(
        num_nodes=num_nodes,
        num_edges=num_edges,
        input_features=input_features,
        gat_hidden_dim=64,  # Reduced from 128
        gat_heads=4,        # Reduced from 8
        lstm_hidden_dim=64, # Reduced from 128
        temporal_window=20,
        output_dim=1,
        dropout=0.3,        # Increased from 0.2
        device=device
    )
    
    return model
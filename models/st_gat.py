"""
Spatio-Temporal Graph Attention Network for EV Supply Chain Volatility Prediction

This module implements the main ST-GAT model that combines:
- Graph Attention Networks (GAT) for spatial supply chain relationships
- Long Short-Term Memory (LSTM) for temporal patterns
- Attention mechanisms for interpretable risk transmission pathways

The model predicts next-day volatility spillovers across the supply chain.

Author: EV Supply Chain GAT Team
Date: December 2025
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multi_head_gat import MultiHeadGATLayer
from models.lstm_layer import TemporalLSTM


class STGAT(nn.Module):
    """
    Spatio-Temporal Graph Attention Network for supply chain volatility prediction.
    
    Architecture:
        1. Multi-head GAT layers capture spatial supply chain relationships
        2. LSTM layers process temporal sequences
        3. Output layer predicts next-day volatility per node
    
    The model learns to identify critical supply chain pathways during
    volatility events using attention mechanisms.
    
    Forward pass flow:
        Input: [batch, num_nodes, seq_len, input_features]
        → GAT layers at each timestep: [batch, num_nodes, seq_len, gat_hidden]
        → LSTM temporal processing: [batch, num_nodes, lstm_hidden]
        → Output layer: [batch, num_nodes, 1]
    
    Attributes:
        num_nodes (int): Number of companies in supply chain graph
        num_edges (int): Number of supply relationships
        input_features (int): Number of input features per node
        gat_hidden_dim (int): Hidden dimension for GAT layers
        gat_heads (int): Number of attention heads
        gat_layers (int): Number of GAT layers
        lstm_hidden_dim (int): Hidden dimension for LSTM
        lstm_layers (int): Number of LSTM layers
        temporal_window (int): Number of past days to consider
        dropout (float): Dropout rate for regularization
        logger (logging.Logger): Logger instance
    """
    
    # Model hyperparameters (following real-world robust configuration)
    NUM_NODES = 8              # Companies in supply chain
    NUM_EDGES = 21             # Supply chain relationships
    INPUT_FEATURES = 13        # Features per node (5 stock + 8 macro)
    
    # GAT configuration
    GAT_HIDDEN_DIM = 128       # Hidden dimension for GAT
    GAT_HEADS = 8              # Number of attention heads
    GAT_LAYERS = 2             # Number of GAT layers (captures 2-hop dependencies)
    GAT_DROPOUT = 0.1          # Dropout on attention coefficients
    
    # LSTM configuration
    LSTM_HIDDEN_DIM = 128      # Hidden dimension for LSTM (matches GAT output)
    LSTM_LAYERS = 2            # Number of LSTM layers
    LSTM_DROPOUT = 0.2         # Dropout between LSTM layers
    
    # Training configuration
    TEMPORAL_WINDOW = 20       # Days of history to consider (1 month trading days)
    OUTPUT_DIM = 1             # Predict single volatility value per node
    
    def __init__(
        self,
        num_nodes: int = NUM_NODES,
        num_edges: int = NUM_EDGES,
        input_features: int = INPUT_FEATURES,
        gat_hidden_dim: int = GAT_HIDDEN_DIM,
        gat_heads: int = GAT_HEADS,
        gat_layers: int = GAT_LAYERS,
        gat_dropout: float = GAT_DROPOUT,
        lstm_hidden_dim: int = LSTM_HIDDEN_DIM,
        lstm_layers: int = LSTM_LAYERS,
        lstm_dropout: float = LSTM_DROPOUT,
        temporal_window: int = TEMPORAL_WINDOW,
        output_dim: int = OUTPUT_DIM,
        device: str = "cpu"
    ):
        """
        Initialize the ST-GAT model.
        
        Args:
            num_nodes: Number of nodes in supply chain graph
            num_edges: Number of directed edges (relationships)
            input_features: Number of input features per node
            gat_hidden_dim: Hidden dimension for GAT layers
            gat_heads: Number of attention heads per GAT layer
            gat_layers: Number of GAT layers
            gat_dropout: Dropout rate for GAT attention
            lstm_hidden_dim: Hidden dimension for LSTM
            lstm_layers: Number of LSTM layers
            lstm_dropout: Dropout rate between LSTM layers
            temporal_window: Number of past time steps to process
            output_dim: Output dimension (1 for volatility prediction)
            device: Device to run model on ('cpu' or 'cuda')
        """
        super(STGAT, self).__init__()
        
        # Store configuration
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.input_features = input_features
        self.gat_hidden_dim = gat_hidden_dim
        self.gat_heads = gat_heads
        self.gat_layers = gat_layers
        self.gat_dropout = gat_dropout
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.temporal_window = temporal_window
        self.output_dim = output_dim
        self.device = device
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Build model layers
        self._build_gat_layers()
        self._build_lstm_layer()
        self._build_output_layer()
        
        # Track model statistics
        self.model_stats = {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "gat_parameters": 0,
            "lstm_parameters": 0,
            "output_parameters": 0
        }
        
        # Count parameters
        self._count_parameters()
        
        self.logger.info("ST-GAT model fully initialized")
        self._print_config()
    
    def _build_gat_layers(self):
        """Build the GAT layers."""
        self.gat_layers_list = nn.ModuleList()
        
        # Each head outputs gat_hidden_dim / gat_heads features
        out_per_head = self.gat_hidden_dim // self.gat_heads
        
        for layer_idx in range(self.gat_layers):
            if layer_idx == 0:
                # First layer: input_features -> gat_hidden_dim
                gat_layer = MultiHeadGATLayer(
                    num_heads=self.gat_heads,
                    in_features=self.input_features,
                    out_features_per_head=out_per_head,
                    dropout=self.gat_dropout,
                    concat=True  # Concatenate heads
                )
            else:
                # Subsequent layers: gat_hidden_dim -> gat_hidden_dim
                gat_layer = MultiHeadGATLayer(
                    num_heads=self.gat_heads,
                    in_features=self.gat_hidden_dim,
                    out_features_per_head=out_per_head,
                    dropout=self.gat_dropout,
                    concat=True
                )
            
            self.gat_layers_list.append(gat_layer)
        
        self.logger.info(f"Built {self.gat_layers} GAT layers")
    
    def _build_lstm_layer(self):
        """Build the LSTM layer."""
        self.lstm = TemporalLSTM(
            input_dim=self.gat_hidden_dim,
            hidden_dim=self.lstm_hidden_dim,
            num_layers=self.lstm_layers,
            dropout=self.lstm_dropout,
            bidirectional=False
        )
        
        self.logger.info("Built LSTM layer")
    
    def _build_output_layer(self):
        """Build the output prediction layer."""
        self.output_layer = nn.Linear(self.lstm_hidden_dim, self.output_dim)
        
        self.logger.info("Built output layer")
    
    def _setup_logger(self) -> logging.Logger:
        """
        Configure logging for this model.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"{__name__}.STGAT")
        logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if logger.handlers:
            logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _count_parameters(self) -> Dict[str, int]:
        """
        Count total and trainable parameters in the model.
        
        Returns:
            Dictionary with parameter counts
        """
        # GAT parameters
        gat_params = sum(p.numel() for layer in self.gat_layers_list for p in layer.parameters())
        
        # LSTM parameters
        lstm_params = sum(p.numel() for p in self.lstm.parameters())
        
        # Output layer parameters
        output_params = sum(p.numel() for p in self.output_layer.parameters())
        
        # Total
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        self.model_stats["gat_parameters"] = gat_params
        self.model_stats["lstm_parameters"] = lstm_params
        self.model_stats["output_parameters"] = output_params
        self.model_stats["total_parameters"] = total
        self.model_stats["trainable_parameters"] = trainable
        
        return {
            "total": total,
            "trainable": trainable,
            "gat": gat_params,
            "lstm": lstm_params,
            "output": output_params
        }
    
    def _print_config(self) -> None:
        """Print model configuration summary."""
        print("\n" + "="*70)
        print("ST-GAT MODEL CONFIGURATION")
        print("="*70)
        
        print("\n📊 Graph Structure:")
        print(f"  Nodes: {self.num_nodes}")
        print(f"  Edges: {self.num_edges}")
        print(f"  Input features per node: {self.input_features}")
        
        print("\n🔷 GAT Configuration:")
        print(f"  Hidden dimension: {self.gat_hidden_dim}")
        print(f"  Attention heads: {self.gat_heads}")
        print(f"  Number of layers: {self.gat_layers}")
        print(f"  Dropout rate: {self.gat_dropout}")
        
        print("\n⏱️  LSTM Configuration:")
        print(f"  Hidden dimension: {self.lstm_hidden_dim}")
        print(f"  Number of layers: {self.lstm_layers}")
        print(f"  Dropout rate: {self.lstm_dropout}")
        
        print("\n📈 Temporal Configuration:")
        print(f"  Temporal window: {self.temporal_window} days")
        print(f"  Output dimension: {self.output_dim}")
        
        print("\n💻 Device:")
        print(f"  Running on: {self.device}")
        
        print("\n🔢 Model Parameters:")
        print(f"  GAT parameters: {self.model_stats['gat_parameters']:,}")
        print(f"  LSTM parameters: {self.model_stats['lstm_parameters']:,}")
        print(f"  Output parameters: {self.model_stats['output_parameters']:,}")
        print(f"  Total parameters: {self.model_stats['total_parameters']:,}")
        print(f"  Trainable parameters: {self.model_stats['trainable_parameters']:,}")
        
        print("="*70 + "\n")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ST-GAT model.
        
        Args:
            x: Node features [batch_size, num_nodes, seq_len, features]
            edge_index: Graph edge indices [2, num_edges]
        
        Returns:
            Predicted volatility [batch_size, num_nodes, output_dim]
        """
        batch_size, num_nodes, seq_len, features = x.shape
        
        # Step 1: Apply GAT layers at each timestep
        # We need to process each timestep independently through GAT
        # Then stack the results back into temporal dimension
        
        gat_outputs = []
        
        for t in range(seq_len):
            # Extract features for timestep t: [batch, num_nodes, features]
            x_t = x[:, :, t, :]
            
            # Reshape for GAT: [batch * num_nodes, features]
            x_t_flat = x_t.reshape(batch_size * num_nodes, features)
            
            # Apply GAT layers
            h = x_t_flat
            for gat_layer in self.gat_layers_list:
                h = gat_layer(h, edge_index)
            
            # Reshape back: [batch, num_nodes, gat_hidden_dim]
            h = h.reshape(batch_size, num_nodes, self.gat_hidden_dim)
            
            gat_outputs.append(h)
        
        # Stack temporal outputs: [batch, num_nodes, seq_len, gat_hidden_dim]
        gat_output = torch.stack(gat_outputs, dim=2)
        
        # Step 2: Apply LSTM to temporal sequences
        # Input: [batch, num_nodes, seq_len, gat_hidden_dim]
        # Output: [batch, num_nodes, lstm_hidden_dim]
        lstm_output, _ = self.lstm(gat_output)
        
        # Step 3: Apply output layer to predict volatility
        # Input: [batch, num_nodes, lstm_hidden_dim]
        # Output: [batch, num_nodes, output_dim]
        predictions = self.output_layer(lstm_output)
        
        return predictions
    
    def get_config(self) -> Dict:
        """
        Get model configuration as dictionary.
        
        Returns:
            Dictionary containing all model hyperparameters
        """
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "input_features": self.input_features,
            "gat_hidden_dim": self.gat_hidden_dim,
            "gat_heads": self.gat_heads,
            "gat_layers": self.gat_layers,
            "gat_dropout": self.gat_dropout,
            "lstm_hidden_dim": self.lstm_hidden_dim,
            "lstm_layers": self.lstm_layers,
            "lstm_dropout": self.lstm_dropout,
            "temporal_window": self.temporal_window,
            "output_dim": self.output_dim,
            "device": self.device
        }
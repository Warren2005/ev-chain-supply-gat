"""
LSTM Temporal Layer for EV Supply Chain GAT Project

This module implements a multi-layer LSTM for processing temporal sequences
of node features after GAT spatial processing.

The LSTM captures temporal patterns in volatility spillovers across the
20-day temporal window.

Key features:
- Multi-layer LSTM (stacked layers)
- Dropout between layers for regularization
- Processes per-node temporal sequences independently
- Returns final hidden state for prediction

Author: EV Supply Chain GAT Team
Date: December 2025
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn


class TemporalLSTM(nn.Module):
    """
    Multi-layer LSTM for temporal sequence processing.
    
    Processes temporal sequences for each node independently to capture
    time-series patterns in features (volatility, returns, etc.).
    
    Architecture:
        - Input: [batch, num_nodes, seq_len, features]
        - Per-node processing: [batch * num_nodes, seq_len, features]
        - LSTM: processes sequences
        - Output: [batch, num_nodes, hidden_dim]
    
    Attributes:
        input_dim (int): Input feature dimension (from GAT output)
        hidden_dim (int): LSTM hidden state dimension
        num_layers (int): Number of stacked LSTM layers
        dropout (float): Dropout rate between LSTM layers
        bidirectional (bool): Whether to use bidirectional LSTM
        lstm (nn.LSTM): PyTorch LSTM module
        logger (logging.Logger): Logger instance
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Initialize the Temporal LSTM layer.
        
        Args:
            input_dim: Input feature dimension (typically GAT output dim)
            hidden_dim: Hidden state dimension
            num_layers: Number of stacked LSTM layers (default: 2)
            dropout: Dropout rate between layers (default: 0.2)
            bidirectional: If True, use bidirectional LSTM (default: False)
        """
        super(TemporalLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # LSTM layer
        # Note: dropout is only applied between layers if num_layers > 1
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True  # Input/output format: [batch, seq, features]
        )
        
        # Output dimension (doubled if bidirectional)
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Setup logging
        self.logger = self._setup_logger()
        self.logger.info(
            f"TemporalLSTM initialized: input_dim={input_dim}, "
            f"hidden_dim={hidden_dim}, num_layers={num_layers}, "
            f"dropout={dropout}, bidirectional={bidirectional}"
        )
    
    def _setup_logger(self) -> logging.Logger:
        """
        Configure logging for this layer.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"{__name__}.TemporalLSTM")
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
    
    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        c0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the LSTM layer.
        
        Processes temporal sequences for each node independently.
        
        Args:
            x: Input features [batch, num_nodes, seq_len, features]
            h0: Initial hidden state [num_layers * num_directions, batch * num_nodes, hidden_dim]
                If None, initialized to zeros
            c0: Initial cell state [num_layers * num_directions, batch * num_nodes, hidden_dim]
                If None, initialized to zeros
        
        Returns:
            Tuple of:
                - output: Final hidden state [batch, num_nodes, hidden_dim]
                - (h_n, c_n): Final hidden and cell states for all layers
        """
        batch_size, num_nodes, seq_len, features = x.shape
        
        # Reshape to process each node's sequence independently
        # [batch, num_nodes, seq_len, features] -> [batch * num_nodes, seq_len, features]
        x_reshaped = x.view(batch_size * num_nodes, seq_len, features)
        
        # Initialize hidden and cell states if not provided
        num_directions = 2 if self.bidirectional else 1
        if h0 is None:
            h0 = torch.zeros(
                self.num_layers * num_directions,
                batch_size * num_nodes,
                self.hidden_dim,
                device=x.device,
                dtype=x.dtype
            )
        if c0 is None:
            c0 = torch.zeros(
                self.num_layers * num_directions,
                batch_size * num_nodes,
                self.hidden_dim,
                device=x.device,
                dtype=x.dtype
            )
        
        # LSTM forward pass
        # lstm_out: [batch * num_nodes, seq_len, hidden_dim * num_directions]
        # h_n: [num_layers * num_directions, batch * num_nodes, hidden_dim]
        # c_n: [num_layers * num_directions, batch * num_nodes, hidden_dim]
        lstm_out, (h_n, c_n) = self.lstm(x_reshaped, (h0, c0))
        
        # We only need the final time step's output
        # [batch * num_nodes, hidden_dim * num_directions]
        final_hidden = lstm_out[:, -1, :]
        
        # Reshape back to [batch, num_nodes, hidden_dim * num_directions]
        output = final_hidden.view(batch_size, num_nodes, self.output_dim)
        
        return output, (h_n, c_n)
    
    def get_output_dim(self) -> int:
        """
        Get the output dimension of this layer.
        
        Returns:
            Output feature dimension
        """
        return self.output_dim
    
    def __repr__(self):
        """String representation of the layer."""
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"dropout={self.dropout}, "
            f"bidirectional={self.bidirectional}, "
            f"output_dim={self.output_dim})"
        )
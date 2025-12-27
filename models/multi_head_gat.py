"""
Multi-Head Graph Attention Layer for EV Supply Chain GAT Project

This module implements a multi-head GAT layer that runs multiple single-head
attention mechanisms in parallel and combines their outputs.

Multi-head attention allows the model to jointly attend to information from
different representation subspaces (e.g., different types of supply relationships).

Author: EV Supply Chain GAT Team
Date: December 2025
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gat_layer import GATLayer


class MultiHeadGATLayer(nn.Module):
    """
    Multi-head Graph Attention Network layer.
    
    Runs multiple single-head GAT layers in parallel and combines outputs
    either by concatenation or averaging.
    
    For intermediate layers: concatenate heads
        Output dim = num_heads * out_features_per_head
    
    For final layer: average heads
        Output dim = out_features_per_head
    
    Attributes:
        num_heads (int): Number of attention heads
        in_features (int): Number of input features per node
        out_features_per_head (int): Output features per attention head
        dropout (float): Dropout rate for attention coefficients
        alpha (float): Negative slope for LeakyReLU
        concat (bool): If True, concatenate heads; if False, average heads
        attention_heads (nn.ModuleList): List of single-head GAT layers
        logger (logging.Logger): Logger instance
    """
    
    def __init__(
        self,
        num_heads: int,
        in_features: int,
        out_features_per_head: int,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True
    ):
        """
        Initialize the multi-head GAT layer.
        
        Args:
            num_heads: Number of attention heads
            in_features: Number of input features per node
            out_features_per_head: Output features per attention head
            dropout: Dropout rate for attention coefficients (default: 0.1)
            alpha: Negative slope for LeakyReLU activation (default: 0.2)
            concat: If True, concatenate heads; if False, average heads (default: True)
        """
        super(MultiHeadGATLayer, self).__init__()
        
        self.num_heads = num_heads
        self.in_features = in_features
        self.out_features_per_head = out_features_per_head
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Create multiple attention heads
        self.attention_heads = nn.ModuleList([
            GATLayer(
                in_features=in_features,
                out_features=out_features_per_head,
                dropout=dropout,
                alpha=alpha,
                concat=concat
            )
            for _ in range(num_heads)
        ])
        
        # Setup logging
        self.logger = self._setup_logger()
        self.logger.info(
            f"MultiHeadGATLayer initialized: {num_heads} heads, "
            f"{in_features} -> {out_features_per_head} per head, "
            f"concat={concat}"
        )
    
    def _setup_logger(self) -> logging.Logger:
        """
        Configure logging for this layer.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"{__name__}.MultiHeadGATLayer")
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
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the multi-head GAT layer.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Graph connectivity [2, num_edges]
        
        Returns:
            Updated node features:
                If concat=True: [num_nodes, num_heads * out_features_per_head]
                If concat=False: [num_nodes, out_features_per_head]
        """
        # Run each attention head
        head_outputs = [
            attention_head(x, edge_index)
            for attention_head in self.attention_heads
        ]
        
        if self.concat:
            # Concatenate all heads
            # [num_nodes, out_features_per_head] * num_heads -> [num_nodes, num_heads * out_features_per_head]
            output = torch.cat(head_outputs, dim=1)
        else:
            # Average all heads
            # [num_nodes, out_features_per_head] * num_heads -> [num_nodes, out_features_per_head]
            output = torch.stack(head_outputs, dim=0).mean(dim=0)
        
        return output
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Get attention weights from all heads (for interpretability).
        
        This will be useful in Phase 6 for visualizing which supply chain
        links are most important during volatility events.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Graph connectivity [2, num_edges]
        
        Returns:
            Attention weights [num_heads, num_edges]
        
        Note:
            This requires modifying GATLayer to return attention weights.
            For now, this is a placeholder for future implementation.
        """
        # TODO: Implement attention weight extraction in Phase 6
        raise NotImplementedError(
            "Attention weight extraction will be implemented in Phase 6 "
            "for interpretability analysis."
        )
    
    @property
    def output_dim(self) -> int:
        """
        Get the output dimension of this layer.
        
        Returns:
            Output feature dimension
        """
        if self.concat:
            return self.num_heads * self.out_features_per_head
        else:
            return self.out_features_per_head
    
    def __repr__(self):
        """String representation of the layer."""
        return (
            f"{self.__class__.__name__}("
            f"num_heads={self.num_heads}, "
            f"in_features={self.in_features}, "
            f"out_features_per_head={self.out_features_per_head}, "
            f"output_dim={self.output_dim}, "
            f"dropout={self.dropout}, "
            f"concat={self.concat})"
        )
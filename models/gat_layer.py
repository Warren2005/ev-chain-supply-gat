"""
Graph Attention Layer for EV Supply Chain GAT Project

This module implements a single-head Graph Attention Network (GAT) layer
following the architecture from Veličković et al. (2018).

The layer computes attention coefficients between connected nodes and
aggregates neighbor features using learned attention weights.

Key components:
- Linear transformation of node features
- Attention coefficient calculation
- LeakyReLU activation
- Softmax normalization
- Weighted message aggregation

Author: EV Supply Chain GAT Team
Date: December 2025
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Single-head Graph Attention Network layer.
    
    Implements the attention mechanism from the GAT paper:
    1. Linear transformation: h_i' = W * h_i
    2. Attention scores: e_ij = LeakyReLU(a^T [h_i' || h_j'])
    3. Attention weights: α_ij = softmax_j(e_ij)
    4. Aggregation: h_i_new = σ(Σ_j α_ij * h_j')
    
    Attributes:
        in_features (int): Number of input features per node
        out_features (int): Number of output features per node
        dropout (float): Dropout rate for attention coefficients
        alpha (float): Negative slope for LeakyReLU
        concat (bool): Whether to apply ELU activation (True for intermediate layers)
        logger (logging.Logger): Logger instance
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True
    ):
        """
        Initialize the GAT layer.
        
        Args:
            in_features: Number of input features per node
            out_features: Number of output features per node
            dropout: Dropout rate for attention coefficients (default: 0.1)
            alpha: Negative slope for LeakyReLU activation (default: 0.2)
            concat: If True, apply ELU activation; if False, no activation (default: True)
        """
        super(GATLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Linear transformation: W * h_i
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # Attention mechanism parameters: a^T [h_i' || h_j']
        # Since we concatenate h_i' and h_j', the attention vector is 2 * out_features
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        
        # Activation and dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize parameters
        self._reset_parameters()
        
        # Setup logging
        self.logger = self._setup_logger()
        self.logger.info(
            f"GATLayer initialized: {in_features} -> {out_features}, "
            f"dropout={dropout}, alpha={alpha}"
        )
    
    def _reset_parameters(self):
        """Initialize layer parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
    
    def _setup_logger(self) -> logging.Logger:
        """
        Configure logging for this layer.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"{__name__}.GATLayer")
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
        Forward pass through the GAT layer.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Graph connectivity [2, num_edges] where edge_index[0] are source nodes
                       and edge_index[1] are target nodes
        
        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Step 1: Linear transformation
        # h' = W * h
        h = self.W(x)  # [num_nodes, out_features]
        num_nodes = h.size(0)
        
        # Step 2: Compute attention coefficients
        # e_ij = LeakyReLU(a^T [h_i' || h_j'])
        
        # Get source and target node indices
        edge_src = edge_index[0]  # Source nodes (suppliers)
        edge_dst = edge_index[1]  # Target nodes (customers)
        
        # Get transformed features for source and target nodes
        h_src = h[edge_src]  # [num_edges, out_features]
        h_dst = h[edge_dst]  # [num_edges, out_features]
        
        # Concatenate source and target features
        h_concat = torch.cat([h_src, h_dst], dim=1)  # [num_edges, 2 * out_features]
        
        # Compute attention logits
        e = self.leakyrelu(torch.matmul(h_concat, self.a).squeeze(1))  # [num_edges]
        
        # Step 3: Normalize attention coefficients using softmax per target node
        # α_ij = softmax_j(e_ij) for all edges pointing to node i
        attention = self._softmax_per_node(e, edge_dst, num_nodes)  # [num_edges]
        
        # Apply dropout to attention coefficients
        attention = self.dropout_layer(attention)
        
        # Step 4: Aggregate neighbor features using attention weights
        # h_i_new = Σ_j α_ij * h_j'
        h_prime = self._aggregate_neighbors(h, edge_src, edge_dst, attention, num_nodes)
        
        # Step 5: Apply activation function
        if self.concat:
            # ELU activation for intermediate layers
            return F.elu(h_prime)
        else:
            # No activation for final layer
            return h_prime
    
    def _softmax_per_node(
        self,
        scores: torch.Tensor,
        target_nodes: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Apply softmax normalization per target node.
        
        For each node, normalizes attention scores of all incoming edges.
        
        Args:
            scores: Attention scores [num_edges]
            target_nodes: Target node indices [num_edges]
            num_nodes: Total number of nodes
        
        Returns:
            Normalized attention weights [num_edges]
        """
        # Compute max per target node for numerical stability
        max_scores = torch.zeros(num_nodes, device=scores.device)
        max_scores = max_scores.scatter_reduce(
            0, target_nodes, scores, reduce='amax', include_self=False
        )
        
        # Subtract max and exponentiate
        scores_shifted = scores - max_scores[target_nodes]
        exp_scores = torch.exp(scores_shifted)
        
        # Sum exp scores per target node
        sum_exp_scores = torch.zeros(num_nodes, device=scores.device)
        sum_exp_scores = sum_exp_scores.scatter_add(0, target_nodes, exp_scores)
        
        # Normalize
        attention = exp_scores / (sum_exp_scores[target_nodes] + 1e-16)
        
        return attention
    
    def _aggregate_neighbors(
        self,
        h: torch.Tensor,
        source_nodes: torch.Tensor,
        target_nodes: torch.Tensor,
        attention: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Aggregate neighbor features using attention weights.
        
        Args:
            h: Transformed node features [num_nodes, out_features]
            source_nodes: Source node indices [num_edges]
            target_nodes: Target node indices [num_edges]
            attention: Attention weights [num_edges]
            num_nodes: Total number of nodes
        
        Returns:
            Aggregated features [num_nodes, out_features]
        """
        # Weight source features by attention
        weighted_features = h[source_nodes] * attention.unsqueeze(1)  # [num_edges, out_features]
        
        # Sum weighted features per target node
        h_prime = torch.zeros(
            num_nodes, self.out_features,
            device=h.device, dtype=h.dtype
        )
        h_prime = h_prime.scatter_add(0, target_nodes.unsqueeze(1).expand_as(weighted_features), weighted_features)
        
        return h_prime
    
    def __repr__(self):
        """String representation of the layer."""
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"dropout={self.dropout}, "
            f"alpha={self.alpha})"
        )
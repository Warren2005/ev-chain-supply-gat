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
    
    def _prepare_attentional_mechanism_input(
    self,
    Wh: torch.Tensor,
    edge_index: torch.Tensor
) -> torch.Tensor:
        """
        Prepare input for attention mechanism by concatenating source and target features.
        
        For each edge (i -> j), concatenates [Wh_i || Wh_j].
        
        Args:
            Wh: Transformed node features [num_nodes, out_features]
            edge_index: Graph connectivity [2, num_edges]
        
        Returns:
            Concatenated features for attention [num_edges, 2*out_features]
        """
        source_nodes = edge_index[0]  # [num_edges]
        target_nodes = edge_index[1]  # [num_edges]
        
        # Get source and target features
        Wh_source = Wh[source_nodes]  # [num_edges, out_features]
        Wh_target = Wh[target_nodes]  # [num_edges, out_features]
        
        # Concatenate [Wh_i || Wh_j]
        a_input = torch.cat([Wh_source, Wh_target], dim=1)  # [num_edges, 2*out_features]
        
        return a_input


    def forward(
    self,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    return_attention: bool = False
) -> torch.Tensor:
        """
        Forward pass through GAT layer.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Graph connectivity [2, num_edges]
            return_attention: If True, return (output, attention_weights)
        
        Returns:
            Updated node features [num_nodes, out_features]
            If return_attention=True: (features, attention_weights)
        """
        num_nodes = x.size(0)
        
        # Linear transformation
        Wh = self.W(x)  # [num_nodes, out_features]
        
        # Prepare for attention computation
        a_input = self._prepare_attentional_mechanism_input(Wh, edge_index)
        
        # Compute attention coefficients (FIXED: use matmul, not call)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # [num_edges]
        
        # Extract node indices from edge_index
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        # Apply softmax per target node
        attention = self._softmax_per_node(e, target_nodes, num_nodes)
        
        # Store attention weights for extraction
        self.attention_weights = attention.detach()
        
        # Apply dropout
        attention_dropped = F.dropout(attention, self.dropout, training=self.training)
        
        # Aggregate neighbor features
        h_prime = self._aggregate_neighbors(
            Wh, source_nodes, target_nodes, attention_dropped, num_nodes
        )
        
        # Activation
        if self.concat:
            output = F.elu(h_prime)
        else:
            output = h_prime
        
        # Return attention if requested
        if return_attention:
            return output, attention
        
        return output
    
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
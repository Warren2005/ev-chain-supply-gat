"""
Unit tests for GAT Layer (Step 2)

Tests the single-head Graph Attention layer implementation.

Run with: pytest tests/test_gat_layer.py -v
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch
import torch.nn as nn

from models.gat_layer import GATLayer


class TestGATLayer:
    """Test suite for single-head GAT layer"""
    
    def test_initialization(self):
        """Test that GAT layer initializes correctly"""
        layer = GATLayer(in_features=13, out_features=128)
        
        assert layer.in_features == 13
        assert layer.out_features == 128
        assert layer.dropout == 0.1
        assert layer.alpha == 0.2
        assert layer.concat is True
    
    def test_custom_params(self):
        """Test initialization with custom parameters"""
        layer = GATLayer(
            in_features=64,
            out_features=256,
            dropout=0.3,
            alpha=0.15,
            concat=False
        )
        
        assert layer.in_features == 64
        assert layer.out_features == 256
        assert layer.dropout == 0.3
        assert layer.alpha == 0.15
        assert layer.concat is False
    
    def test_is_nn_module(self):
        """Test that layer is a proper PyTorch module"""
        layer = GATLayer(in_features=13, out_features=128)
        
        assert isinstance(layer, nn.Module)
        assert hasattr(layer, 'forward')
        assert hasattr(layer, 'parameters')
    
    def test_parameters_initialized(self):
        """Test that layer parameters are properly initialized"""
        layer = GATLayer(in_features=13, out_features=128)
        
        # Check W parameter exists and has correct shape
        assert hasattr(layer, 'W')
        assert layer.W.weight.shape == (128, 13)
        
        # Check attention parameter exists and has correct shape
        assert hasattr(layer, 'a')
        assert layer.a.shape == (256, 1)  # 2 * out_features
        
        # Check parameters are initialized (not all zeros)
        assert not torch.allclose(layer.W.weight, torch.zeros_like(layer.W.weight))
        assert not torch.allclose(layer.a, torch.zeros_like(layer.a))
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape"""
        layer = GATLayer(in_features=13, out_features=128)
        
        # Create dummy input
        num_nodes = 8
        x = torch.randn(num_nodes, 13)
        
        # Create edge index (simple chain: 0->1->2->...->7)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6],  # Source nodes
            [1, 2, 3, 4, 5, 6, 7]   # Target nodes
        ], dtype=torch.long)
        
        # Forward pass
        output = layer(x, edge_index)
        
        # Check output shape
        assert output.shape == (num_nodes, 128)
    
    def test_forward_pass_with_supply_chain_graph(self):
        """Test forward pass with realistic supply chain graph structure"""
        layer = GATLayer(in_features=13, out_features=128)
        
        # 8 nodes (our supply chain)
        num_nodes = 8
        x = torch.randn(num_nodes, 13)
        
        # Create edge index with 21 edges (our documented relationships)
        # Simplified version: mix of different connection patterns
        edge_index = torch.tensor([
            [7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 5, 6, 4, 3, 2],  # Source
            [3, 2, 3, 1, 2, 0, 1, 0, 0, 1, 0, 1, 0, 2, 1, 2, 1, 2, 3, 2, 3]   # Target
        ], dtype=torch.long)
        
        # Forward pass
        output = layer(x, edge_index)
        
        # Check output shape
        assert output.shape == (num_nodes, 128)
        
        # Check output is not all zeros
        assert not torch.allclose(output, torch.zeros_like(output))
    
    def test_attention_normalization(self):
        """Test that attention weights are properly normalized"""
        layer = GATLayer(in_features=13, out_features=128)
        
        num_nodes = 4
        x = torch.randn(num_nodes, 13)
        
        # Create edges where node 0 receives from nodes 1, 2, 3
        edge_index = torch.tensor([
            [1, 2, 3],  # Source nodes
            [0, 0, 0]   # All point to node 0
        ], dtype=torch.long)
        
        # Get attention scores (we'll need to modify forward to return these for testing)
        # For now, just check that forward runs without errors
        output = layer(x, edge_index)
        
        assert output.shape == (num_nodes, 128)
    
    def test_concat_activation(self):
        """Test that concat parameter controls activation"""
        layer_concat = GATLayer(in_features=13, out_features=128, concat=True)
        layer_no_concat = GATLayer(in_features=13, out_features=128, concat=False)
        
        num_nodes = 8
        x = torch.randn(num_nodes, 13)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        
        output_concat = layer_concat(x, edge_index)
        output_no_concat = layer_no_concat(x, edge_index)
        
        # Both should have same shape
        assert output_concat.shape == output_no_concat.shape
        
        # But values should differ (one has ELU, one doesn't)
        # This is probabilistic, but should hold
        assert not torch.allclose(output_concat, output_no_concat)
    
    def test_dropout_effect(self):
        """Test that dropout is applied during training"""
        layer = GATLayer(in_features=13, out_features=128, dropout=0.5)
        
        num_nodes = 8
        x = torch.randn(num_nodes, 13)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7]
        ], dtype=torch.long)
        
        # Set to training mode
        layer.train()
        
        # Multiple forward passes should give different results (due to dropout)
        output1 = layer(x, edge_index)
        output2 = layer(x, edge_index)
        
        # Should be different due to dropout
        assert not torch.allclose(output1, output2)
        
        # Set to eval mode
        layer.eval()
        
        # Now outputs should be deterministic
        output3 = layer(x, edge_index)
        output4 = layer(x, edge_index)
        
        assert torch.allclose(output3, output4)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the layer"""
        layer = GATLayer(in_features=13, out_features=128)
        
        num_nodes = 8
        x = torch.randn(num_nodes, 13, requires_grad=True)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7]
        ], dtype=torch.long)
        
        # Forward pass
        output = layer(x, edge_index)
        
        # Compute a simple loss
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert layer.W.weight.grad is not None
        assert layer.a.grad is not None
    
    def test_no_edges(self):
        """Test behavior with no edges (isolated nodes)"""
        layer = GATLayer(in_features=13, out_features=128)
        
        num_nodes = 8
        x = torch.randn(num_nodes, 13)
        
        # Empty edge index
        edge_index = torch.tensor([[], []], dtype=torch.long)
        
        # Forward pass should work (all zeros output for isolated nodes)
        output = layer(x, edge_index)
        
        assert output.shape == (num_nodes, 128)
    
    def test_self_loops(self):
        """Test that self-loops are handled correctly"""
        layer = GATLayer(in_features=13, out_features=128)
        
        num_nodes = 4
        x = torch.randn(num_nodes, 13)
        
        # Include self-loops (node points to itself)
        edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3],  # Source (includes self-loops)
            [0, 1, 1, 2, 2, 3, 3, 0]   # Target
        ], dtype=torch.long)
        
        # Forward pass
        output = layer(x, edge_index)
        
        assert output.shape == (num_nodes, 128)
    
    def test_repr(self):
        """Test string representation"""
        layer = GATLayer(in_features=13, out_features=128, dropout=0.2, alpha=0.15)
        
        repr_str = repr(layer)
        
        assert "GATLayer" in repr_str
        assert "13" in repr_str
        assert "128" in repr_str
        assert "0.2" in repr_str
        assert "0.15" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
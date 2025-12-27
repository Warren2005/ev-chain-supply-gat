"""
Unit tests for Multi-Head GAT Layer (Step 3)

Tests the multi-head Graph Attention layer implementation.

Run with: pytest tests/test_multi_head_gat.py -v
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch
import torch.nn as nn

from models.multi_head_gat import MultiHeadGATLayer


class TestMultiHeadGATLayer:
    """Test suite for multi-head GAT layer"""
    
    def test_initialization(self):
        """Test that multi-head GAT layer initializes correctly"""
        layer = MultiHeadGATLayer(
            num_heads=8,
            in_features=13,
            out_features_per_head=16
        )
        
        assert layer.num_heads == 8
        assert layer.in_features == 13
        assert layer.out_features_per_head == 16
        assert layer.dropout == 0.1
        assert layer.concat is True
    
    def test_custom_params(self):
        """Test initialization with custom parameters"""
        layer = MultiHeadGATLayer(
            num_heads=4,
            in_features=64,
            out_features_per_head=32,
            dropout=0.3,
            alpha=0.15,
            concat=False
        )
        
        assert layer.num_heads == 4
        assert layer.in_features == 64
        assert layer.out_features_per_head == 32
        assert layer.dropout == 0.3
        assert layer.concat is False
    
    def test_is_nn_module(self):
        """Test that layer is a proper PyTorch module"""
        layer = MultiHeadGATLayer(num_heads=8, in_features=13, out_features_per_head=16)
        
        assert isinstance(layer, nn.Module)
        assert hasattr(layer, 'forward')
        assert hasattr(layer, 'parameters')
    
    def test_attention_heads_created(self):
        """Test that attention heads are properly created"""
        num_heads = 8
        layer = MultiHeadGATLayer(num_heads=num_heads, in_features=13, out_features_per_head=16)
        
        assert len(layer.attention_heads) == num_heads
        
        # Check each head is a GATLayer
        from models.gat_layer import GATLayer
        for head in layer.attention_heads:
            assert isinstance(head, GATLayer)
    
    def test_forward_pass_concat(self):
        """Test forward pass with concatenation"""
        num_heads = 8
        out_per_head = 16
        layer = MultiHeadGATLayer(
            num_heads=num_heads,
            in_features=13,
            out_features_per_head=out_per_head,
            concat=True
        )
        
        num_nodes = 8
        x = torch.randn(num_nodes, 13)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7]
        ], dtype=torch.long)
        
        output = layer(x, edge_index)
        
        # Output should be concatenated: num_heads * out_per_head
        expected_dim = num_heads * out_per_head
        assert output.shape == (num_nodes, expected_dim)
    
    def test_forward_pass_average(self):
        """Test forward pass with averaging"""
        num_heads = 8
        out_per_head = 16
        layer = MultiHeadGATLayer(
            num_heads=num_heads,
            in_features=13,
            out_features_per_head=out_per_head,
            concat=False
        )
        
        num_nodes = 8
        x = torch.randn(num_nodes, 13)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7]
        ], dtype=torch.long)
        
        output = layer(x, edge_index)
        
        # Output should be averaged: just out_per_head
        assert output.shape == (num_nodes, out_per_head)
    
    def test_output_dim_property_concat(self):
        """Test output_dim property with concatenation"""
        layer = MultiHeadGATLayer(
            num_heads=8,
            in_features=13,
            out_features_per_head=16,
            concat=True
        )
        
        assert layer.output_dim == 8 * 16
    
    def test_output_dim_property_average(self):
        """Test output_dim property with averaging"""
        layer = MultiHeadGATLayer(
            num_heads=8,
            in_features=13,
            out_features_per_head=16,
            concat=False
        )
        
        assert layer.output_dim == 16
    
    def test_supply_chain_graph(self):
        """Test with realistic supply chain graph (8 nodes, 21 edges)"""
        layer = MultiHeadGATLayer(
            num_heads=8,
            in_features=13,
            out_features_per_head=16,
            concat=True
        )
        
        num_nodes = 8
        x = torch.randn(num_nodes, 13)
        
        # 21 edges (our supply chain)
        edge_index = torch.tensor([
            [7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 5],
            [4, 3, 2, 4, 3, 1, 2, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 2, 1, 2, 3]
        ], dtype=torch.long)
        
        output = layer(x, edge_index)
        
        assert output.shape == (num_nodes, 8 * 16)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gradient_flow(self):
        """Test that gradients flow through all heads"""
        layer = MultiHeadGATLayer(
            num_heads=8,
            in_features=13,
            out_features_per_head=16
        )
        
        num_nodes = 8
        x = torch.randn(num_nodes, 13, requires_grad=True)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7]
        ], dtype=torch.long)
        
        output = layer(x, edge_index)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        
        # Check all heads have gradients
        for head in layer.attention_heads:
            assert head.W.weight.grad is not None
            assert head.a.grad is not None
    
    def test_different_head_counts(self):
        """Test with different numbers of heads"""
        num_nodes = 8
        x = torch.randn(num_nodes, 13)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        
        for num_heads in [1, 4, 8, 16]:
            layer = MultiHeadGATLayer(
                num_heads=num_heads,
                in_features=13,
                out_features_per_head=16,
                concat=True
            )
            
            output = layer(x, edge_index)
            assert output.shape == (num_nodes, num_heads * 16)
    
    def test_real_world_config(self):
        """Test with real-world configuration (8 heads, 128 hidden dim)"""
        # This is our actual config: 8 heads, 128 total dim
        # So each head outputs 128 / 8 = 16 features
        layer = MultiHeadGATLayer(
            num_heads=8,
            in_features=13,
            out_features_per_head=16,
            dropout=0.1,
            concat=True
        )
        
        num_nodes = 8
        x = torch.randn(num_nodes, 13)
        edge_index = torch.randint(0, 8, (2, 21))
        
        output = layer(x, edge_index)
        
        # Total output dim should be 128 (8 heads * 16 per head)
        assert output.shape == (num_nodes, 128)
        assert layer.output_dim == 128
    
    def test_dropout_applied(self):
        """Test that dropout is applied in training mode"""
        layer = MultiHeadGATLayer(
            num_heads=4,
            in_features=13,
            out_features_per_head=16,
            dropout=0.5
        )
        
        num_nodes = 8
        x = torch.randn(num_nodes, 13)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7]
        ], dtype=torch.long)
        
        layer.train()
        output1 = layer(x, edge_index)
        output2 = layer(x, edge_index)
        
        # Should differ due to dropout
        assert not torch.allclose(output1, output2)
        
        layer.eval()
        output3 = layer(x, edge_index)
        output4 = layer(x, edge_index)
        
        # Should be same without dropout
        assert torch.allclose(output3, output4)
    
    def test_repr(self):
        """Test string representation"""
        layer = MultiHeadGATLayer(
            num_heads=8,
            in_features=13,
            out_features_per_head=16,
            concat=True
        )
        
        repr_str = repr(layer)
        
        assert "MultiHeadGATLayer" in repr_str
        assert "num_heads=8" in repr_str
        assert "in_features=13" in repr_str
        assert "out_features_per_head=16" in repr_str
        assert "output_dim=128" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
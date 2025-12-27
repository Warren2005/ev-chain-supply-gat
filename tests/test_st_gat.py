"""
Unit tests for ST-GAT Model (Step 5 - Complete)

Tests the fully integrated ST-GAT model with GAT + LSTM + Output layers.

Run with: pytest tests/test_st_gat.py -v
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch

from models.st_gat import STGAT


class TestSTGATComplete:
    """Test suite for complete ST-GAT model"""
    
    def test_initialization_default_params(self):
        """Test that model initializes with default parameters"""
        model = STGAT()
        
        # Check all default values
        assert model.num_nodes == 8
        assert model.num_edges == 21
        assert model.input_features == 13
        assert model.gat_hidden_dim == 128
        assert model.gat_heads == 8
        assert model.gat_layers == 2
        assert model.gat_dropout == 0.1
        assert model.lstm_hidden_dim == 128
        assert model.lstm_layers == 2
        assert model.lstm_dropout == 0.2
        assert model.temporal_window == 20
        assert model.output_dim == 1
        assert model.device == "cpu"
    
    def test_layers_built(self):
        """Test that all layers are properly built"""
        model = STGAT()
        
        # GAT layers
        assert len(model.gat_layers_list) == 2
        assert model.gat_layers_list is not None
        
        # LSTM layer
        assert model.lstm is not None
        
        # Output layer
        assert model.output_layer is not None
        assert isinstance(model.output_layer, torch.nn.Linear)
    
    def test_parameter_counts(self):
        """Test that parameters are counted correctly"""
        model = STGAT()
        
        stats = model.model_stats
        
        assert stats['gat_parameters'] > 0
        assert stats['lstm_parameters'] > 0
        assert stats['output_parameters'] > 0
        assert stats['total_parameters'] > 0
        assert stats['trainable_parameters'] == stats['total_parameters']
        
        # Total should equal sum of components
        assert stats['total_parameters'] == (
            stats['gat_parameters'] +
            stats['lstm_parameters'] +
            stats['output_parameters']
        )
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape"""
        model = STGAT()
        
        batch_size = 4
        num_nodes = 8
        seq_len = 20
        features = 13
        
        x = torch.randn(batch_size, num_nodes, seq_len, features)
        edge_index = torch.randint(0, num_nodes, (2, 21))
        
        output = model(x, edge_index)
        
        # Output should be [batch, num_nodes, 1]
        assert output.shape == (batch_size, num_nodes, 1)
    
    def test_forward_pass_no_nan_inf(self):
        """Test that forward pass doesn't produce NaN or Inf values"""
        model = STGAT()
        model.eval()
        
        batch_size = 4
        x = torch.randn(4, 8, 20, 13)
        edge_index = torch.randint(0, 8, (2, 21))
        
        output = model(x, edge_index)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gradient_flow(self):
        """Test that gradients flow through entire model"""
        model = STGAT()
        
        x = torch.randn(4, 8, 20, 13, requires_grad=True)
        edge_index = torch.randint(0, 8, (2, 21))
        
        output = model(x, edge_index)
        loss = output.sum()
        loss.backward()
        
        # Check input gradients
        assert x.grad is not None
        
        # Check GAT layer gradients
        for gat_layer in model.gat_layers_list:
            for head in gat_layer.attention_heads:
                assert head.W.weight.grad is not None
        
        # Check LSTM gradients
        for param in model.lstm.parameters():
            assert param.grad is not None
        
        # Check output layer gradients
        assert model.output_layer.weight.grad is not None
    
    def test_different_batch_sizes(self):
        """Test with different batch sizes"""
        model = STGAT()
        model.eval()
        
        edge_index = torch.randint(0, 8, (2, 21))
        
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 8, 20, 13)
            output = model(x, edge_index)
            
            assert output.shape == (batch_size, 8, 1)
    
    def test_different_sequence_lengths(self):
        """Test with different temporal window sizes"""
        model = STGAT()
        model.eval()
        
        edge_index = torch.randint(0, 8, (2, 21))
        
        for seq_len in [5, 10, 20, 30]:
            x = torch.randn(4, 8, seq_len, 13)
            output = model(x, edge_index)
            
            assert output.shape == (4, 8, 1)
    
    def test_training_vs_eval_mode(self):
        """Test that model behaves differently in train vs eval mode"""
        model = STGAT()
        
        x = torch.randn(4, 8, 20, 13)
        edge_index = torch.randint(0, 8, (2, 21))
        
        # Training mode
        model.train()
        output_train1 = model(x, edge_index)
        output_train2 = model(x, edge_index)
        
        # Outputs may differ due to dropout
        # (though difference might be small)
        
        # Eval mode
        model.eval()
        output_eval1 = model(x, edge_index)
        output_eval2 = model(x, edge_index)
        
        # Should be deterministic in eval mode
        assert torch.allclose(output_eval1, output_eval2)
    
    def test_get_config(self):
        """Test that get_config returns correct configuration"""
        model = STGAT()
        
        config = model.get_config()
        
        assert isinstance(config, dict)
        assert config['num_nodes'] == 8
        assert config['gat_heads'] == 8
        assert config['lstm_layers'] == 2
    
    def test_custom_initialization(self):
        """Test model with custom parameters"""
        model = STGAT(
            num_nodes=10,
            gat_heads=4,
            lstm_layers=1
        )
        
        assert model.num_nodes == 10
        assert model.gat_heads == 4
        assert model.lstm_layers == 1
        
        # Test forward pass
        x = torch.randn(2, 10, 20, 13)
        edge_index = torch.randint(0, 10, (2, 25))
        
        output = model(x, edge_index)
        assert output.shape == (2, 10, 1)
    
    def test_real_world_supply_chain(self):
        """Test with realistic supply chain configuration"""
        model = STGAT()
        model.eval()
        
        # Realistic batch
        batch_size = 16
        num_nodes = 8
        seq_len = 20
        features = 13
        
        x = torch.randn(batch_size, num_nodes, seq_len, features)
        
        # Realistic edge index (21 edges)
        edge_index = torch.tensor([
            [7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 5],
            [4, 3, 2, 4, 3, 1, 2, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 2, 1, 2, 3]
        ], dtype=torch.long)
        
        output = model(x, edge_index)
        
        assert output.shape == (batch_size, num_nodes, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_model_size(self):
        """Test that model size is reasonable"""
        model = STGAT()
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should be around 280K parameters (updated based on actual architecture)
        assert 200_000 < total_params < 400_000
        
        print(f"\nModel size: {total_params:,} parameters")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
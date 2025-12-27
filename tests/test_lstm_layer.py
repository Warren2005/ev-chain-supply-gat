"""
Unit tests for LSTM Temporal Layer (Step 4)

Tests the temporal LSTM layer implementation.

Run with: pytest tests/test_lstm_layer.py -v
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch
import torch.nn as nn

from models.lstm_layer import TemporalLSTM


class TestTemporalLSTM:
    """Test suite for Temporal LSTM layer"""
    
    def test_initialization(self):
        """Test that LSTM layer initializes correctly"""
        layer = TemporalLSTM(
            input_dim=128,
            hidden_dim=128,
            num_layers=2,
            dropout=0.2
        )
        
        assert layer.input_dim == 128
        assert layer.hidden_dim == 128
        assert layer.num_layers == 2
        assert layer.dropout == 0.2
        assert layer.bidirectional is False
        assert layer.output_dim == 128
    
    def test_custom_params(self):
        """Test initialization with custom parameters"""
        layer = TemporalLSTM(
            input_dim=64,
            hidden_dim=256,
            num_layers=1,
            dropout=0.3,
            bidirectional=True
        )
        
        assert layer.input_dim == 64
        assert layer.hidden_dim == 256
        assert layer.num_layers == 1
        assert layer.dropout == 0.3
        assert layer.bidirectional is True
        assert layer.output_dim == 512  # 256 * 2 for bidirectional
    
    def test_is_nn_module(self):
        """Test that layer is a proper PyTorch module"""
        layer = TemporalLSTM(input_dim=128, hidden_dim=128)
        
        assert isinstance(layer, nn.Module)
        assert hasattr(layer, 'forward')
        assert hasattr(layer, 'parameters')
    
    def test_lstm_initialized(self):
        """Test that LSTM module is properly initialized"""
        layer = TemporalLSTM(input_dim=128, hidden_dim=128, num_layers=2)
        
        assert hasattr(layer, 'lstm')
        assert isinstance(layer.lstm, nn.LSTM)
        assert layer.lstm.input_size == 128
        assert layer.lstm.hidden_size == 128
        assert layer.lstm.num_layers == 2
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape"""
        layer = TemporalLSTM(input_dim=128, hidden_dim=128, num_layers=2)
        
        batch_size = 4
        num_nodes = 8
        seq_len = 20
        features = 128
        
        x = torch.randn(batch_size, num_nodes, seq_len, features)
        
        output, (h_n, c_n) = layer(x)
        
        # Output should be [batch, num_nodes, hidden_dim]
        assert output.shape == (batch_size, num_nodes, 128)
        
        # Hidden state shape: [num_layers * num_directions, batch * num_nodes, hidden_dim]
        assert h_n.shape == (2, batch_size * num_nodes, 128)
        assert c_n.shape == (2, batch_size * num_nodes, 128)
    
    def test_forward_pass_bidirectional(self):
        """Test forward pass with bidirectional LSTM"""
        layer = TemporalLSTM(
            input_dim=128,
            hidden_dim=128,
            num_layers=2,
            bidirectional=True
        )
        
        batch_size = 4
        num_nodes = 8
        seq_len = 20
        
        x = torch.randn(batch_size, num_nodes, seq_len, 128)
        
        output, (h_n, c_n) = layer(x)
        
        # Output should be doubled for bidirectional
        assert output.shape == (batch_size, num_nodes, 256)  # 128 * 2
        
        # Hidden states: [num_layers * 2, batch * num_nodes, hidden_dim]
        assert h_n.shape == (4, batch_size * num_nodes, 128)  # 2 layers * 2 directions
        assert c_n.shape == (4, batch_size * num_nodes, 128)
    
    def test_real_world_config(self):
        """Test with real-world ST-GAT configuration"""
        layer = TemporalLSTM(
            input_dim=128,  # From GAT output
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
            bidirectional=False
        )
        
        # Typical batch for our supply chain
        batch_size = 16
        num_nodes = 8
        seq_len = 20  # Temporal window
        
        x = torch.randn(batch_size, num_nodes, seq_len, 128)
        
        output, _ = layer(x)
        
        assert output.shape == (batch_size, num_nodes, 128)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths"""
        layer = TemporalLSTM(input_dim=128, hidden_dim=128)
        
        batch_size = 4
        num_nodes = 8
        
        for seq_len in [5, 10, 20, 30]:
            x = torch.randn(batch_size, num_nodes, seq_len, 128)
            output, _ = layer(x)
            
            assert output.shape == (batch_size, num_nodes, 128)
    
    def test_single_timestep(self):
        """Test with single timestep (edge case)"""
        layer = TemporalLSTM(input_dim=128, hidden_dim=128)
        
        batch_size = 4
        num_nodes = 8
        seq_len = 1
        
        x = torch.randn(batch_size, num_nodes, seq_len, 128)
        output, _ = layer(x)
        
        assert output.shape == (batch_size, num_nodes, 128)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the layer"""
        layer = TemporalLSTM(input_dim=128, hidden_dim=128, num_layers=2)
        
        batch_size = 4
        num_nodes = 8
        seq_len = 20
        
        x = torch.randn(batch_size, num_nodes, seq_len, 128, requires_grad=True)
        
        output, _ = layer(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        
        # Check LSTM parameters have gradients
        for name, param in layer.lstm.named_parameters():
            assert param.grad is not None
    
    def test_custom_initial_states(self):
        """Test forward pass with custom initial hidden/cell states"""
        layer = TemporalLSTM(input_dim=128, hidden_dim=128, num_layers=2)
        
        batch_size = 4
        num_nodes = 8
        seq_len = 20
        
        x = torch.randn(batch_size, num_nodes, seq_len, 128)
        
        # Custom initial states
        h0 = torch.randn(2, batch_size * num_nodes, 128)
        c0 = torch.randn(2, batch_size * num_nodes, 128)
        
        output, (h_n, c_n) = layer(x, h0, c0)
        
        assert output.shape == (batch_size, num_nodes, 128)
        assert h_n.shape == h0.shape
        assert c_n.shape == c0.shape
    
    def test_output_dim_property(self):
        """Test get_output_dim method"""
        layer_uni = TemporalLSTM(input_dim=128, hidden_dim=128, bidirectional=False)
        layer_bi = TemporalLSTM(input_dim=128, hidden_dim=128, bidirectional=True)
        
        assert layer_uni.get_output_dim() == 128
        assert layer_bi.get_output_dim() == 256
    
    def test_dropout_single_layer(self):
        """Test that dropout is 0 for single layer LSTM"""
        layer = TemporalLSTM(input_dim=128, hidden_dim=128, num_layers=1, dropout=0.5)
        
        # Single layer LSTM should have dropout=0 (PyTorch requirement)
        assert layer.lstm.dropout == 0.0
    
    def test_dropout_multi_layer(self):
        """Test that dropout is applied for multi-layer LSTM"""
        layer = TemporalLSTM(input_dim=128, hidden_dim=128, num_layers=2, dropout=0.3)
        
        # Multi-layer LSTM should have the specified dropout
        assert layer.lstm.dropout == 0.3
    
    def test_deterministic_eval_mode(self):
        """Test that output is deterministic in eval mode"""
        layer = TemporalLSTM(input_dim=128, hidden_dim=128, num_layers=2, dropout=0.5)
        
        batch_size = 4
        num_nodes = 8
        seq_len = 20
        
        x = torch.randn(batch_size, num_nodes, seq_len, 128)
        
        layer.eval()
        
        output1, _ = layer(x)
        output2, _ = layer(x)
        
        # Should be deterministic in eval mode
        assert torch.allclose(output1, output2)
    
    def test_different_batch_sizes(self):
        """Test with different batch sizes"""
        layer = TemporalLSTM(input_dim=128, hidden_dim=128)
        
        num_nodes = 8
        seq_len = 20
        
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, num_nodes, seq_len, 128)
            output, _ = layer(x)
            
            assert output.shape == (batch_size, num_nodes, 128)
    
    def test_repr(self):
        """Test string representation"""
        layer = TemporalLSTM(
            input_dim=128,
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
            bidirectional=False
        )
        
        repr_str = repr(layer)
        
        assert "TemporalLSTM" in repr_str
        assert "input_dim=128" in repr_str
        assert "hidden_dim=128" in repr_str
        assert "num_layers=2" in repr_str
        assert "dropout=0.2" in repr_str
        assert "output_dim=128" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
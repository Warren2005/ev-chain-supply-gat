"""
Quick validation script for LSTM Temporal Layer (Step 4)

This script performs a quick test to verify the LSTM layer is working correctly.

Run this script to test before proceeding to Step 5.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from models.lstm_layer import TemporalLSTM


def main():
    """Run quick validation test for LSTM layer"""
    print("="*70)
    print("LSTM TEMPORAL LAYER - QUICK VALIDATION TEST (STEP 4)")
    print("="*70)
    print()
    
    test_results = []
    
    # Test 1: Initialize layer
    print("Test 1: Initializing LSTM layer...")
    try:
        layer = TemporalLSTM(
            input_dim=128,
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
            bidirectional=False
        )
        print("✓ LSTM layer initialized successfully")
        print(f"  {layer}")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        test_results.append(False)
        return False
    
    print()
    
    # Test 2: Check parameters
    print("Test 2: Verifying layer parameters...")
    try:
        total_params = sum(p.numel() for p in layer.parameters())
        print(f"  Total parameters: {total_params:,}")
        print(f"  Input dim: {layer.input_dim}")
        print(f"  Hidden dim: {layer.hidden_dim}")
        print(f"  Num layers: {layer.num_layers}")
        print(f"  Output dim: {layer.output_dim}")
        
        assert layer.lstm is not None
        assert isinstance(layer.lstm, torch.nn.LSTM)
        
        print("✓ Parameters verified correctly")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Parameter check failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 3: Forward pass with real-world dimensions
    print("Test 3: Testing forward pass (real-world config)...")
    try:
        batch_size = 16
        num_nodes = 8
        seq_len = 20  # Temporal window
        features = 128  # From GAT output
        
        x = torch.randn(batch_size, num_nodes, seq_len, features)
        
        output, (h_n, c_n) = layer(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Hidden state shape: {h_n.shape}")
        print(f"  Cell state shape: {c_n.shape}")
        
        assert output.shape == (batch_size, num_nodes, 128)
        assert h_n.shape == (2, batch_size * num_nodes, 128)
        assert c_n.shape == (2, batch_size * num_nodes, 128)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        print("✓ Forward pass successful")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 4: Different sequence lengths
    print("Test 4: Testing with different sequence lengths...")
    try:
        for seq_len in [5, 10, 20, 30]:
            x_test = torch.randn(4, 8, seq_len, 128)
            output_test, _ = layer(x_test)
            assert output_test.shape == (4, 8, 128)
        
        print("  ✓ Seq len 5: passed")
        print("  ✓ Seq len 10: passed")
        print("  ✓ Seq len 20: passed")
        print("  ✓ Seq len 30: passed")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Sequence length test failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 5: Gradient flow
    print("Test 5: Testing gradient flow...")
    try:
        x_grad = torch.randn(batch_size, num_nodes, seq_len, features, requires_grad=True)
        output = layer(x_grad)[0]
        loss = output.sum()
        loss.backward()
        
        assert x_grad.grad is not None
        
        # Check LSTM parameters have gradients
        param_grads = []
        for name, param in layer.lstm.named_parameters():
            if param.grad is not None:
                param_grads.append(param.grad.norm().item())
        
        print(f"  Input gradient norm: {x_grad.grad.norm().item():.4f}")
        print(f"  LSTM parameters with gradients: {len(param_grads)}")
        print(f"  Average parameter gradient norm: {sum(param_grads)/len(param_grads):.4f}")
        
        print("✓ Gradients computed successfully")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 6: Bidirectional LSTM
    print("Test 6: Testing bidirectional LSTM...")
    try:
        layer_bi = TemporalLSTM(
            input_dim=128,
            hidden_dim=128,
            num_layers=2,
            bidirectional=True
        )
        
        x_bi = torch.randn(4, 8, 20, 128)
        output_bi, (h_n_bi, c_n_bi) = layer_bi(x_bi)
        
        print(f"  Output shape: {output_bi.shape}")
        print(f"  Expected output dim: 256 (128 * 2 directions)")
        print(f"  Actual output dim: {output_bi.shape[2]}")
        
        assert output_bi.shape == (4, 8, 256)  # 128 * 2 for bidirectional
        assert h_n_bi.shape == (4, 32, 128)  # 2 layers * 2 directions
        
        print("✓ Bidirectional LSTM working correctly")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Bidirectional test failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 7: Deterministic in eval mode
    print("Test 7: Testing deterministic behavior in eval mode...")
    try:
        layer.eval()
        
        x_test = torch.randn(4, 8, 20, 128)
        output1, _ = layer(x_test)
        output2, _ = layer(x_test)
        
        diff = (output1 - output2).abs().max().item()
        
        print(f"  Max difference between runs: {diff:.10f}")
        
        assert torch.allclose(output1, output2)
        print("✓ Deterministic in eval mode")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Deterministic test failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 8: Integration with GAT output
    print("Test 8: Testing integration with GAT output dimensions...")
    try:
        # Simulate GAT output: [batch, nodes, seq_len, gat_output_dim]
        gat_output = torch.randn(16, 8, 20, 128)
        
        lstm_output, _ = layer(gat_output)
        
        print(f"  GAT output (input): {gat_output.shape}")
        print(f"  LSTM output: {lstm_output.shape}")
        print(f"  Ready for final prediction layer")
        
        assert lstm_output.shape == (16, 8, 128)
        print("✓ Integration with GAT output successful")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        test_results.append(False)
    
    print()
    print("="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if all(test_results):
        print()
        print("✓ ALL VALIDATION TESTS PASSED!")
        print()
        print("LSTM temporal layer is working correctly.")
        print("Ready to proceed to Step 5: Integration & Forward Pass")
        print()
        return True
    else:
        print()
        print("✗ SOME VALIDATION TESTS FAILED")
        print()
        print("Please check the errors above and fix before proceeding.")
        print()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
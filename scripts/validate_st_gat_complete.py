"""
Quick validation script for Complete ST-GAT Model (Step 5)

This script performs a quick test to verify the complete integrated model works.

Run this script to test the final ST-GAT implementation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from models.st_gat import STGAT


def main():
    """Run quick validation test for complete ST-GAT model"""
    print("="*70)
    print("COMPLETE ST-GAT MODEL - QUICK VALIDATION TEST (STEP 5)")
    print("="*70)
    print()
    
    test_results = []
    
    # Test 1: Initialize complete model
    print("Test 1: Initializing complete ST-GAT model...")
    try:
        model = STGAT()
        print("✓ Complete model initialized successfully")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        test_results.append(False)
        return False
    
    print()
    
    # Test 2: Verify all layers built
    print("Test 2: Verifying all layers are built...")
    try:
        print(f"  GAT layers: {len(model.gat_layers_list)}")
        print(f"  LSTM layer: {'✓' if model.lstm is not None else '✗'}")
        print(f"  Output layer: {'✓' if model.output_layer is not None else '✗'}")
        
        assert len(model.gat_layers_list) == 2
        assert model.lstm is not None
        assert model.output_layer is not None
        
        print("✓ All layers built correctly")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Layer verification failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 3: Check parameter counts
    print("Test 3: Checking model parameters...")
    try:
        stats = model.model_stats
        
        print(f"  GAT parameters: {stats['gat_parameters']:,}")
        print(f"  LSTM parameters: {stats['lstm_parameters']:,}")
        print(f"  Output parameters: {stats['output_parameters']:,}")
        print(f"  Total parameters: {stats['total_parameters']:,}")
        
        assert stats['total_parameters'] > 0
        assert stats['total_parameters'] == (
            stats['gat_parameters'] +
            stats['lstm_parameters'] +
            stats['output_parameters']
        )
        
        print("✓ Parameter counts verified")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Parameter count failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 4: End-to-end forward pass
    print("Test 4: Testing end-to-end forward pass...")
    try:
        batch_size = 16
        num_nodes = 8
        seq_len = 20
        features = 13
        
        x = torch.randn(batch_size, num_nodes, seq_len, features)
        edge_index = torch.randint(0, num_nodes, (2, 21))
        
        model.eval()
        output = model(x, edge_index)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Edge index shape: {edge_index.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        assert output.shape == (batch_size, num_nodes, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        print("✓ Forward pass successful")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 5: Supply chain graph with real edges
    print("Test 5: Testing with realistic supply chain graph...")
    try:
        edge_index_sc = torch.tensor([
            [7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 5],
            [4, 3, 2, 4, 3, 1, 2, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 2, 1, 2, 3]
        ], dtype=torch.long)
        
        output_sc = model(x, edge_index_sc)
        
        print(f"  Output shape: {output_sc.shape}")
        print(f"  Per-node predictions (first batch):")
        for node_idx in range(8):
            pred = output_sc[0, node_idx, 0].item()
            print(f"    Node {node_idx}: {pred:.6f}")
        
        assert output_sc.shape == (batch_size, num_nodes, 1)
        print("✓ Supply chain graph processing successful")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Supply chain test failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 6: Gradient flow through entire model
    print("Test 6: Testing gradient flow...")
    try:
        x_grad = torch.randn(4, 8, 20, 13, requires_grad=True)
        edge_index = torch.randint(0, 8, (2, 21))
        
        output = model(x_grad, edge_index)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x_grad.grad is not None
        
        # Count parameters with gradients
        params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for p in model.parameters())
        
        print(f"  Input gradient norm: {x_grad.grad.norm().item():.4f}")
        print(f"  Parameters with gradients: {params_with_grad}/{total_params}")
        
        assert params_with_grad == total_params
        print("✓ Gradients flowing through all parameters")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 7: Different batch sizes
    print("Test 7: Testing with different batch sizes...")
    try:
        edge_index = torch.randint(0, 8, (2, 21))
        
        for batch_size in [1, 4, 16, 32]:
            x_test = torch.randn(batch_size, 8, 20, 13)
            output_test = model(x_test, edge_index)
            assert output_test.shape == (batch_size, 8, 1)
        
        print("  ✓ Batch size 1: passed")
        print("  ✓ Batch size 4: passed")
        print("  ✓ Batch size 16: passed")
        print("  ✓ Batch size 32: passed")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Batch size test failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 8: Train vs Eval mode
    print("Test 8: Testing train vs eval mode...")
    try:
        x_test = torch.randn(4, 8, 20, 13)
        edge_index = torch.randint(0, 8, (2, 21))
        
        # Eval mode should be deterministic
        model.eval()
        output1 = model(x_test, edge_index)
        output2 = model(x_test, edge_index)
        
        diff = (output1 - output2).abs().max().item()
        
        print(f"  Max difference in eval mode: {diff:.10f}")
        
        assert torch.allclose(output1, output2)
        print("✓ Deterministic behavior in eval mode")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Train/eval mode test failed: {e}")
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
        print("=" *70)
        print("ST-GAT MODEL COMPLETE!")
        print("="*70)
        print()
        print("The complete Spatio-Temporal Graph Attention Network is working.")
        print()
        print("Model summary:")
        print(f"  • {stats['total_parameters']:,} total parameters")
        print(f"  • 8 nodes (companies) in supply chain")
        print(f"  • 21 directed edges (supply relationships)")
        print(f"  • 13 input features per node")
        print(f"  • 20-day temporal window")
        print(f"  • Predicts next-day volatility")
        print()
        print("Ready for Phase 4: Model Training!")
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
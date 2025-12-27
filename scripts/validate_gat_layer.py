"""
Quick validation script for GAT Layer (Step 2)

This script performs a quick test to verify the GAT layer is working correctly.

Run this script to test before proceeding to Step 3.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from models.gat_layer import GATLayer


def main():
    """Run quick validation test for GAT layer"""
    print("="*70)
    print("GAT LAYER - QUICK VALIDATION TEST (STEP 2)")
    print("="*70)
    print()
    
    test_results = []
    
    # Test 1: Initialize layer
    print("Test 1: Initializing GAT layer...")
    try:
        layer = GATLayer(in_features=13, out_features=128, dropout=0.1)
        print("✓ GAT layer initialized successfully")
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
        print(f"  W shape: {layer.W.weight.shape}")
        print(f"  a shape: {layer.a.shape}")
        
        # Check parameters are initialized (not all zeros)
        assert not torch.allclose(layer.W.weight, torch.zeros_like(layer.W.weight))
        assert not torch.allclose(layer.a, torch.zeros_like(layer.a))
        print("✓ Parameters initialized correctly")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Parameter check failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 3: Forward pass with simple graph
    print("Test 3: Testing forward pass with simple chain graph...")
    try:
        num_nodes = 8
        x = torch.randn(num_nodes, 13)
        
        # Simple chain: 0->1->2->3->4->5->6->7
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7]
        ], dtype=torch.long)
        
        output = layer(x, edge_index)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Edge index shape: {edge_index.shape}")
        print(f"  Output shape: {output.shape}")
        
        assert output.shape == (num_nodes, 128)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        print("✓ Forward pass successful")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 4: Forward pass with supply chain graph structure
    print("Test 4: Testing with realistic supply chain graph (21 edges)...")
    try:
        # Create supply chain-like graph with 21 edges
        edge_index = torch.tensor([
            # Tier 3 -> Tier 2 -> Tier 0 (simplified)
            [7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 5],
            [4, 3, 2, 4, 3, 1, 2, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 2, 1, 2, 3]
        ], dtype=torch.long)
        
        output = layer(x, edge_index)
        
        print(f"  Number of edges: {edge_index.shape[1]}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output mean: {output.mean().item():.4f}")
        print(f"  Output std: {output.std().item():.4f}")
        
        assert output.shape == (num_nodes, 128)
        print("✓ Supply chain graph forward pass successful")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Supply chain forward pass failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 5: Gradient flow
    print("Test 5: Testing gradient flow...")
    try:
        x_grad = torch.randn(num_nodes, 13, requires_grad=True)
        output = layer(x_grad, edge_index)
        loss = output.sum()
        loss.backward()
        
        assert x_grad.grad is not None
        assert layer.W.weight.grad is not None
        assert layer.a.grad is not None
        
        print(f"  Input gradient norm: {x_grad.grad.norm().item():.4f}")
        print(f"  W gradient norm: {layer.W.weight.grad.norm().item():.4f}")
        print(f"  a gradient norm: {layer.a.grad.norm().item():.4f}")
        print("✓ Gradients computed successfully")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 6: Dropout effect
    print("Test 6: Testing dropout functionality...")
    try:
        layer.train()  # Training mode (dropout active)
        output1 = layer(x, edge_index)
        output2 = layer(x, edge_index)
        
        # Should be different due to dropout
        train_diff = (output1 - output2).abs().mean().item()
        
        layer.eval()  # Eval mode (dropout off)
        output3 = layer(x, edge_index)
        output4 = layer(x, edge_index)
        
        # Should be same without dropout
        eval_diff = (output3 - output4).abs().mean().item()
        
        print(f"  Training mode difference: {train_diff:.6f}")
        print(f"  Eval mode difference: {eval_diff:.6f}")
        
        assert train_diff > eval_diff
        print("✓ Dropout working correctly")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Dropout test failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 7: Multiple output dimensions
    print("Test 7: Testing different output dimensions...")
    try:
        for out_dim in [64, 128, 256]:
            test_layer = GATLayer(in_features=13, out_features=out_dim)
            test_output = test_layer(x, edge_index)
            assert test_output.shape == (num_nodes, out_dim)
        
        print("  ✓ Output dim 64: passed")
        print("  ✓ Output dim 128: passed")
        print("  ✓ Output dim 256: passed")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Multiple dimensions test failed: {e}")
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
        print("GAT layer is working correctly.")
        print("Ready to proceed to Step 3: Multi-Head GAT")
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
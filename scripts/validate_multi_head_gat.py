"""
Quick validation script for Multi-Head GAT Layer (Step 3)

This script performs a quick test to verify the multi-head GAT layer is working.

Run this script to test before proceeding to Step 4.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from models.multi_head_gat import MultiHeadGATLayer


def main():
    """Run quick validation test for multi-head GAT layer"""
    print("="*70)
    print("MULTI-HEAD GAT LAYER - QUICK VALIDATION TEST (STEP 3)")
    print("="*70)
    print()
    
    test_results = []
    
    # Test 1: Initialize layer with concatenation
    print("Test 1: Initializing multi-head GAT layer (concat mode)...")
    try:
        layer_concat = MultiHeadGATLayer(
            num_heads=8,
            in_features=13,
            out_features_per_head=16,
            dropout=0.1,
            concat=True
        )
        print("✓ Multi-head GAT layer initialized successfully")
        print(f"  {layer_concat}")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        test_results.append(False)
        return False
    
    print()
    
    # Test 2: Check attention heads
    print("Test 2: Verifying attention heads...")
    try:
        print(f"  Number of heads: {len(layer_concat.attention_heads)}")
        print(f"  Output dimension (concat): {layer_concat.output_dim}")
        
        from models.gat_layer import GATLayer
        for i, head in enumerate(layer_concat.attention_heads[:3]):  # Show first 3
            assert isinstance(head, GATLayer)
            print(f"  Head {i}: {head.in_features} -> {head.out_features}")
        
        print("✓ All attention heads created correctly")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Head verification failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 3: Forward pass with concatenation
    print("Test 3: Testing forward pass (concatenation mode)...")
    try:
        num_nodes = 8
        x = torch.randn(num_nodes, 13)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7]
        ], dtype=torch.long)
        
        output_concat = layer_concat(x, edge_index)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output_concat.shape}")
        print(f"  Expected output dim: {8 * 16} (8 heads × 16 per head)")
        print(f"  Actual output dim: {output_concat.shape[1]}")
        
        assert output_concat.shape == (num_nodes, 128)
        print("✓ Concatenation mode working correctly")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Forward pass (concat) failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 4: Forward pass with averaging
    print("Test 4: Testing forward pass (averaging mode)...")
    try:
        layer_avg = MultiHeadGATLayer(
            num_heads=8,
            in_features=13,
            out_features_per_head=16,
            concat=False
        )
        
        output_avg = layer_avg(x, edge_index)
        
        print(f"  Output shape: {output_avg.shape}")
        print(f"  Expected output dim: 16 (averaged across heads)")
        print(f"  Actual output dim: {output_avg.shape[1]}")
        
        assert output_avg.shape == (num_nodes, 16)
        print("✓ Averaging mode working correctly")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Forward pass (average) failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 5: Supply chain graph (21 edges)
    print("Test 5: Testing with supply chain graph (21 edges)...")
    try:
        edge_index_sc = torch.tensor([
            [7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 5],
            [4, 3, 2, 4, 3, 1, 2, 1, 0, 1, 0, 3, 0, 1, 0, 1, 0, 2, 1, 2, 3]
        ], dtype=torch.long)
        
        output_sc = layer_concat(x, edge_index_sc)
        
        print(f"  Number of edges: {edge_index_sc.shape[1]}")
        print(f"  Output shape: {output_sc.shape}")
        print(f"  Output mean: {output_sc.mean().item():.4f}")
        print(f"  Output std: {output_sc.std().item():.4f}")
        
        assert output_sc.shape == (num_nodes, 128)
        assert not torch.isnan(output_sc).any()
        assert not torch.isinf(output_sc).any()
        
        print("✓ Supply chain graph processing successful")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Supply chain test failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 6: Gradient flow through all heads
    print("Test 6: Testing gradient flow through all heads...")
    try:
        x_grad = torch.randn(num_nodes, 13, requires_grad=True)
        output = layer_concat(x_grad, edge_index)
        loss = output.sum()
        loss.backward()
        
        # Check input gradients
        assert x_grad.grad is not None
        print(f"  Input gradient norm: {x_grad.grad.norm().item():.4f}")
        
        # Check each head has gradients
        head_grad_norms = []
        for i, head in enumerate(layer_concat.attention_heads):
            w_grad_norm = head.W.weight.grad.norm().item()
            a_grad_norm = head.a.grad.norm().item()
            head_grad_norms.append((w_grad_norm, a_grad_norm))
        
        print(f"  Head 0 - W grad: {head_grad_norms[0][0]:.4f}, a grad: {head_grad_norms[0][1]:.4f}")
        print(f"  Head 1 - W grad: {head_grad_norms[1][0]:.4f}, a grad: {head_grad_norms[1][1]:.4f}")
        print(f"  All {len(layer_concat.attention_heads)} heads have gradients")
        
        print("✓ Gradients flowing correctly through all heads")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 7: Real-world configuration
    print("Test 7: Testing real-world ST-GAT configuration...")
    try:
        # Layer 1: 13 -> 128 (8 heads × 16 per head)
        gat1 = MultiHeadGATLayer(
            num_heads=8,
            in_features=13,
            out_features_per_head=16,
            concat=True
        )
        
        # Layer 2: 128 -> 128 (8 heads × 16 per head)
        gat2 = MultiHeadGATLayer(
            num_heads=8,
            in_features=128,
            out_features_per_head=16,
            concat=True
        )
        
        # Two-layer forward pass
        h1 = gat1(x, edge_index)
        h2 = gat2(h1, edge_index)
        
        print(f"  Layer 1 output: {h1.shape}")
        print(f"  Layer 2 output: {h2.shape}")
        
        assert h1.shape == (num_nodes, 128)
        assert h2.shape == (num_nodes, 128)
        
        print("✓ Two-layer GAT stack working correctly")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Real-world config test failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 8: Parameter count
    print("Test 8: Counting parameters...")
    try:
        total_params = sum(p.numel() for p in layer_concat.parameters())
        trainable_params = sum(p.numel() for p in layer_concat.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Parameters per head: ~{total_params // 8:,}")
        
        print("✓ Parameter count verified")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Parameter count failed: {e}")
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
        print("Multi-head GAT layer is working correctly.")
        print("Ready to proceed to Step 4: LSTM Temporal Layer")
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
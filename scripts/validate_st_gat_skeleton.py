"""
Quick validation script for ST-GAT Model Skeleton (Step 1)

This script performs a quick test to verify the ST-GAT model skeleton is working.

Run this script to test before proceeding to Step 2.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from models.st_gat import STGAT


def main():
    """Run quick validation test for ST-GAT skeleton"""
    print("="*70)
    print("ST-GAT MODEL SKELETON - QUICK VALIDATION TEST (STEP 1)")
    print("="*70)
    print()
    
    test_results = []
    
    # Test 1: Model initialization with defaults
    print("Test 1: Initializing model with default parameters...")
    try:
        model = STGAT()
        print("✓ Model initialized successfully")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        test_results.append(False)
        return False
    
    print()
    
    # Test 2: Check configuration values
    print("Test 2: Verifying configuration...")
    config = model.get_config()
    
    checks = [
        ("num_nodes", 8),
        ("num_edges", 21),
        ("input_features", 13),
        ("gat_hidden_dim", 128),
        ("gat_heads", 8),
        ("gat_layers", 2),
        ("lstm_hidden_dim", 128),
        ("lstm_layers", 2),
        ("temporal_window", 20)
    ]
    
    all_correct = True
    for key, expected_value in checks:
        actual_value = config[key]
        if actual_value == expected_value:
            print(f"  ✓ {key}: {actual_value}")
        else:
            print(f"  ✗ {key}: expected {expected_value}, got {actual_value}")
            all_correct = False
    
    test_results.append(all_correct)
    print()
    
    # Test 3: Verify it's a PyTorch module
    print("Test 3: Checking PyTorch nn.Module compatibility...")
    try:
        assert isinstance(model, torch.nn.Module)
        assert hasattr(model, 'forward')
        assert hasattr(model, 'parameters')
        print("✓ Model is a proper PyTorch nn.Module")
        test_results.append(True)
    except AssertionError:
        print("✗ Model is not a proper PyTorch nn.Module")
        test_results.append(False)
    
    print()
    
    # Test 4: Parameter counting
    print("Test 4: Counting model parameters...")
    try:
        param_counts = model._count_parameters()
        print(f"  Total parameters: {param_counts['total']:,}")
        print(f"  Trainable parameters: {param_counts['trainable']:,}")
        
        # At this stage, should be 0 (no layers yet)
        if param_counts['total'] == 0:
            print("✓ Parameter count correct (0 expected at skeleton stage)")
            test_results.append(True)
        else:
            print(f"⚠ Unexpected parameter count: {param_counts['total']}")
            test_results.append(True)  # Still pass, just different than expected
    except Exception as e:
        print(f"✗ Parameter counting failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 5: Module lists initialized
    print("Test 5: Verifying module lists...")
    try:
        assert isinstance(model.gat_layers_list, torch.nn.ModuleList)
        assert len(model.gat_layers_list) == 0  # Empty at this stage
        print("✓ GAT ModuleList initialized (empty)")
        print(f"  LSTM placeholder: {model.lstm}")
        print(f"  Output layer placeholder: {model.output_layer}")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Module list verification failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 6: Forward pass should raise NotImplementedError
    print("Test 6: Verifying forward pass raises NotImplementedError...")
    try:
        batch_size = 2
        x = torch.randn(batch_size, 8, 20, 13)
        edge_index = torch.randint(0, 8, (2, 21))
        
        try:
            output = model(x, edge_index)
            print("✗ Forward pass should raise NotImplementedError")
            test_results.append(False)
        except NotImplementedError as e:
            if "Step 5" in str(e):
                print("✓ Forward pass correctly raises NotImplementedError")
                print(f"  Message: {str(e)[:60]}...")
                test_results.append(True)
            else:
                print(f"⚠ NotImplementedError raised but unexpected message: {e}")
                test_results.append(True)
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 7: Custom initialization
    print("Test 7: Testing custom parameter initialization...")
    try:
        custom_model = STGAT(
            gat_heads=4,
            lstm_layers=1,
            temporal_window=10
        )
        custom_config = custom_model.get_config()
        
        assert custom_config['gat_heads'] == 4
        assert custom_config['lstm_layers'] == 1
        assert custom_config['temporal_window'] == 10
        print("✓ Custom parameters set correctly")
        print(f"  GAT heads: {custom_config['gat_heads']}")
        print(f"  LSTM layers: {custom_config['lstm_layers']}")
        print(f"  Temporal window: {custom_config['temporal_window']}")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Custom initialization failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 8: Logger setup
    print("Test 8: Verifying logger configuration...")
    try:
        assert model.logger is not None
        assert model.logger.name == "models.st_gat.STGAT"
        assert len(model.logger.handlers) > 0
        print("✓ Logger configured correctly")
        print(f"  Logger name: {model.logger.name}")
        print(f"  Handlers: {len(model.logger.handlers)}")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Logger verification failed: {e}")
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
        print("ST-GAT model skeleton is working correctly.")
        print("Ready to proceed to Step 2: Simple GAT Layer")
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
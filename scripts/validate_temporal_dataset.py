"""
Quick validation script for Temporal Graph Dataset

This script tests the dataset with real processed data files.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from utils.temporal_dataset import TemporalGraphDataset


def main():
    """Run validation test for TemporalGraphDataset"""
    print("="*70)
    print("TEMPORAL GRAPH DATASET - VALIDATION TEST")
    print("="*70)
    print()
    
    # Define paths - use FILTERED data (7 stocks, no RIVN)
    data_dir = project_root / "data" / "processed"
    train_path = data_dir / "train_features_filtered.parquet"
    
    # Edge index for 7 stocks (excluding RIVN)
    # Stock order: 0:ALB, 1:APTV, 2:F, 3:GM, 4:MGA, 5:SQM, 6:TSLA
    edge_index = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 4, 4, 4, 0], [6, 2, 3, 4, 1, 2, 3, 6, 6, 2, 3, 1]], dtype=torch.long)
    
    test_results = []
    
    # Test 1: Initialize dataset
    print("Test 1: Initializing dataset with training data...")
    try:
        dataset = TemporalGraphDataset(
            data_path=str(train_path),
            edge_index=edge_index,
            window_size=20,
        )
        print(f"✓ Dataset initialized: {len(dataset)} samples")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        test_results.append(False)
        return False
    
    print()
    
    # Test 2: Check dataset statistics
    print("Test 2: Checking dataset statistics...")
    try:
        stats = dataset.get_stats()
        
        print(f"  Number of samples: {stats['num_samples']}")
        print(f"  Window size: {stats['window_size']}")
        print(f"  Number of nodes: {stats['num_nodes']}")
        print(f"  Number of edges: {stats['num_edges']}")
        print(f"  Number of features: {stats['num_features']}")
        print(f"  Number of dates: {stats['num_dates']}")
        print(f"  Target feature: {stats['target_feature']}")
        
        assert stats['num_samples'] > 0
        print("✓ Statistics retrieved successfully")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Statistics check failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 3: Get sample and check shapes
    print("Test 3: Getting sample and checking shapes...")
    try:
        features, edge_idx, target = dataset[0]
        
        print(f"  Features shape: {features.shape}")
        print(f"  Edge index shape: {edge_idx.shape}")
        print(f"  Target shape: {target.shape}")
        
        # Check shapes (use actual sample size, not auto-detected num_nodes)
        num_stocks_in_sample = features.shape[0]
        assert num_stocks_in_sample == 7, f"Expected 7 stocks in sample, got {num_stocks_in_sample}"
        assert features.shape[1] == 20, f"Expected window_size=20, got {features.shape[1]}"
        assert edge_idx.shape[0] == 2, f"Edge index should have shape [2, num_edges]"
        assert edge_idx.shape[1] == 12, f"Expected 12 edges, got {edge_idx.shape[1]}"
        assert target.shape[0] == 7, f"Expected 7 target values, got {target.shape[0]}"
        
        print("✓ Sample shapes are correct")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Sample shape check failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 4: Check for NaN/Inf values
    print("Test 4: Checking for NaN/Inf values...")
    try:
        has_nan_features = torch.isnan(features).any().item()
        has_inf_features = torch.isinf(features).any().item()
        has_nan_target = torch.isnan(target).any().item()
        has_inf_target = torch.isinf(target).any().item()
        
        print(f"  Features NaN: {has_nan_features}")
        print(f"  Features Inf: {has_inf_features}")
        print(f"  Target NaN: {has_nan_target}")
        print(f"  Target Inf: {has_inf_target}")
        
        assert not has_nan_features, "Features contain NaN values"
        assert not has_inf_features, "Features contain Inf values"
        assert not has_nan_target, "Target contains NaN values"
        assert not has_inf_target, "Target contains Inf values"
        
        print("✓ No NaN/Inf values found")
        test_results.append(True)
    except Exception as e:
        print(f"✗ NaN/Inf check failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 5: Iterate through multiple samples
    print("Test 5: Testing iteration...")
    try:
        num_to_check = min(5, len(dataset))
        
        for i in range(num_to_check):
            f, e, t = dataset[i]
            assert f.shape[0] == 7, f"Sample {i}: expected 7 stocks, got {f.shape[0]}"
            assert e.shape == edge_idx.shape, f"Sample {i}: edge_index shape mismatch"
            assert t.shape[0] == 7, f"Sample {i}: expected 7 targets, got {t.shape[0]}"
        
        print(f"✓ Successfully iterated through {num_to_check} samples")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Iteration failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 6: Test with validation data (FILTERED)
    print("Test 6: Testing with validation data...")
    try:
        val_path = data_dir / "val_features_filtered.parquet"  # FIXED: Use filtered data
        
        val_dataset = TemporalGraphDataset(
            data_path=str(val_path),
            edge_index=edge_index,
            window_size=20,
        )
        
        print(f"  Validation samples: {len(val_dataset)}")
        
        val_features, _, val_target = val_dataset[0]
        print(f"  Val features shape: {val_features.shape}")
        print(f"  Val target shape: {val_target.shape}")
        
        assert len(val_dataset) > 0, "Validation dataset is empty"
        assert val_features.shape[0] == 7, f"Expected 7 stocks in val, got {val_features.shape[0]}"
        
        print("✓ Validation dataset loaded successfully")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Validation dataset failed: {e}")
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
        print("Dataset Summary:")
        print(f"  • 7 stocks: ALB, APTV, F, GM, MGA, SQM, TSLA")
        print(f"  • {len(dataset)} training samples")
        print(f"  • {len(val_dataset)} validation samples")
        print(f"  • 20-day temporal windows")
        print(f"  • 12 supply chain edges")
        print(f"  • 15 features per node")
        print()
        print("="*70)
        print("✓ PHASE 4 STEP 1 COMPLETE!")
        print("="*70)
        print()
        print("Ready to proceed to Phase 4 Step 2: DataLoader & Training Loop")
        print()
        return True
    else:
        print()
        print("✗ SOME TESTS FAILED")
        print("Please check errors above.")
        print()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

    
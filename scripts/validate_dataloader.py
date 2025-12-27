"""
Validation script for Temporal Graph DataLoader

Tests the DataLoader with real processed data.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from utils.temporal_dataset import TemporalGraphDataset
from utils.graph_dataloader import TemporalGraphDataLoader


def main():
    """Validate DataLoader with real data"""
    print("="*70)
    print("TEMPORAL GRAPH DATALOADER - VALIDATION TEST")
    print("="*70)
    print()
    
    # Load datasets
    data_dir = project_root / "data" / "processed"
    
    # Edge index for 7 stocks
    edge_index = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 4, 4, 4, 0], 
                               [6, 2, 3, 4, 1, 2, 3, 6, 6, 2, 3, 1]], 
                              dtype=torch.long)
    
    print("Creating datasets...")
    train_dataset = TemporalGraphDataset(
        data_path=str(data_dir / "train_features_filtered.parquet"),
        edge_index=edge_index,
        window_size=20
    )
    
    val_dataset = TemporalGraphDataset(
        data_path=str(data_dir / "val_features_filtered.parquet"),
        edge_index=edge_index,
        window_size=20
    )
    
    print(f"✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Val dataset: {len(val_dataset)} samples")
    print()
    
    test_results = []
    
    # Test 1: Create DataLoader
    print("Test 1: Creating DataLoader...")
    try:
        train_loader, val_loader, _ = TemporalGraphDataLoader.create_train_val_test_loaders(
            train_dataset,
            val_dataset,
            val_dataset,  # Use val as test for this validation
            batch_size=16,
            num_workers=0
        )
        
        print(f"  ✓ Train loader: {len(train_loader)} batches")
        print(f"  ✓ Val loader: {len(val_loader)} batches")
        test_results.append(True)
    except Exception as e:
        print(f"  ✗ DataLoader creation failed: {e}")
        test_results.append(False)
        return False
    
    print()
    
    # Test 2: Get single batch
    print("Test 2: Getting single batch...")
    try:
        features, edge_idx, targets = next(iter(train_loader))
        
        print(f"  Batch features shape: {features.shape}")
        print(f"  Edge index shape: {edge_idx.shape}")
        print(f"  Batch targets shape: {targets.shape}")
        
        # Expected: [batch, 7, 20, 15]
        assert features.shape[0] <= 16  # Batch size
        assert features.shape[1] == 7   # Num nodes
        assert features.shape[2] == 20  # Seq len
        assert features.shape[3] == 15  # Features
        assert targets.shape[0] <= 16
        assert targets.shape[1] == 7
        
        print("  ✓ Batch shapes correct")
        test_results.append(True)
    except Exception as e:
        print(f"  ✗ Batch test failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 3: Iterate through full epoch
    print("Test 3: Iterating through full epoch...")
    try:
        batch_count = 0
        sample_count = 0
        
        for batch_features, batch_edge_index, batch_targets in train_loader:
            batch_count += 1
            sample_count += batch_features.shape[0]
            
            # Verify shapes
            assert batch_features.shape[1:] == (7, 20, 15)
            assert batch_edge_index.shape == (2, 12)
            assert batch_targets.shape[1] == 7
        
        print(f"  ✓ Processed {batch_count} batches")
        print(f"  ✓ Total samples: {sample_count}")
        
        # Should be close to dataset length (minus dropped last batch)
        assert sample_count >= len(train_dataset) - 16
        
        test_results.append(True)
    except Exception as e:
        print(f"  ✗ Epoch iteration failed: {e}")
        test_results.append(False)
    
    print()
    
    # Test 4: Validate no NaN/Inf in batches
    print("Test 4: Checking for NaN/Inf values...")
    try:
        has_nan = False
        has_inf = False
        
        for features, _, targets in train_loader:
            if torch.isnan(features).any() or torch.isnan(targets).any():
                has_nan = True
            if torch.isinf(features).any() or torch.isinf(targets).any():
                has_inf = True
        
        print(f"  Contains NaN: {has_nan}")
        print(f"  Contains Inf: {has_inf}")
        
        assert not has_nan, "Batches contain NaN values"
        assert not has_inf, "Batches contain Inf values"
        
        print("  ✓ No NaN/Inf values found")
        test_results.append(True)
    except Exception as e:
        print(f"  ✗ NaN/Inf check failed: {e}")
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
        print("✓ ALL DATALOADER TESTS PASSED!")
        print()
        print("DataLoader Summary:")
        print(f"  • Batch size: 16")
        print(f"  • Train batches: {len(train_loader)}")
        print(f"  • Val batches: {len(val_loader)}")
        print(f"  • Batch shape: [16, 7, 20, 15]")
        print(f"  • Custom collate: ✓")
        print()
        print("="*70)
        print("✓ PHASE 4 STEP 2 COMPLETE!")
        print("="*70)
        print()
        print("Ready for Phase 4 Step 3: Training Loop Implementation")
        print()
        return True
    else:
        print()
        print("✗ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
"""
Test feature normalization pipeline

This tests the complete feature engineering + normalization flow.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.feature_engineering import FeatureEngineer


def main():
    """Test normalization pipeline"""
    print("="*70)
    print("FEATURE NORMALIZATION - PIPELINE TEST")
    print("="*70)
    print()
    
    # Initialize engineer
    engineer = FeatureEngineer(
        market_data_dir="data/raw/market_data",
        macro_data_dir="data/raw/macro_data",
        output_dir="data/processed"
    )
    
    # Test with 3 stocks
    test_tickers = ["TSLA", "F", "GM"]
    
    print(f"Testing normalization pipeline for: {', '.join(test_tickers)}")
    print()
    
    # Run complete pipeline
    splits = engineer.process_and_normalize_all(test_tickers, save_splits=True)
    
    print()
    print("="*70)
    print("TEST RESULTS")
    print("="*70)
    
    if 'train' in splits and 'val' in splits and 'test' in splits:
        print("✓ ALL PIPELINE STEPS COMPLETED")
        
        # Show sample normalized features
        train = splits['train']
        norm_cols = [col for col in train.columns if col.endswith('_norm')]
        
        print(f"\n✓ Normalized {len(norm_cols)} features")
        print(f"✓ Train set: {len(train)} rows")
        print(f"✓ Val set: {len(splits['val'])} rows")
        print(f"✓ Test set: {len(splits['test'])} rows")
        
        # Verify normalization (mean ≈ 0, std ≈ 1)
        print(f"\n✓ Checking normalization quality (train set):")
        sample_feature = norm_cols[0]
        mean = train[sample_feature].mean()
        std = train[sample_feature].std()
        print(f"  {sample_feature}: mean={mean:.4f}, std={std:.4f}")
        
        if abs(mean) < 0.1 and abs(std - 1.0) < 0.1:
            print("  ✓ Normalization verified (mean≈0, std≈1)")
        
        print("\n✓ Feature engineering pipeline complete!")
        print("\nNext: Phase 2.2 - Graph Construction")
        return True
    else:
        print("✗ PIPELINE FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
"""
Quick validation script for Feature Engineering

This script performs a quick test to verify the FeatureEngineer is working.

Run this script to test before running full feature calculation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.feature_engineering import FeatureEngineer


def main():
    """Run quick validation test"""
    print("="*70)
    print("FEATURE ENGINEER - QUICK VALIDATION TEST")
    print("="*70)
    print()
    
    # Initialize engineer
    engineer = FeatureEngineer(
        market_data_dir="data/raw/market_data",
        macro_data_dir="data/raw/macro_data",
        output_dir="data/processed_test"
    )
    
    # Test with 3 stocks
    test_tickers = ["TSLA", "F", "GM"]
    
    print(f"Testing feature calculation for: {', '.join(test_tickers)}")
    print()
    
    # Process stocks
    results = engineer.process_all_stocks(test_tickers)
    
    print()
    print("="*70)
    print("TEST RESULTS")
    print("="*70)
    
    # Check results
    if len(results) == len(test_tickers):
        print("✓ ALL TESTS PASSED")
        print(f"✓ Successfully processed {len(results)} stocks")
        
        # Show sample of features
        sample_ticker = test_tickers[0]
        sample_df = results[sample_ticker]
        
        print(f"\n✓ Sample features for {sample_ticker}:")
        print(sample_df[['date', 'ticker', 'log_return', 'realized_vol', 'garch_vol', 'volume_shock', 'rsi']].head(10))
        
        # Validate one stock
        is_valid, issues = engineer.validate_features(sample_df, sample_ticker)
        if is_valid:
            print(f"\n✓ Feature validation passed for {sample_ticker}")
        else:
            print(f"\n⚠ Validation issues for {sample_ticker}: {issues}")
        
        # Save features
        success = engineer.save_features(results, "test_features.parquet")
        if success:
            print("\n✓ Features saved to Parquet successfully")
        
        print()
        print("Feature Engineer is working correctly!")
        return True
    else:
        print("✗ SOME TESTS FAILED")
        print(f"Expected {len(test_tickers)} stocks, got {len(results)}")
        if engineer.processing_stats['stocks_failed']:
            print(f"Failed stocks: {', '.join(engineer.processing_stats['stocks_failed'])}")
        print()
        print("Please check the logs for errors.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
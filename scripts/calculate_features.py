"""
Calculate features for ALL MVP stocks

This is the production version that processes all stocks in the supply chain.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.feature_engineering import FeatureEngineer


def main():
    """Calculate features for all MVP stocks"""
    print("="*70)
    print("FEATURE CALCULATION - ALL MVP STOCKS")
    print("="*70)
    print()
    
    # Initialize engineer
    engineer = FeatureEngineer(
        market_data_dir="data/raw/market_data",
        macro_data_dir="data/raw/macro_data",
        output_dir="data/processed"
    )
    
    # All MVP stocks from create_sample_relationships.py
    # Using stocks that we know have raw data based on diagnostic
    all_tickers = [
        # Tier 0: OEMs
        "TSLA", "F", "GM", "RIVN",
        
        # Tier 2: Component Suppliers
        "MGA", "APTV",
        
        # Tier 3: Raw Materials
        "ALB", "SQM",  # Both have full history
        # Note: LTHM not included - no raw data downloaded
    ]
    
    print(f"Processing {len(all_tickers)} stocks:")
    print(f"  {', '.join(all_tickers)}")
    print()
    print("This will take 5-10 minutes...")
    print()
    
    # Process all stocks
    results = engineer.process_all_stocks(all_tickers)
    
    print()
    print("="*70)
    print("FEATURE CALCULATION COMPLETE")
    print("="*70)
    
    # Check results
    successful = len(results)
    failed = len(all_tickers) - successful
    
    print(f"\nProcessed: {successful}/{len(all_tickers)} stocks")
    
    if engineer.processing_stats['stocks_failed']:
        print(f"Failed: {', '.join(engineer.processing_stats['stocks_failed'])}")
    
    if successful >= 6:  # Need at least 6 stocks for meaningful supply chain
        print(f"\n✓ Successfully processed {successful} stocks")
        
        # Save features
        print("\nSaving features to Parquet...")
        success = engineer.save_features(results, "all_features.parquet")
        
        if success:
            print("✓ Features saved to: data/processed/all_features.parquet")
            print()
            print("Next steps:")
            print("  1. Split into train/val/test: python scripts/split_temporal_data.py")
            print("  2. Normalize features: python scripts/normalize_features.py")
            return True
        else:
            print("✗ Failed to save features")
            return False
    else:
        print(f"\n✗ Too few stocks processed ({successful} < 6)")
        print("Need at least 6 stocks for supply chain structure")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
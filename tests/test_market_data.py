"""
Quick validation script for Market Data Collector

This script performs a quick test download of a few stocks to verify
the Market Data Collector is working correctly.

Run this script to test before running full data collection.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.market_data import MarketDataCollector


def main():
    """Run quick validation test"""
    print("="*70)
    print("MARKET DATA COLLECTOR - QUICK VALIDATION TEST")
    print("="*70)
    print()
    
    # Initialize collector with test output directory
    test_dir = "data/raw/market_data_test"
    collector = MarketDataCollector(output_dir=test_dir)
    
    # Test with 3 stocks, small date range (fast download)
    test_tickers = ["TSLA", "AAPL", "F"]
    start_date = "2023-01-01"
    end_date = "2023-03-31"
    
    print(f"Testing download for: {', '.join(test_tickers)}")
    print(f"Date range: {start_date} to {end_date}")
    print()
    
    # Download
    results = collector.download_all_stocks(
        test_tickers,
        start_date,
        end_date,
        validate=True
    )
    
    print()
    print("="*70)
    print("TEST RESULTS")
    print("="*70)
    
    # Check results
    if len(results) == len(test_tickers):
        print("✓ ALL TESTS PASSED")
        print(f"✓ Successfully downloaded {len(results)} stocks")
        print(f"✓ Data saved to: {test_dir}")
        print()
        print("Market Data Collector is working correctly!")
        return True
    else:
        print("✗ SOME TESTS FAILED")
        print(f"Expected {len(test_tickers)} stocks, got {len(results)}")
        if collector.download_stats['failed']:
            print(f"Failed tickers: {', '.join(collector.download_stats['failed'])}")
        print()
        print("Please check the logs for errors.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

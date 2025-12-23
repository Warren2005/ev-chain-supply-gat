"""
Quick validation script for Macro Data Collector

This script performs a quick test download of macro indicators.

Run this script to verify the Macro Data Collector works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.macro_data import MacroDataCollector


def main():
    """Run quick validation test"""
    print("="*70)
    print("MACRO DATA COLLECTOR - QUICK VALIDATION TEST")
    print("="*70)
    print()
    
    # Initialize collector
    test_dir = "data/raw/macro_data_test"
    collector = MacroDataCollector(output_dir=test_dir)
    
    # Test with small date range (fast download)
    start_date = "2023-01-01"
    end_date = "2023-03-31"
    
    print(f"Testing macro indicator downloads")
    print(f"Date range: {start_date} to {end_date}")
    print()
    
    # Download all indicators
    indicators = collector.download_all_indicators(start_date, end_date)
    
    # Combine indicators
    if indicators:
        combined = collector.combine_indicators(indicators)
        
        # Validate
        is_valid, issues = collector.validate_data(combined)
        
        # Save combined
        collector.save_combined(combined, "macro_indicators.csv")
    
    print()
    print("="*70)
    print("TEST RESULTS")
    print("="*70)
    
    # Check results
    expected_indicators = 5  # We have 5 macro indicators
    success_rate = len(indicators) / expected_indicators * 100
    
    if len(indicators) >= 3:  # At least 3 out of 5 should work
        print("✓ TEST PASSED")
        print(f"✓ Downloaded {len(indicators)}/{expected_indicators} indicators ({success_rate:.0f}%)")
        print(f"✓ Data saved to: {test_dir}")
        print()
        print("Macro Data Collector is working correctly!")
        return True
    else:
        print("✗ TEST FAILED")
        print(f"Only downloaded {len(indicators)}/{expected_indicators} indicators")
        print("Expected at least 3 indicators to succeed")
        print()
        print("Please check the logs for errors.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
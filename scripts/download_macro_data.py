"""
Download macro data for feature engineering

This downloads the macroeconomic indicators needed for Phase 2.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.macro_data import MacroDataCollector


def main():
    """Download macro indicators"""
    print("="*70)
    print("DOWNLOADING MACRO DATA")
    print("="*70)
    print()
    
    # Initialize collector
    collector = MacroDataCollector(output_dir="data/raw/macro_data")
    
    # Date range matching market data
    start_date = "2018-01-01"
    end_date = "2024-06-30"
    
    print(f"Date range: {start_date} to {end_date}")
    print()
    
    # Download all indicators
    indicators = collector.download_all_indicators(start_date, end_date)
    
    if len(indicators) >= 3:  # At least 3 out of 5 should work
        print("\n✓ Macro data download successful!")
        
        # Combine and save
        combined = collector.combine_indicators(indicators)
        collector.save_combined(combined, "macro_indicators.csv")
        
        print("✓ Combined macro data saved")
        print("\nNext step: Run feature engineering with macro features")
        print("  python scripts/test_feature_engineering.py")
        return True
    else:
        print("\n⚠ Some macro indicators failed to download")
        print("You may proceed but some features will be missing")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
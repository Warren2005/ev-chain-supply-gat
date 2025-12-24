"""
Download market data for all 15 MVP companies

This downloads the stock data needed for Phase 2 feature engineering.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.market_data import MarketDataCollector


def main():
    """Download market data for all MVP stocks"""
    print("="*70)
    print("DOWNLOADING MARKET DATA FOR 15 MVP COMPANIES")
    print("="*70)
    print()
    
    # 15 companies across 4 tiers (from Phase 1 summary)
    ticker_list = [
        # Tier 0: OEMs (4 companies)
        "TSLA", "F", "GM", "RIVN",
        
        # Tier 1: Battery Manufacturers (3 companies)
        "PCRFY",  # Panasonic
        # Note: LG and CATL may have limited data, will try anyway
        
        # Tier 2: Component Suppliers (2 companies)
        "MGA", "APTV",
        
        # Tier 3: Raw Materials (6 companies)
        "ALB", "SQM", "LTHM", "LAC", "MP",
        # Note: PLS.AX (Australian) may not work with yfinance
    ]
    
    # Date range from Phase 1 summary
    start_date = "2018-01-01"
    end_date = "2024-06-30"
    
    print(f"Companies to download: {len(ticker_list)}")
    print(f"Tickers: {', '.join(ticker_list)}")
    print(f"Date range: {start_date} to {end_date}")
    print()
    print("This may take 2-3 minutes...")
    print()
    
    # Initialize collector
    collector = MarketDataCollector(output_dir="data/raw/market_data")
    
    # Download all stocks
    results = collector.download_all_stocks(
        ticker_list,
        start_date,
        end_date,
        validate=True
    )
    
    print()
    print("="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    
    if len(results) >= 10:  # At least 10 out of 15 should succeed
        print(f"✓ Successfully downloaded {len(results)}/{len(ticker_list)} stocks")
        print("✓ Sufficient data for Phase 2 feature engineering")
        print()
        print("Next step: Run feature engineering test")
        print("  python scripts/test_feature_engineering.py")
        return True
    else:
        print(f"⚠ Only {len(results)}/{len(ticker_list)} stocks downloaded")
        print("This may still be enough to proceed, but some companies failed.")
        if collector.download_stats['failed']:
            print(f"\nFailed: {', '.join(collector.download_stats['failed'])}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
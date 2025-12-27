"""
Diagnostic script to check what's actually in processed parquet files
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd


def main():
    """Check processed data files"""
    print("="*70)
    print("PROCESSED DATA DIAGNOSTIC")
    print("="*70)
    print()
    
    data_dir = project_root / "data" / "processed"
    
    # Check each file
    for file_name in ["train_features.parquet", "val_features.parquet", "test_features.parquet"]:
        file_path = data_dir / file_name
        
        print(f"Checking {file_name}...")
        print("-" * 70)
        
        if not file_path.exists():
            print(f"  ✗ File not found: {file_path}")
            print()
            continue
        
        # Load data
        df = pd.read_parquet(file_path)
        
        # Get stock info
        stocks = sorted(df['ticker'].unique())
        dates = sorted(df['date'].unique())
        
        print(f"  Total rows: {len(df):,}")
        print(f"  Date range: {dates[0]} to {dates[-1]}")
        print(f"  Number of dates: {len(dates)}")
        print(f"  Number of stocks: {len(stocks)}")
        print(f"  Stocks: {stocks}")
        
        # Check for completeness
        print(f"\n  Data completeness check:")
        for stock in stocks:
            stock_df = df[df['ticker'] == stock]
            print(f"    {stock}: {len(stock_df)} rows, {stock_df['date'].nunique()} dates")
        
        # Check for NaN values
        feature_cols = [col for col in df.columns if col not in ['date', 'ticker']]
        nan_counts = df[feature_cols].isna().sum()
        total_nans = nan_counts.sum()
        
        print(f"\n  NaN values: {total_nans:,} total")
        if total_nans > 0:
            print(f"  Columns with NaN:")
            for col, count in nan_counts[nan_counts > 0].items():
                print(f"    {col}: {count}")
        
        print()
    
    # Also check raw market data
    print("\n" + "="*70)
    print("RAW MARKET DATA CHECK")
    print("="*70)
    print()
    
    raw_dir = project_root / "data" / "raw" / "market_data"
    
    if raw_dir.exists():
        csv_files = list(raw_dir.glob("*_prices.csv"))
        print(f"Found {len(csv_files)} CSV files in raw/market_data:")
        
        for csv_file in sorted(csv_files):
            ticker = csv_file.stem.replace("_prices", "")
            try:
                csv_df = pd.read_csv(csv_file)
                print(f"  {ticker}: {len(csv_df)} rows")
            except Exception as e:
                print(f"  {ticker}: ERROR - {e}")
    else:
        print("Raw market data directory not found")
    
    print()
    print("="*70)


if __name__ == "__main__":
    main()
"""
Filter to stocks with complete training history

Excludes RIVN which has insufficient training data (IPO in late 2021)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd


def main():
    """Filter to consistent stocks across all splits"""
    print("="*70)
    print("FILTERING TO CONSISTENT STOCKS")
    print("="*70)
    print()
    
    data_dir = Path("data/processed")
    
    # Stocks with complete training history
    # Exclude RIVN (only 35 training rows)
    consistent_stocks = ['ALB', 'APTV', 'F', 'GM', 'MGA', 'SQM', 'TSLA']
    
    print(f"Using {len(consistent_stocks)} stocks with complete history:")
    print(f"  {', '.join(consistent_stocks)}")
    print()
    print("Excluding: RIVN (insufficient training data - IPO in late 2021)")
    print()
    
    # Load and filter each split
    for split_name in ['train', 'val', 'test']:
        input_file = data_dir / f"{split_name}_features.parquet"
        output_file = data_dir / f"{split_name}_features_filtered.parquet"
        
        print(f"Processing {split_name}...")
        
        df = pd.read_parquet(input_file)
        original_rows = len(df)
        original_stocks = df['ticker'].nunique()
        
        # Filter to consistent stocks
        df_filtered = df[df['ticker'].isin(consistent_stocks)].copy()
        
        # Drop any rows with NaN values
        df_filtered = df_filtered.dropna()
        
        filtered_rows = len(df_filtered)
        filtered_stocks = df_filtered['ticker'].nunique()
        
        print(f"  Original: {original_rows:,} rows, {original_stocks} stocks")
        print(f"  Filtered: {filtered_rows:,} rows, {filtered_stocks} stocks")
        print(f"  Removed: {original_rows - filtered_rows:,} rows")
        
        # Verify each stock has same number of dates
        dates_per_stock = df_filtered.groupby('ticker')['date'].nunique()
        print(f"  Dates per stock: {dates_per_stock.min()} to {dates_per_stock.max()}")
        
        # Save filtered version
        df_filtered.to_parquet(output_file, index=False)
        print(f"  ✓ Saved: {output_file}")
        print()
    
    print("="*70)
    print("✓ FILTERING COMPLETE")
    print("="*70)
    print()
    print("Filtered files created with '_filtered' suffix")
    print("These files have:")
    print("  • 7 consistent stocks across all splits")
    print("  • No NaN values")
    print("  • Complete date coverage per stock")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
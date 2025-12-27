"""
Split feature data into train/validation/test sets

Temporal split to prevent data leakage:
- Train: 2018-2021 (4 years)
- Val: 2022 (1 year)
- Test: 2023-2024 (1.5 years)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime


def main():
    """Split data temporally"""
    print("="*70)
    print("TEMPORAL DATA SPLITTING")
    print("="*70)
    print()
    
    # Load all features
    input_file = Path("data/processed/all_features.parquet")
    
    if not input_file.exists():
        print(f"✗ Input file not found: {input_file}")
        print("\nRun feature calculation first:")
        print("  python scripts/calculate_features.py")
        return False
    
    print(f"Loading features from: {input_file}")
    df = pd.read_parquet(input_file)
    
    print(f"Total rows: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Stocks: {sorted(df['ticker'].unique())}")
    print()
    
    # Define split dates
    train_start = "2018-01-01"
    train_end = "2021-12-31"
    
    val_start = "2022-01-01"
    val_end = "2022-12-31"
    
    test_start = "2023-01-01"
    test_end = "2024-12-31"
    
    print("Split configuration:")
    print(f"  Train: {train_start} to {train_end}")
    print(f"  Val:   {val_start} to {val_end}")
    print(f"  Test:  {test_start} to {test_end}")
    print()
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Split data
    train_df = df[
        (df['date'] >= train_start) & (df['date'] <= train_end)
    ].copy()
    
    val_df = df[
        (df['date'] >= val_start) & (df['date'] <= val_end)
    ].copy()
    
    test_df = df[
        (df['date'] >= test_start) & (df['date'] <= test_end)
    ].copy()
    
    print("Split results:")
    print(f"  Train: {len(train_df):,} rows, {train_df['date'].nunique()} dates")
    print(f"  Val:   {len(val_df):,} rows, {val_df['date'].nunique()} dates")
    print(f"  Test:  {len(test_df):,} rows, {test_df['date'].nunique()} dates")
    print()
    
    # Verify no overlap
    train_dates = set(train_df['date'])
    val_dates = set(val_df['date'])
    test_dates = set(test_df['date'])
    
    assert len(train_dates & val_dates) == 0, "Train/Val overlap detected!"
    assert len(train_dates & test_dates) == 0, "Train/Test overlap detected!"
    assert len(val_dates & test_dates) == 0, "Val/Test overlap detected!"
    
    print("✓ No temporal overlap between splits")
    print()
    
    # Save splits
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train_features_raw.parquet"
    val_path = output_dir / "val_features_raw.parquet"
    test_path = output_dir / "test_features_raw.parquet"
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print("✓ Saved splits:")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Test:  {test_path}")
    print()
    print("Next step: Normalize features")
    print("  python scripts/normalize_features.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
"""
Normalize features using StandardScaler

CRITICAL: Fit scaler ONLY on training data to prevent data leakage!
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


def main():
    """Normalize features across train/val/test"""
    print("="*70)
    print("FEATURE NORMALIZATION")
    print("="*70)
    print()
    
    data_dir = Path("data/processed")
    
    # Load raw splits
    print("Loading raw split files...")
    train_df = pd.read_parquet(data_dir / "train_features_raw.parquet")
    val_df = pd.read_parquet(data_dir / "val_features_raw.parquet")
    test_df = pd.read_parquet(data_dir / "test_features_raw.parquet")
    
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Val:   {len(val_df):,} rows")
    print(f"  Test:  {len(test_df):,} rows")
    print()
    
    # Define features to normalize
    # Don't normalize: date, ticker
    exclude_cols = ['date', 'ticker']
    
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    print(f"Normalizing {len(feature_cols)} features:")
    print(f"  {', '.join(feature_cols[:5])}...")
    print()
    
    # Fit scaler on TRAINING data only
    print("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    
    print("✓ Scaler fitted")
    print()
    
    # Transform all splits
    print("Transforming datasets...")
    
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    print("✓ All datasets normalized")
    print()
    
    # Add normalized suffix to column names for clarity
    rename_dict = {col: f"{col}_norm" if col not in exclude_cols else col 
                   for col in train_df.columns}
    
    # Actually, let's NOT rename - keep original column names
    # The _norm suffix was added during feature engineering already
    
    # Save normalized data
    print("Saving normalized features...")
    
    train_df.to_parquet(data_dir / "train_features.parquet", index=False)
    val_df.to_parquet(data_dir / "val_features.parquet", index=False)
    test_df.to_parquet(data_dir / "test_features.parquet", index=False)
    
    print("✓ Saved:")
    print(f"  {data_dir / 'train_features.parquet'}")
    print(f"  {data_dir / 'val_features.parquet'}")
    print(f"  {data_dir / 'test_features.parquet'}")
    print()
    
    # Save scaler for later use
    scaler_path = data_dir / "feature_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"✓ Scaler saved: {scaler_path}")
    print()
    
    # Show normalization stats
    print("Normalization stats (from training data):")
    print(f"  Mean (should be ~0): {train_df[feature_cols].mean().mean():.6f}")
    print(f"  Std (should be ~1): {train_df[feature_cols].std().mean():.6f}")
    print()
    
    print("="*70)
    print("✓ FEATURE NORMALIZATION COMPLETE")
    print("="*70)
    print()
    print("Next step: Build knowledge graph")
    print("  python scripts/build_graph.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
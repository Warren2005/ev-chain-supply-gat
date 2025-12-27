"""
Fix data preprocessing to handle extreme outliers

This script:
1. Clips extreme outliers (>3 std)
2. Applies log-transform to garch_vol
3. Renormalizes with robust scaling
4. Saves fixed datasets for retraining

Author: EV Supply Chain GAT Team
Date: December 2024
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns


def clip_outliers(df: pd.DataFrame, column: str, n_std: float = 3.0) -> pd.DataFrame:
    """
    Clip outliers beyond n standard deviations.
    
    Args:
        df: DataFrame
        column: Column to clip
        n_std: Number of standard deviations for clipping
    
    Returns:
        DataFrame with clipped values
    """
    mean = df[column].mean()
    std = df[column].std()
    
    lower_bound = mean - n_std * std
    upper_bound = mean + n_std * std
    
    original_min = df[column].min()
    original_max = df[column].max()
    
    df[column] = df[column].clip(lower_bound, upper_bound)
    
    clipped_count = ((df[column] == lower_bound) | (df[column] == upper_bound)).sum()
    
    print(f"  {column}:")
    print(f"    Original range: [{original_min:.4f}, {original_max:.4f}]")
    print(f"    Clipped range: [{lower_bound:.4f}, {upper_bound:.4f}]")
    print(f"    Clipped values: {clipped_count} ({clipped_count/len(df)*100:.2f}%)")
    
    return df


def apply_log_transform(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Apply log(1+x) transform to make distribution more normal.
    
    This handles negative values by using sign(x) * log(1 + |x|)
    
    Args:
        df: DataFrame
        column: Column to transform
    
    Returns:
        DataFrame with transformed column
    """
    values = df[column].values
    
    # Handle negative values: sign(x) * log(1 + |x|)
    transformed = np.sign(values) * np.log1p(np.abs(values))
    
    print(f"  {column} log-transform:")
    print(f"    Original: mean={values.mean():.4f}, std={values.std():.4f}")
    print(f"    Transformed: mean={transformed.mean():.4f}, std={transformed.std():.4f}")
    print(f"    Skewness reduction: {np.abs(pd.Series(values).skew()):.2f} → {np.abs(pd.Series(transformed).skew()):.2f}")
    
    df[column] = transformed
    
    return df


def normalize_with_robust_scaler(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list
) -> tuple:
    """
    Normalize using RobustScaler (median and IQR instead of mean and std).
    
    RobustScaler is less sensitive to outliers than StandardScaler.
    
    Args:
        train_df, val_df, test_df: DataFrames to normalize
        feature_columns: Columns to normalize
    
    Returns:
        Tuple of (train_df, val_df, test_df, scaler)
    """
    scaler = RobustScaler()
    
    # Fit on training data only
    scaler.fit(train_df[feature_columns])
    
    # Transform all splits
    train_df[feature_columns] = scaler.transform(train_df[feature_columns])
    val_df[feature_columns] = scaler.transform(val_df[feature_columns])
    test_df[feature_columns] = scaler.transform(test_df[feature_columns])
    
    print(f"  Applied RobustScaler to {len(feature_columns)} features")
    
    return train_df, val_df, test_df, scaler


def main():
    """Fix data preprocessing pipeline"""
    print("="*70)
    print("FIXING DATA PREPROCESSING")
    print("="*70)
    print()
    
    data_dir = Path("data/processed")
    
    # Load raw normalized data (before the issues)
    print("Loading processed data...")
    train_df = pd.read_parquet(data_dir / "train_features_filtered.parquet")
    val_df = pd.read_parquet(data_dir / "val_features_filtered.parquet")
    test_df = pd.read_parquet(data_dir / "test_features_filtered.parquet")
    
    print(f"✓ Train: {len(train_df)} rows")
    print(f"✓ Val: {len(val_df)} rows")
    print(f"✓ Test: {len(test_df)} rows")
    print()
    
    # Feature columns (excluding date and ticker)
    feature_columns = [col for col in train_df.columns if col not in ['date', 'ticker']]
    
    print(f"Feature columns: {len(feature_columns)}")
    print()
    
    # Step 1: Clip extreme outliers in garch_vol (target)
    print("Step 1: Clipping extreme outliers in garch_vol...")
    train_df = clip_outliers(train_df, 'garch_vol', n_std=3.0)
    val_df = clip_outliers(val_df, 'garch_vol', n_std=3.0)
    test_df = clip_outliers(test_df, 'garch_vol', n_std=3.0)
    print()
    
    # Step 2: Apply log-transform to garch_vol
    print("Step 2: Applying log-transform to garch_vol...")
    train_df = apply_log_transform(train_df, 'garch_vol')
    val_df = apply_log_transform(val_df, 'garch_vol')
    test_df = apply_log_transform(test_df, 'garch_vol')
    print()
    
    # Step 3: Clip outliers in other volatile features
    print("Step 3: Clipping outliers in other features...")
    volatile_features = ['realized_vol', 'volume_shock', 'rsi']
    
    for feat in volatile_features:
        if feat in train_df.columns:
            train_df = clip_outliers(train_df, feat, n_std=3.0)
            val_df = clip_outliers(val_df, feat, n_std=3.0)
            test_df = clip_outliers(test_df, feat, n_std=3.0)
    print()
    
    # Step 4: Renormalize with RobustScaler
    print("Step 4: Renormalizing with RobustScaler...")
    train_df, val_df, test_df, scaler = normalize_with_robust_scaler(
        train_df, val_df, test_df, feature_columns
    )
    print()
    
    # Visualize improvement
    print("Step 5: Generating distribution comparison plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, (name, df) in enumerate([("Train", train_df), ("Val", val_df), ("Test", test_df)]):
        # Histogram
        ax = axes[0, idx]
        ax.hist(df['garch_vol'], bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'{name} Set - Fixed Distribution')
        ax.set_xlabel('garch_vol (log-transformed + robust-scaled)')
        ax.set_ylabel('Frequency')
        ax.axvline(0, color='r', linestyle='--', label='Zero')
        ax.legend()
        
        # Q-Q plot
        from scipy import stats
        ax = axes[1, idx]
        stats.probplot(df['garch_vol'], dist="norm", plot=ax)
        ax.set_title(f'{name} Set Q-Q Plot')
    
    plt.tight_layout()
    plt.savefig('results/diagnosis/fixed_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Distribution comparison saved")
    print()
    
    # Save fixed datasets
    print("Step 6: Saving fixed datasets...")
    train_df.to_parquet(data_dir / "train_features_fixed.parquet", index=False)
    val_df.to_parquet(data_dir / "val_features_fixed.parquet", index=False)
    test_df.to_parquet(data_dir / "test_features_fixed.parquet", index=False)
    
    print(f"✓ Saved: train_features_fixed.parquet")
    print(f"✓ Saved: val_features_fixed.parquet")
    print(f"✓ Saved: test_features_fixed.parquet")
    print()
    
    # Save scaler
    import pickle
    with open(data_dir / "robust_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved: robust_scaler.pkl")
    print()
    
    # Print summary statistics
    print("="*70)
    print("FIXED DATA SUMMARY")
    print("="*70)
    print()
    print("garch_vol statistics (after fixes):")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        vals = df['garch_vol'].values
        print(f"\n{name}:")
        print(f"  Mean: {vals.mean():.6f}")
        print(f"  Std: {vals.std():.6f}")
        print(f"  Min: {vals.min():.6f}")
        print(f"  Max: {vals.max():.6f}")
        print(f"  Skewness: {pd.Series(vals).skew():.4f}")
        print(f"  Kurtosis: {pd.Series(vals).kurtosis():.4f}")
    
    print()
    print("="*70)
    print("✓ DATA PREPROCESSING FIXED!")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Retrain model with fixed data")
    print("  2. Use learning rate scheduler")
    print("  3. Lower initial learning rate to 1e-4")
    print()
    print("Run: python scripts/retrain_model.py")
    print()


if __name__ == "__main__":
    main()
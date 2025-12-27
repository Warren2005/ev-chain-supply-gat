"""
Diagnostic script to understand model collapse and training issues

This will help identify what went wrong with training.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from models.st_gat import STGAT
from utils.temporal_dataset import TemporalGraphDataset
from utils.graph_dataloader import TemporalGraphDataLoader


def analyze_target_distribution():
    """Analyze the distribution of target values"""
    print("="*70)
    print("ANALYZING TARGET FEATURE DISTRIBUTION")
    print("="*70)
    print()
    
    data_dir = Path("data/processed")
    
    # Load all splits
    train_df = pd.read_parquet(data_dir / "train_features_filtered.parquet")
    val_df = pd.read_parquet(data_dir / "val_features_filtered.parquet")
    test_df = pd.read_parquet(data_dir / "test_features_filtered.parquet")
    
    print("Target Feature Statistics (garch_vol):")
    print("-" * 70)
    
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        garch_vals = df['garch_vol'].values
        
        print(f"\n{name} Set:")
        print(f"  Mean: {garch_vals.mean():.6f}")
        print(f"  Std: {garch_vals.std():.6f}")
        print(f"  Min: {garch_vals.min():.6f}")
        print(f"  Max: {garch_vals.max():.6f}")
        print(f"  Median: {np.median(garch_vals):.6f}")
        print(f"  25th percentile: {np.percentile(garch_vals, 25):.6f}")
        print(f"  75th percentile: {np.percentile(garch_vals, 75):.6f}")
        print(f"  Skewness: {stats.skew(garch_vals):.6f}")
        print(f"  Kurtosis: {stats.kurtosis(garch_vals):.6f}")
        
        # Check for outliers
        q1 = np.percentile(garch_vals, 25)
        q3 = np.percentile(garch_vals, 75)
        iqr = q3 - q1
        outliers = np.sum((garch_vals < q1 - 1.5*iqr) | (garch_vals > q3 + 1.5*iqr))
        print(f"  Outliers (IQR method): {outliers} ({outliers/len(garch_vals)*100:.2f}%)")
    
    # Plot distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, (name, df) in enumerate([("Train", train_df), ("Val", val_df), ("Test", test_df)]):
        # Histogram
        ax = axes[0, idx]
        ax.hist(df['garch_vol'], bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'{name} Set Distribution')
        ax.set_xlabel('garch_vol (normalized)')
        ax.set_ylabel('Frequency')
        ax.axvline(0, color='r', linestyle='--', label='Zero')
        ax.legend()
        
        # Q-Q plot
        ax = axes[1, idx]
        stats.probplot(df['garch_vol'], dist="norm", plot=ax)
        ax.set_title(f'{name} Set Q-Q Plot')
    
    plt.tight_layout()
    plt.savefig('results/diagnosis/target_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Distribution plot saved to results/diagnosis/target_distribution.png")
    
    return train_df, val_df, test_df


def analyze_model_predictions():
    """Analyze what the model is actually predicting"""
    print("\n" + "="*70)
    print("ANALYZING MODEL PREDICTIONS")
    print("="*70)
    print()
    
    # Load predictions
    pred_df = pd.read_csv('results/evaluation/predictions_detailed.csv')
    
    stocks = ['ALB', 'APTV', 'F', 'GM', 'MGA', 'SQM', 'TSLA']
    
    print("Prediction Statistics by Stock:")
    print("-" * 70)
    
    for stock in stocks:
        pred_col = f'pred_{stock}'
        actual_col = f'actual_{stock}'
        
        preds = pred_df[pred_col].values
        actuals = pred_df[actual_col].values
        
        print(f"\n{stock}:")
        print(f"  Pred Mean: {preds.mean():.6f}, Std: {preds.std():.6f}")
        print(f"  Pred Range: [{preds.min():.6f}, {preds.max():.6f}]")
        print(f"  Actual Mean: {actuals.mean():.6f}, Std: {actuals.std():.6f}")
        print(f"  Actual Range: [{actuals.min():.6f}, {actuals.max():.6f}]")
        print(f"  Variance Ratio (pred/actual): {(preds.std()/actuals.std()):.6f}")
        
        # Check if predictions are constant
        unique_preds = len(np.unique(np.round(preds, 4)))
        print(f"  Unique prediction values (rounded): {unique_preds}")
        
        if unique_preds < 10:
            print(f"  ⚠️  WARNING: Model predicting near-constant values!")
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, stock in enumerate(stocks):
        ax = axes[idx]
        
        preds = pred_df[f'pred_{stock}'].values
        actuals = pred_df[f'actual_{stock}'].values
        
        ax.scatter(actuals, preds, alpha=0.3, s=10)
        
        # Perfect prediction line
        min_val = min(actuals.min(), preds.min())
        max_val = max(actuals.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{stock}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/diagnosis/prediction_scatter.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Prediction scatter saved to results/diagnosis/prediction_scatter.png")


def analyze_gradient_flow():
    """Check if gradients are flowing properly"""
    print("\n" + "="*70)
    print("ANALYZING GRADIENT FLOW")
    print("="*70)
    print()
    
    # Load training history
    import json
    with open('results/training/training_report.json', 'r') as f:
        report = json.load(f)
    
    train_losses = report['history']['train_losses']
    val_losses = report['history']['val_losses']
    
    print("Training Dynamics:")
    print("-" * 70)
    print(f"Initial train loss: {train_losses[0]:.6f}")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Train loss reduction: {(train_losses[0] - train_losses[-1]):.6f}")
    print(f"Train loss reduction %: {(train_losses[0] - train_losses[-1])/train_losses[0]*100:.2f}%")
    print()
    print(f"Initial val loss: {val_losses[0]:.6f}")
    print(f"Best val loss: {min(val_losses):.6f}")
    print(f"Val loss reduction: {(val_losses[0] - min(val_losses)):.6f}")
    print(f"Val loss reduction %: {(val_losses[0] - min(val_losses))/val_losses[0]*100:.2f}%")
    
    # Check if loss plateaued
    last_10_train = train_losses[-10:]
    train_std = np.std(last_10_train)
    print(f"\nLast 10 epochs train loss std: {train_std:.6f}")
    
    if train_std < 0.001:
        print("⚠️  WARNING: Training loss has plateaued!")
    
    # Plot loss evolution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax = axes[0]
    ax.plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss change per epoch
    ax = axes[1]
    train_changes = np.diff(train_losses)
    val_changes = np.diff(val_losses)
    ax.plot(epochs[1:], train_changes, 'b-', label='Train Change', alpha=0.7)
    ax.plot(epochs[1:], val_changes, 'r-', label='Val Change', alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Change')
    ax.set_title('Loss Gradient (per epoch)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/diagnosis/training_dynamics.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Training dynamics saved to results/diagnosis/training_dynamics.png")


def main():
    """Run all diagnostic analyses"""
    # Create diagnosis directory
    Path("results/diagnosis").mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("ST-GAT MODEL DIAGNOSTICS")
    print("="*70)
    print()
    
    # Run analyses
    analyze_target_distribution()
    analyze_model_predictions()
    analyze_gradient_flow()
    
    print("\n" + "="*70)
    print("✓ DIAGNOSTICS COMPLETE")
    print("="*70)
    print("\nGenerated diagnostic plots:")
    print("  • results/diagnosis/target_distribution.png")
    print("  • results/diagnosis/prediction_scatter.png")
    print("  • results/diagnosis/training_dynamics.png")
    print()
    print("KEY FINDINGS:")
    print("  1. Check if target values have extreme outliers")
    print("  2. Model may be predicting constant values (mode collapse)")
    print("  3. Training loss barely decreased - model didn't learn")
    print()
    print("RECOMMENDED FIXES:")
    print("  1. Use log-transform on garch_vol before normalization")
    print("  2. Clip extreme outliers (>3 std)")
    print("  3. Try lower learning rate (1e-4 instead of 1e-3)")
    print("  4. Add learning rate scheduler")
    print("  5. Increase gradient clipping (try 5.0)")
    print()


if __name__ == "__main__":
    main()
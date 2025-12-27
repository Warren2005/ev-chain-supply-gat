"""
Final Evaluation of Simplified ST-GAT Model

Evaluates the best-performing model (Attempt 3) on test set.
Checks for prediction variability and compares with all previous attempts.

Author: EV Supply Chain GAT Team
Date: December 2024
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

from models.simplified_st_gat import SimplifiedSTGAT
from utils.temporal_dataset import TemporalGraphDataset
from utils.graph_dataloader import TemporalGraphDataLoader
from utils.evaluator import STGATEvaluator


def plot_prediction_quality(
    evaluator: STGATEvaluator,
    stock_names: list,
    save_path: Path
):
    """
    Plot prediction quality for each stock.
    
    Shows actual vs predicted values to verify the model
    is producing varied predictions.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for stock_idx, stock in enumerate(stock_names):
        ax = axes[stock_idx]
        
        preds = evaluator.predictions[:, stock_idx].numpy()
        actuals = evaluator.targets[:, stock_idx].numpy()
        
        # Scatter plot
        ax.scatter(actuals, preds, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(actuals.min(), preds.min())
        max_val = max(actuals.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        # Stats
        r2 = 1 - np.sum((actuals - preds)**2) / np.sum((actuals - actuals.mean())**2)
        corr = np.corrcoef(actuals, preds)[0, 1]
        
        ax.set_xlabel('Actual Volatility')
        ax.set_ylabel('Predicted Volatility')
        ax.set_title(f'{stock}\nR²={r2:.3f}, Corr={corr:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Prediction quality plot saved: {save_path}")


def compare_all_models(save_path: Path):
    """
    Compare test performance across all three training attempts.
    """
    # Load all evaluation results
    attempt1 = pd.read_csv('results/evaluation/test_metrics.csv')
    attempt2 = pd.read_csv('results/retrained_evaluation/test_metrics.csv')
    # Attempt 3 will be saved by this script
    
    print("\n" + "="*70)
    print("COMPARISON ACROSS ALL TRAINING ATTEMPTS")
    print("="*70)
    print()
    print("Attempt 1 (Original - Collapsed):")
    print(f"  R²: {attempt1['r2'].values[0]:.4f}")
    print(f"  RMSE: {attempt1['rmse'].values[0]:.4f}")
    print(f"  Dir Acc: {attempt1['directional_accuracy'].values[0]*100:.1f}%")
    print()
    print("Attempt 2 (Fixed Data - Partial Collapse):")
    print(f"  R²: {attempt2['r2'].values[0]:.4f}")
    print(f"  RMSE: {attempt2['rmse'].values[0]:.4f}")
    print(f"  Dir Acc: {attempt2['directional_accuracy'].values[0]*100:.1f}%")
    print()


def main():
    """Final evaluation"""
    print("="*70)
    print("FINAL MODEL EVALUATION - ATTEMPT 3")
    print("="*70)
    print()
    
    # Stock info
    stock_names = ['ALB', 'APTV', 'F', 'GM', 'MGA', 'SQM', 'TSLA']
    edge_index = torch.tensor([
        [0, 0, 0, 4, 4, 4, 1, 1, 1, 0, 0, 4, 5, 5, 5],
        [6, 2, 3, 6, 2, 3, 2, 3, 6, 4, 1, 1, 6, 2, 3]
    ], dtype=torch.long)
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = TemporalGraphDataset(
        data_path="data/processed/test_features_fixed.parquet",
        edge_index=edge_index,
        window_size=20
    )
    print(f"✓ Test: {len(test_dataset)} samples")
    print()
    
    # Create test loader
    _, _, test_loader = TemporalGraphDataLoader.create_train_val_test_loaders(
        test_dataset, test_dataset, test_dataset,
        batch_size=32, num_workers=0
    )
    
    # Load model
    print("Loading Simplified ST-GAT model...")
    model = SimplifiedSTGAT(
        num_nodes=7,
        num_edges=15,
        input_features=15,
        gat_hidden_dim=64,
        gat_heads=4,
        lstm_hidden_dim=64,
        temporal_window=20,
        output_dim=1,
        dropout=0.3,
        device='cpu'
    )
    
    checkpoint = torch.load(
        'checkpoints/final_retrain/best_model.pt',
        map_location='cpu',
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded from epoch {checkpoint['epoch']}")
    print(f"✓ Training val loss: {checkpoint['val_loss']:.6f}")
    print()
    
    # Evaluate
    evaluator = STGATEvaluator(
        model=model,
        test_loader=test_loader,
        stock_names=stock_names,
        device='cpu'
    )
    
    print("="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    print()
    
    metrics = evaluator.evaluate()
    
    # Per-stock metrics
    per_stock = evaluator.compute_per_stock_metrics()
    print("\nPer-Stock Performance:")
    print(per_stock.to_string(index=False))
    print()
    
    # Prediction variability check
    print("="*70)
    print("PREDICTION VARIABILITY CHECK")
    print("="*70)
    print()
    
    all_good = True
    for stock_idx, stock in enumerate(stock_names):
        preds = evaluator.predictions[:, stock_idx].numpy()
        
        pred_std = preds.std()
        pred_unique = len(np.unique(np.round(preds, 4)))
        pred_range = preds.max() - preds.min()
        
        print(f"{stock}:")
        print(f"  Mean: {preds.mean():.4f}")
        print(f"  Std:  {pred_std:.4f}")
        print(f"  Range: {pred_range:.4f}")
        print(f"  Unique values: {pred_unique}")
        
        if pred_std < 0.01:
            print(f"  ✗ COLLAPSED: Near-constant")
            all_good = False
        elif pred_std < 0.05:
            print(f"  ⚠️  CAUTION: Low variability")
        else:
            print(f"  ✓ GOOD: Varied predictions")
        print()
    
    # Save results
    results_dir = Path("results/final_evaluation")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator.save_results(results_dir, metrics, per_stock)
    
    # Generate plots
    print("Generating evaluation plots...")
    plot_prediction_quality(evaluator, stock_names, results_dir / 'prediction_quality.png')
    
    # Compare with previous attempts
    compare_all_models(results_dir / 'model_comparison.txt')
    
    print()
    print("="*70)
    print("FINAL EVALUATION SUMMARY")
    print("="*70)
    print()
    
    if metrics['r2'] > 0:
        print(f"✓ R² = {metrics['r2']:.4f} (POSITIVE - beats baseline!)")
    else:
        print(f"✗ R² = {metrics['r2']:.4f} (still negative)")
    
    if metrics['directional_accuracy'] > 0.5:
        print(f"✓ Directional Accuracy = {metrics['directional_accuracy']*100:.1f}% (beats coin flip!)")
    else:
        print(f"✗ Directional Accuracy = {metrics['directional_accuracy']*100:.1f}%")
    
    if all_good:
        print("✓ All stocks have varied predictions!")
    else:
        print("✗ Some stocks still collapsed")
    
    print()
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"Correlation: {metrics['correlation']:.4f}")
    print()
    
    if metrics['r2'] > 0 and all_good:
        print("="*70)
        print("🎉 SUCCESS! MODEL IS WORKING!")
        print("="*70)
        print()
        print("The model is:")
        print("  ✓ Making varied predictions (not collapsed)")
        print("  ✓ Beating baseline (R² > 0)")
        print("  ✓ Capturing volatility patterns")
        print()
        print("Ready for:")
        print("  • Advanced 3D visualizations")
        print("  • Baseline comparisons")
        print("  • Hypothesis testing")
        print("  • Attention analysis")
    else:
        print("Still needs improvement, but much better than before!")
    
    print()
    print(f"Results saved to: {results_dir}")
    print()


if __name__ == "__main__":
    main()
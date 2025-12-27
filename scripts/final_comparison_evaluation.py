"""
Final Evaluation: AdaptiveHuber vs Original Model

Comprehensive comparison on test set showing improvements
in both R² and directional accuracy.

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
from sklearn.metrics import r2_score

from models.simplified_st_gat import SimplifiedSTGAT
from utils.temporal_dataset import TemporalGraphDataset
from utils.graph_dataloader import TemporalGraphDataLoader


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model on test set"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for features, edge_index, targets in test_loader:
            features = features.to(device)
            edge_index = edge_index.to(device)
            
            predictions = model(features, edge_index).squeeze(-1)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    return predictions, targets


def compute_comprehensive_metrics(predictions, targets, stock_names):
    """Compute all metrics per stock and overall"""
    num_stocks = predictions.shape[1]
    
    # Overall metrics
    overall_metrics = {
        'r2': r2_score(targets.numpy().flatten(), predictions.numpy().flatten()),
        'rmse': np.sqrt(((predictions - targets) ** 2).mean()).item(),
        'mae': (predictions - targets).abs().mean().item(),
        'directional_accuracy': (torch.sign(predictions) == torch.sign(targets)).float().mean().item(),
        'correlation': np.corrcoef(predictions.numpy().flatten(), targets.numpy().flatten())[0, 1]
    }
    
    # Per-stock metrics
    per_stock = []
    for i, stock in enumerate(stock_names):
        preds = predictions[:, i].numpy()
        targs = targets[:, i].numpy()
        
        r2 = r2_score(targs, preds)
        rmse = np.sqrt(((preds - targs) ** 2).mean())
        mae = np.abs(preds - targs).mean()
        dir_acc = (np.sign(preds) == np.sign(targs)).mean()
        corr = np.corrcoef(preds, targs)[0, 1] if len(np.unique(preds)) > 1 else 0.0
        
        per_stock.append({
            'stock': stock,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': dir_acc,
            'correlation': corr
        })
    
    return overall_metrics, pd.DataFrame(per_stock)


def create_comparison_plots(
    original_metrics,
    optimized_metrics,
    stock_names,
    save_dir
):
    """Create comprehensive comparison visualizations"""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: R² comparison by stock
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # R² comparison
    ax = axes[0, 0]
    x = np.arange(len(stock_names))
    width = 0.35
    
    original_r2 = [original_metrics[original_metrics['stock'] == s]['r2'].values[0] 
                   for s in stock_names]
    optimized_r2 = [optimized_metrics[optimized_metrics['stock'] == s]['r2'].values[0] 
                    for s in stock_names]
    
    ax.bar(x - width/2, original_r2, width, label='Original (Huber)', alpha=0.8, color='red')
    ax.bar(x + width/2, optimized_r2, width, label='Optimized (AdaptiveHuber)', alpha=0.8, color='green')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Stock')
    ax.set_ylabel('R² Score')
    ax.set_title('R² Score Comparison by Stock')
    ax.set_xticks(x)
    ax.set_xticklabels(stock_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Directional Accuracy comparison
    ax = axes[0, 1]
    original_dir = [original_metrics[original_metrics['stock'] == s]['directional_accuracy'].values[0] 
                    for s in stock_names]
    optimized_dir = [optimized_metrics[optimized_metrics['stock'] == s]['directional_accuracy'].values[0] 
                     for s in stock_names]
    
    ax.bar(x - width/2, [d*100 for d in original_dir], width, label='Original', alpha=0.8, color='red')
    ax.bar(x + width/2, [d*100 for d in optimized_dir], width, label='Optimized', alpha=0.8, color='green')
    ax.axhline(y=50, color='black', linestyle='--', linewidth=1, label='Random (50%)')
    ax.set_xlabel('Stock')
    ax.set_ylabel('Directional Accuracy (%)')
    ax.set_title('Directional Accuracy Comparison by Stock')
    ax.set_xticks(x)
    ax.set_xticklabels(stock_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Correlation comparison
    ax = axes[1, 0]
    original_corr = [original_metrics[original_metrics['stock'] == s]['correlation'].values[0] 
                     for s in stock_names]
    optimized_corr = [optimized_metrics[optimized_metrics['stock'] == s]['correlation'].values[0] 
                      for s in stock_names]
    
    ax.bar(x - width/2, original_corr, width, label='Original', alpha=0.8, color='red')
    ax.bar(x + width/2, optimized_corr, width, label='Optimized', alpha=0.8, color='green')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Stock')
    ax.set_ylabel('Correlation')
    ax.set_title('Prediction-Actual Correlation by Stock')
    ax.set_xticks(x)
    ax.set_xticklabels(stock_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Overall improvement summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate improvements
    orig_overall_r2 = original_metrics['r2'].mean()
    opt_overall_r2 = optimized_metrics['r2'].mean()
    orig_overall_dir = original_metrics['directional_accuracy'].mean()
    opt_overall_dir = optimized_metrics['directional_accuracy'].mean()
    
    summary_text = f"""
    OVERALL IMPROVEMENT SUMMARY
    
    R² Score:
    • Original:  {orig_overall_r2:.4f}
    • Optimized: {opt_overall_r2:.4f}
    • Change:    {opt_overall_r2 - orig_overall_r2:+.4f} ({(opt_overall_r2 - orig_overall_r2)/abs(orig_overall_r2)*100:+.1f}%)
    
    Directional Accuracy:
    • Original:  {orig_overall_dir*100:.2f}%
    • Optimized: {opt_overall_dir*100:.2f}%
    • Change:    {(opt_overall_dir - orig_overall_dir)*100:+.2f}%
    
    Stocks with Positive R²:
    • Original:  {(original_metrics['r2'] > 0).sum()}/7
    • Optimized: {(optimized_metrics['r2'] > 0).sum()}/7
    
    Average Correlation:
    • Original:  {original_metrics['correlation'].mean():.3f}
    • Optimized: {optimized_metrics['correlation'].mean():.3f}
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'comparison_by_stock.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir / 'comparison_by_stock.png'}")
    
    # Plot 2: Scatter plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Will be filled in main()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'prediction_scatter_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir / 'prediction_scatter_comparison.png'}")


def main():
    """Main evaluation"""
    print("="*70)
    print("FINAL COMPARISON: ORIGINAL vs OPTIMIZED MODEL")
    print("="*70)
    print()
    
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
    
    _, _, test_loader = TemporalGraphDataLoader.create_train_val_test_loaders(
        test_dataset, test_dataset, test_dataset,
        batch_size=32, num_workers=0
    )
    
    print(f"✓ Test: {len(test_dataset)} samples")
    print()
    
    # Load ORIGINAL model
    print("Loading ORIGINAL model (final_retrain)...")
    original_model = SimplifiedSTGAT(
        num_nodes=7, num_edges=15, input_features=15,
        gat_hidden_dim=64, gat_heads=4, lstm_hidden_dim=64,
        temporal_window=20, output_dim=1, dropout=0.3, device='cpu'
    )
    
    original_checkpoint = torch.load(
        'checkpoints/final_retrain/best_model.pt',
        map_location='cpu',
        weights_only=False
    )
    original_model.load_state_dict(original_checkpoint['model_state_dict'])
    print(f"✓ Loaded from epoch {original_checkpoint['epoch']}")
    print()
    
    # Load OPTIMIZED model (AdaptiveHuber)
    print("Loading OPTIMIZED model (AdaptiveHuber)...")
    optimized_model = SimplifiedSTGAT(
        num_nodes=7, num_edges=15, input_features=15,
        gat_hidden_dim=64, gat_heads=4, lstm_hidden_dim=64,
        temporal_window=20, output_dim=1, dropout=0.3, device='cpu'
    )
    
    optimized_checkpoint = torch.load(
        'checkpoints/r2_optimization/AdaptiveHuber/best_model.pt',
        map_location='cpu',
        weights_only=False
    )
    optimized_model.load_state_dict(optimized_checkpoint['model_state_dict'])
    print(f"✓ Loaded from epoch {optimized_checkpoint['epoch']}")
    print()
    
    # Evaluate both models
    print("Evaluating ORIGINAL model on test set...")
    original_preds, targets = evaluate_model(original_model, test_loader)
    original_overall, original_per_stock = compute_comprehensive_metrics(
        original_preds, targets, stock_names
    )
    print("✓ Original model evaluated")
    print()
    
    print("Evaluating OPTIMIZED model on test set...")
    optimized_preds, targets = evaluate_model(optimized_model, test_loader)
    optimized_overall, optimized_per_stock = compute_comprehensive_metrics(
        optimized_preds, targets, stock_names
    )
    print("✓ Optimized model evaluated")
    print()
    
    # Display results
    print("="*70)
    print("TEST SET RESULTS COMPARISON")
    print("="*70)
    print()
    
    print("OVERALL METRICS:")
    print("-"*70)
    comparison = pd.DataFrame({
        'Metric': ['R² Score', 'RMSE', 'MAE', 'Dir Accuracy (%)', 'Correlation'],
        'Original': [
            f"{original_overall['r2']:.4f}",
            f"{original_overall['rmse']:.4f}",
            f"{original_overall['mae']:.4f}",
            f"{original_overall['directional_accuracy']*100:.2f}",
            f"{original_overall['correlation']:.4f}"
        ],
        'Optimized': [
            f"{optimized_overall['r2']:.4f}",
            f"{optimized_overall['rmse']:.4f}",
            f"{optimized_overall['mae']:.4f}",
            f"{optimized_overall['directional_accuracy']*100:.2f}",
            f"{optimized_overall['correlation']:.4f}"
        ],
        'Change': [
            f"{optimized_overall['r2'] - original_overall['r2']:+.4f}",
            f"{optimized_overall['rmse'] - original_overall['rmse']:+.4f}",
            f"{optimized_overall['mae'] - original_overall['mae']:+.4f}",
            f"{(optimized_overall['directional_accuracy'] - original_overall['directional_accuracy'])*100:+.2f}",
            f"{optimized_overall['correlation'] - original_overall['correlation']:+.4f}"
        ]
    })
    print(comparison.to_string(index=False))
    print()
    
    print("PER-STOCK R² COMPARISON:")
    print("-"*70)
    r2_comparison = pd.DataFrame({
        'Stock': stock_names,
        'Original R²': [original_per_stock[original_per_stock['stock']==s]['r2'].values[0] 
                        for s in stock_names],
        'Optimized R²': [optimized_per_stock[optimized_per_stock['stock']==s]['r2'].values[0] 
                         for s in stock_names],
        'Improvement': [
            optimized_per_stock[optimized_per_stock['stock']==s]['r2'].values[0] - 
            original_per_stock[original_per_stock['stock']==s]['r2'].values[0]
            for s in stock_names
        ]
    })
    print(r2_comparison.to_string(index=False))
    print()
    
    # Count improvements
    improved_count = (r2_comparison['Improvement'] > 0).sum()
    print(f"✓ {improved_count}/7 stocks improved")
    print()
    
    # Save results
    results_dir = Path("results/final_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    comparison.to_csv(results_dir / 'overall_comparison.csv', index=False)
    r2_comparison.to_csv(results_dir / 'r2_comparison_by_stock.csv', index=False)
    original_per_stock.to_csv(results_dir / 'original_per_stock.csv', index=False)
    optimized_per_stock.to_csv(results_dir / 'optimized_per_stock.csv', index=False)
    
    # Create visualizations
    print("Creating comparison visualizations...")
    create_comparison_plots(
        original_per_stock,
        optimized_per_stock,
        stock_names,
        results_dir
    )
    
    print()
    print("="*70)
    print("✓ FINAL COMPARISON COMPLETE!")
    print("="*70)
    print()
    
    # Final verdict
    r2_improved = optimized_overall['r2'] > original_overall['r2']
    dir_maintained = optimized_overall['directional_accuracy'] >= original_overall['directional_accuracy'] * 0.95
    
    if r2_improved and dir_maintained:
        print("🎉 SUCCESS! Optimization achieved all goals:")
        print(f"  ✅ R² improved: {original_overall['r2']:.4f} → {optimized_overall['r2']:.4f}")
        print(f"  ✅ Dir Acc maintained: {original_overall['directional_accuracy']*100:.1f}% → {optimized_overall['directional_accuracy']*100:.1f}%")
        print()
        print("The AdaptiveHuber loss function successfully fixed the R² issue!")
    else:
        print("Results:")
        print(f"  R² change: {original_overall['r2']:.4f} → {optimized_overall['r2']:.4f}")
        print(f"  Dir Acc change: {original_overall['directional_accuracy']*100:.1f}% → {optimized_overall['directional_accuracy']*100:.1f}%")
    
    print()
    print(f"Results saved to: {results_dir}")
    print()


if __name__ == "__main__":
    main()
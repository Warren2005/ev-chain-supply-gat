"""
Phase 7 - Step 2: Evaluate All Baselines vs ST-GAT

Comprehensive evaluation comparing:
1. ST-GAT (Optimized with AdaptiveHuber)
2. GARCH(1,1)
3. VAR(5)
4. Simple LSTM
5. Persistence

Metrics: R², RMSE, MAE, Directional Accuracy, Correlation

Author: EV Supply Chain GAT Team
Date: December 2024
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime

from models.simplified_st_gat import SimplifiedSTGAT
from models.baseline_models import GARCHBaseline, VARBaseline, SimpleLSTM, PersistenceBaseline
from utils.temporal_dataset import TemporalGraphDataset
from utils.graph_dataloader import TemporalGraphDataLoader


def extract_targets_from_dataset(dataset):
    """Extract all targets from dataset"""
    targets = []
    for _, _, target in dataset:
        targets.append(target.numpy())
    return np.array(targets)


def evaluate_model(model, test_loader, model_name, device='cpu'):
    """Evaluate neural network model"""
    print(f"\nEvaluating {model_name}...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for features, edge_index, targets in test_loader:
            features = features.to(device)
            
            if model_name == "SimpleLSTM":
                # Simple LSTM doesn't use edge_index
                predictions = model(features).squeeze(-1)
            else:
                # ST-GAT uses edge_index
                edge_index = edge_index.to(device)
                predictions = model(features, edge_index).squeeze(-1)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    return predictions, targets


def compute_metrics(predictions, targets, stock_names):
    """Compute comprehensive metrics"""
    
    # Overall metrics
    overall = {
        'r2': r2_score(targets.flatten(), predictions.flatten()),
        'rmse': np.sqrt(mean_squared_error(targets.flatten(), predictions.flatten())),
        'mae': mean_absolute_error(targets.flatten(), predictions.flatten()),
        'directional_accuracy': (np.sign(predictions) == np.sign(targets)).mean(),
        'correlation': np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]
    }
    
    # Per-stock metrics
    per_stock = []
    for i, stock in enumerate(stock_names):
        preds = predictions[:, i]
        targs = targets[:, i]
        
        per_stock.append({
            'stock': stock,
            'r2': r2_score(targs, preds),
            'rmse': np.sqrt(mean_squared_error(targs, preds)),
            'mae': mean_absolute_error(targs, preds),
            'directional_accuracy': (np.sign(preds) == np.sign(targs)).mean(),
            'correlation': np.corrcoef(preds, targs)[0, 1] if len(np.unique(preds)) > 1 else 0.0
        })
    
    return overall, pd.DataFrame(per_stock)


def create_comparison_plots(results_dict, save_dir):
    """Create comprehensive comparison visualizations"""
    
    print("\nCreating comparison visualizations...")
    
    # Extract data
    models = list(results_dict.keys())
    
    # Plot 1: Overall R² comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # R² comparison
    ax = axes[0, 0]
    r2_scores = [results_dict[m]['overall']['r2'] for m in models]
    colors = ['green' if r2 > 0 else 'red' for r2 in r2_scores]
    
    bars = ax.bar(models, r2_scores, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('R² Score', fontsize=11)
    ax.set_title('Overall R² Score Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02 if height > 0 else height - 0.05,
                f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # Directional Accuracy comparison
    ax = axes[0, 1]
    dir_accs = [results_dict[m]['overall']['directional_accuracy'] * 100 for m in models]
    
    bars = ax.bar(models, dir_accs, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1, label='Random (50%)')
    ax.set_ylabel('Directional Accuracy (%)', fontsize=11)
    ax.set_title('Overall Directional Accuracy', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, dir_accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # RMSE comparison
    ax = axes[1, 0]
    rmse_scores = [results_dict[m]['overall']['rmse'] for m in models]
    
    bars = ax.bar(models, rmse_scores, color='coral', alpha=0.7, edgecolor='black')
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('Overall RMSE (Lower is Better)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, rmse_scores):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Correlation comparison
    ax = axes[1, 1]
    corr_scores = [results_dict[m]['overall']['correlation'] for m in models]
    
    bars = ax.bar(models, corr_scores, color='mediumpurple', alpha=0.7, edgecolor='black')
    ax.set_ylabel('Correlation', fontsize=11)
    ax.set_title('Overall Correlation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, corr_scores):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'overall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_dir / 'overall_comparison.png'}")
    
    # Plot 2: Per-stock R² heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    stock_names = results_dict[models[0]]['per_stock']['stock'].values
    r2_matrix = np.array([
        results_dict[m]['per_stock']['r2'].values for m in models
    ])
    
    sns.heatmap(
        r2_matrix,
        xticklabels=stock_names,
        yticklabels=models,
        cmap='RdYlGn',
        center=0,
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'R² Score'},
        ax=ax
    )
    
    ax.set_title('R² Score by Model and Stock', fontsize=13, fontweight='bold')
    ax.set_xlabel('Stock', fontsize=11)
    ax.set_ylabel('Model', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'r2_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_dir / 'r2_heatmap.png'}")
    
    # Plot 3: Per-stock directional accuracy
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(stock_names))
    width = 0.15
    
    for i, model in enumerate(models):
        dir_acc = results_dict[model]['per_stock']['directional_accuracy'].values * 100
        offset = (i - len(models)/2 + 0.5) * width
        ax.bar(x + offset, dir_acc, width, label=model, alpha=0.8)
    
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random (50%)')
    ax.set_xlabel('Stock', fontsize=11)
    ax.set_ylabel('Directional Accuracy (%)', fontsize=11)
    ax.set_title('Directional Accuracy by Model and Stock', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stock_names)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'directional_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_dir / 'directional_accuracy_comparison.png'}")


def main():
    """Main evaluation"""
    print("="*70)
    print("PHASE 7 - STEP 2: BASELINE EVALUATION")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup
    stock_names = ['ALB', 'APTV', 'F', 'GM', 'MGA', 'SQM', 'TSLA']
    edge_index = torch.tensor([
        [0, 0, 0, 4, 4, 4, 1, 1, 1, 0, 0, 4, 5, 5, 5],
        [6, 2, 3, 6, 2, 3, 2, 3, 6, 4, 1, 1, 6, 2, 3]
    ], dtype=torch.long)
    
    results_dir = Path("results/phase7_baselines")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = TemporalGraphDataset(
        data_path="data/processed/train_features_fixed.parquet",
        edge_index=edge_index,
        window_size=20
    )
    
    test_dataset = TemporalGraphDataset(
        data_path="data/processed/test_features_fixed.parquet",
        edge_index=edge_index,
        window_size=20
    )
    
    _, _, test_loader = TemporalGraphDataLoader.create_train_val_test_loaders(
        test_dataset, test_dataset, test_dataset,
        batch_size=32, num_workers=0
    )
    
    train_targets = extract_targets_from_dataset(train_dataset)
    test_targets = extract_targets_from_dataset(test_dataset)
    
    print(f"✓ Test: {len(test_dataset)} samples")
    print()
    
    # Store all results
    all_results = {}
    
    # 1. Evaluate ST-GAT (Optimized)
    print("="*70)
    print("MODEL 1: ST-GAT (Optimized with AdaptiveHuber)")
    print("="*70)
    
    st_gat = SimplifiedSTGAT(
        num_nodes=7, num_edges=15, input_features=15,
        gat_hidden_dim=64, gat_heads=4, lstm_hidden_dim=64,
        temporal_window=20, output_dim=1, dropout=0.3, device='cpu'
    )
    
    checkpoint = torch.load(
        'checkpoints/r2_optimization/AdaptiveHuber/best_model.pt',
        map_location='cpu',
        weights_only=False
    )
    st_gat.load_state_dict(checkpoint['model_state_dict'])
    
    st_gat_preds, targets = evaluate_model(st_gat, test_loader, "ST-GAT", device='cpu')
    st_gat_overall, st_gat_per_stock = compute_metrics(st_gat_preds, targets, stock_names)
    
    all_results['ST-GAT'] = {
        'overall': st_gat_overall,
        'per_stock': st_gat_per_stock,
        'predictions': st_gat_preds
    }
    
    print(f"✓ ST-GAT R²: {st_gat_overall['r2']:.4f}, Dir Acc: {st_gat_overall['directional_accuracy']*100:.1f}%")
    
    # 2. Evaluate GARCH
    print("\n" + "="*70)
    print("MODEL 2: GARCH(1,1)")
    print("="*70)
    
    garch = GARCHBaseline(stock_names)
    garch.fit(train_targets)
    garch_preds = garch.evaluate(test_targets)
    
    garch_overall, garch_per_stock = compute_metrics(garch_preds, test_targets, stock_names)
    
    all_results['GARCH'] = {
        'overall': garch_overall,
        'per_stock': garch_per_stock,
        'predictions': garch_preds
    }
    
    print(f"✓ GARCH R²: {garch_overall['r2']:.4f}, Dir Acc: {garch_overall['directional_accuracy']*100:.1f}%")
    
    # 3. Evaluate VAR
    print("\n" + "="*70)
    print("MODEL 3: VAR(5)")
    print("="*70)
    
    var = VARBaseline(stock_names, lag_order=5)
    var.fit(train_targets)
    var_preds = var.evaluate(test_targets, train_targets)
    
    var_overall, var_per_stock = compute_metrics(var_preds, test_targets, stock_names)
    
    all_results['VAR'] = {
        'overall': var_overall,
        'per_stock': var_per_stock,
        'predictions': var_preds
    }
    
    print(f"✓ VAR R²: {var_overall['r2']:.4f}, Dir Acc: {var_overall['directional_accuracy']*100:.1f}%")
    
    # 4. Evaluate Simple LSTM
    print("\n" + "="*70)
    print("MODEL 4: Simple LSTM (No Graph)")
    print("="*70)
    
    simple_lstm = SimpleLSTM(
        num_stocks=7, input_features=15,
        hidden_dim=64, num_layers=1, dropout=0.3
    )
    simple_lstm.load_state_dict(torch.load('checkpoints/baselines/simple_lstm_best.pt', weights_only=True))
    
    lstm_preds, _ = evaluate_model(simple_lstm, test_loader, "SimpleLSTM", device='cpu')
    lstm_overall, lstm_per_stock = compute_metrics(lstm_preds, test_targets, stock_names)
    
    all_results['SimpleLSTM'] = {
        'overall': lstm_overall,
        'per_stock': lstm_per_stock,
        'predictions': lstm_preds
    }
    
    print(f"✓ SimpleLSTM R²: {lstm_overall['r2']:.4f}, Dir Acc: {lstm_overall['directional_accuracy']*100:.1f}%")
    
    # 5. Evaluate Persistence
    print("\n" + "="*70)
    print("MODEL 5: Persistence (Naive Baseline)")
    print("="*70)
    
    persistence = PersistenceBaseline(stock_names)
    persistence.fit(train_targets)
    last_train_value = train_targets[-1]
    persistence_preds = persistence.evaluate(test_targets, last_train_value)
    
    persistence_overall, persistence_per_stock = compute_metrics(persistence_preds, test_targets, stock_names)
    
    all_results['Persistence'] = {
        'overall': persistence_overall,
        'per_stock': persistence_per_stock,
        'predictions': persistence_preds
    }
    
    print(f"✓ Persistence R²: {persistence_overall['r2']:.4f}, Dir Acc: {persistence_overall['directional_accuracy']*100:.1f}%")
    
    # Create comparison table
    print("\n" + "="*70)
    print("OVERALL COMPARISON")
    print("="*70)
    print()
    
    comparison_df = pd.DataFrame([
        {
            'Model': model,
            'R²': f"{results['overall']['r2']:.4f}",
            'RMSE': f"{results['overall']['rmse']:.4f}",
            'MAE': f"{results['overall']['mae']:.4f}",
            'Dir Acc (%)': f"{results['overall']['directional_accuracy']*100:.2f}",
            'Correlation': f"{results['overall']['correlation']:.4f}"
        }
        for model, results in all_results.items()
    ])
    
    print(comparison_df.to_string(index=False))
    print()
    
    # Save results
    comparison_df.to_csv(results_dir / 'overall_comparison.csv', index=False)
    
    # Save per-stock results
    for model, results in all_results.items():
        results['per_stock'].to_csv(results_dir / f'{model}_per_stock.csv', index=False)
    
    # Create visualizations
    create_comparison_plots(all_results, results_dir)
    
    # Ranking
    print("="*70)
    print("MODEL RANKING")
    print("="*70)
    print()
    
    # Rank by R²
    r2_ranking = sorted(all_results.items(), key=lambda x: x[1]['overall']['r2'], reverse=True)
    print("By R² Score:")
    for i, (model, results) in enumerate(r2_ranking, 1):
        print(f"  {i}. {model}: {results['overall']['r2']:.4f}")
    print()
    
    # Rank by Directional Accuracy
    dir_ranking = sorted(all_results.items(), key=lambda x: x[1]['overall']['directional_accuracy'], reverse=True)
    print("By Directional Accuracy:")
    for i, (model, results) in enumerate(dir_ranking, 1):
        print(f"  {i}. {model}: {results['overall']['directional_accuracy']*100:.2f}%")
    print()
    
    print("="*70)
    print("✓ BASELINE EVALUATION COMPLETE!")
    print("="*70)
    print()
    print(f"Results saved to: {results_dir}")
    print()
    print("Next: Statistical significance testing")
    print("  python scripts/test_significance_phase7.py")
    print()


if __name__ == "__main__":
    main()
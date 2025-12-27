"""
Phase 7 - Step 3: Statistical Significance Testing

Tests whether performance differences between models are statistically significant:
1. Diebold-Mariano test (forecast accuracy comparison)
2. Paired t-tests (metric differences)
3. Hypothesis testing framework

Author: EV Supply Chain GAT Team
Date: December 2024
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def diebold_mariano_test(errors1, errors2, alternative='two-sided'):
    """
    Diebold-Mariano test for comparing forecast accuracy.
    
    Tests if two models have significantly different forecast errors.
    
    Args:
        errors1: Forecast errors from model 1 [num_samples, num_stocks]
        errors2: Forecast errors from model 2 [num_samples, num_stocks]
        alternative: 'two-sided', 'less', or 'greater'
    
    Returns:
        Dict with test statistic and p-value
    """
    # Compute loss differential (MSE difference)
    d = errors1**2 - errors2**2
    d_mean = d.mean()
    
    # Flatten for overall test
    d_flat = d.flatten()
    
    # Standard error (accounting for autocorrelation with Newey-West)
    n = len(d_flat)
    d_var = d_flat.var(ddof=1)
    
    # Simple standard error (could be improved with HAC)
    se = np.sqrt(d_var / n)
    
    # Test statistic
    dm_stat = d_mean / se
    
    # P-value (asymptotically normal)
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    elif alternative == 'less':
        p_value = stats.norm.cdf(dm_stat)
    else:  # greater
        p_value = 1 - stats.norm.cdf(dm_stat)
    
    return {
        'statistic': dm_stat,
        'p_value': p_value,
        'mean_diff': d_mean,
        'significant': p_value < 0.05
    }


def paired_t_test(metric1, metric2):
    """
    Paired t-test for metric differences.
    
    Args:
        metric1: Metric values from model 1 [num_stocks]
        metric2: Metric values from model 2 [num_stocks]
    
    Returns:
        Dict with test results
    """
    differences = metric1 - metric2
    
    t_stat, p_value = stats.ttest_rel(metric1, metric2)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_diff': differences.mean(),
        'significant': p_value < 0.05
    }


def mcnemar_test(correct1, correct2):
    """
    McNemar's test for directional accuracy comparison.
    
    Tests if two models have significantly different classification accuracy.
    
    Args:
        correct1: Binary array of correct predictions from model 1 [num_samples]
        correct2: Binary array of correct predictions from model 2 [num_samples]
    
    Returns:
        Dict with test results
    """
    # Contingency table
    both_correct = ((correct1 == 1) & (correct2 == 1)).sum()
    both_wrong = ((correct1 == 0) & (correct2 == 0)).sum()
    model1_only = ((correct1 == 1) & (correct2 == 0)).sum()
    model2_only = ((correct1 == 0) & (correct2 == 1)).sum()
    
    # McNemar statistic (with continuity correction)
    numerator = (abs(model1_only - model2_only) - 1) ** 2
    denominator = model1_only + model2_only
    
    if denominator == 0:
        return {'statistic': 0, 'p_value': 1.0, 'significant': False}
    
    mcnemar_stat = numerator / denominator
    p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
    
    return {
        'statistic': mcnemar_stat,
        'p_value': p_value,
        'both_correct': both_correct,
        'both_wrong': both_wrong,
        'model1_only_correct': model1_only,
        'model2_only_correct': model2_only,
        'significant': p_value < 0.05
    }


def compute_prediction_errors(predictions, targets):
    """Compute prediction errors"""
    return predictions - targets


def compute_directional_correct(predictions, targets):
    """Compute binary directional correctness"""
    return (np.sign(predictions) == np.sign(targets)).astype(int)


def create_significance_matrix(results_dict, metric='r2'):
    """
    Create matrix of pairwise significance tests.
    
    Args:
        results_dict: Dictionary of model results
        metric: Metric to compare
    
    Returns:
        DataFrame with p-values
    """
    models = list(results_dict.keys())
    n_models = len(models)
    
    p_matrix = np.ones((n_models, n_models))
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i != j:
                metric1 = results_dict[model1]['per_stock'][metric].values
                metric2 = results_dict[model2]['per_stock'][metric].values
                
                test_result = paired_t_test(metric1, metric2)
                p_matrix[i, j] = test_result['p_value']
    
    return pd.DataFrame(p_matrix, index=models, columns=models)


def main():
    """Main significance testing"""
    print("="*70)
    print("PHASE 7 - STEP 3: STATISTICAL SIGNIFICANCE TESTING")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load results
    results_dir = Path("results/phase7_baselines")
    
    models = ['ST-GAT', 'GARCH', 'VAR', 'SimpleLSTM', 'Persistence']
    
    # Load per-stock results
    results = {}
    for model in models:
        results[model] = {
            'per_stock': pd.read_csv(results_dir / f'{model}_per_stock.csv')
        }
    
    # Load predictions (we'll need to save these in evaluation script)
    # For now, we'll use per-stock metrics
    
    print("Loaded results for all models")
    print()
    
    # Test 1: Pairwise R² comparison (ST-GAT vs others)
    print("="*70)
    print("TEST 1: R² SCORE COMPARISON (Paired t-test)")
    print("="*70)
    print()
    
    print("ST-GAT vs Other Models:")
    print("-"*70)
    
    st_gat_r2 = results['ST-GAT']['per_stock']['r2'].values
    
    comparisons = []
    
    for model in models:
        if model != 'ST-GAT':
            model_r2 = results[model]['per_stock']['r2'].values
            
            test_result = paired_t_test(st_gat_r2, model_r2)
            
            print(f"\nST-GAT vs {model}:")
            print(f"  Mean R² difference: {test_result['mean_diff']:.4f}")
            print(f"  t-statistic: {test_result['t_statistic']:.4f}")
            print(f"  p-value: {test_result['p_value']:.4f}")
            print(f"  Significant: {'YES ✓' if test_result['significant'] else 'NO ✗'}")
            
            if test_result['significant']:
                winner = 'ST-GAT' if test_result['mean_diff'] > 0 else model
                print(f"  → {winner} is significantly better")
            
            comparisons.append({
                'comparison': f'ST-GAT vs {model}',
                'mean_diff': test_result['mean_diff'],
                't_stat': test_result['t_statistic'],
                'p_value': test_result['p_value'],
                'significant': test_result['significant']
            })
    
    print()
    
    # Test 2: SimpleLSTM vs ST-GAT (key comparison)
    print("="*70)
    print("TEST 2: SIMPLELSTM vs ST-GAT (Detailed Analysis)")
    print("="*70)
    print()
    
    lstm_r2 = results['SimpleLSTM']['per_stock']['r2'].values
    st_gat_r2 = results['ST-GAT']['per_stock']['r2'].values
    
    # R² comparison
    r2_test = paired_t_test(lstm_r2, st_gat_r2)
    
    print("R² Comparison:")
    print(f"  SimpleLSTM mean R²: {lstm_r2.mean():.4f}")
    print(f"  ST-GAT mean R²: {st_gat_r2.mean():.4f}")
    print(f"  Difference: {r2_test['mean_diff']:.4f}")
    print(f"  t-statistic: {r2_test['t_statistic']:.4f}")
    print(f"  p-value: {r2_test['p_value']:.6f}")
    print(f"  Significant: {'YES ✓' if r2_test['significant'] else 'NO ✗'}")
    print()
    
    # RMSE comparison
    lstm_rmse = results['SimpleLSTM']['per_stock']['rmse'].values
    st_gat_rmse = results['ST-GAT']['per_stock']['rmse'].values
    
    rmse_test = paired_t_test(st_gat_rmse, lstm_rmse)  # Lower is better
    
    print("RMSE Comparison:")
    print(f"  SimpleLSTM mean RMSE: {lstm_rmse.mean():.4f}")
    print(f"  ST-GAT mean RMSE: {st_gat_rmse.mean():.4f}")
    print(f"  Difference: {rmse_test['mean_diff']:.4f}")
    print(f"  t-statistic: {rmse_test['t_statistic']:.4f}")
    print(f"  p-value: {rmse_test['p_value']:.6f}")
    print(f"  Significant: {'YES ✓' if rmse_test['significant'] else 'NO ✗'}")
    print()
    
    # Directional accuracy comparison
    lstm_dir = results['SimpleLSTM']['per_stock']['directional_accuracy'].values
    st_gat_dir = results['ST-GAT']['per_stock']['directional_accuracy'].values
    
    dir_test = paired_t_test(lstm_dir, st_gat_dir)
    
    print("Directional Accuracy Comparison:")
    print(f"  SimpleLSTM mean: {lstm_dir.mean()*100:.2f}%")
    print(f"  ST-GAT mean: {st_gat_dir.mean()*100:.2f}%")
    print(f"  Difference: {dir_test['mean_diff']*100:.2f}%")
    print(f"  t-statistic: {dir_test['t_statistic']:.4f}")
    print(f"  p-value: {dir_test['p_value']:.6f}")
    print(f"  Significant: {'YES ✓' if dir_test['significant'] else 'NO ✗'}")
    print()
    
    # Test 3: Pairwise significance matrix
    print("="*70)
    print("TEST 3: PAIRWISE SIGNIFICANCE MATRIX (R²)")
    print("="*70)
    print()
    
    p_matrix = create_significance_matrix(results, metric='r2')
    
    print("P-values (row vs column):")
    print(p_matrix.round(4).to_string())
    print()
    print("Significant differences (p < 0.05):")
    sig_matrix = (p_matrix < 0.05).astype(int)
    print(sig_matrix.to_string())
    print()
    
    # Test 4: Effect sizes
    print("="*70)
    print("TEST 4: EFFECT SIZES (Cohen's d)")
    print("="*70)
    print()
    
    def cohens_d(x1, x2):
        """Compute Cohen's d effect size"""
        n1, n2 = len(x1), len(x2)
        var1, var2 = x1.var(ddof=1), x2.var(ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (x1.mean() - x2.mean()) / pooled_std
    
    print("SimpleLSTM vs ST-GAT Effect Sizes:")
    print("-"*70)
    
    r2_effect = cohens_d(lstm_r2, st_gat_r2)
    rmse_effect = cohens_d(st_gat_rmse, lstm_rmse)  # Reversed (lower better)
    dir_effect = cohens_d(lstm_dir, st_gat_dir)
    
    print(f"  R² Cohen's d: {r2_effect:.4f} ", end='')
    if abs(r2_effect) < 0.2:
        print("(small effect)")
    elif abs(r2_effect) < 0.5:
        print("(medium effect)")
    else:
        print("(large effect)")
    
    print(f"  RMSE Cohen's d: {rmse_effect:.4f} ", end='')
    if abs(rmse_effect) < 0.2:
        print("(small effect)")
    elif abs(rmse_effect) < 0.5:
        print("(medium effect)")
    else:
        print("(large effect)")
    
    print(f"  Dir Acc Cohen's d: {dir_effect:.4f} ", end='')
    if abs(dir_effect) < 0.2:
        print("(small effect)")
    elif abs(dir_effect) < 0.5:
        print("(medium effect)")
    else:
        print("(large effect)")
    
    print()
    
    # Save results
    print("="*70)
    print("SAVING RESULTS")
    print("="*70)
    print()
    
    # Save comparison table
    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(results_dir / 'significance_tests.csv', index=False)
    
    # Save detailed SimpleLSTM vs ST-GAT results
    with open(results_dir / 'simplelstm_vs_stgat_test.txt', 'w') as f:
        f.write("SIMPLELSTM vs ST-GAT SIGNIFICANCE TEST\n")
        f.write("="*70 + "\n\n")
        
        f.write("R² Score Comparison:\n")
        f.write(f"  SimpleLSTM mean: {lstm_r2.mean():.4f}\n")
        f.write(f"  ST-GAT mean: {st_gat_r2.mean():.4f}\n")
        f.write(f"  Difference: {r2_test['mean_diff']:.4f}\n")
        f.write(f"  t-statistic: {r2_test['t_statistic']:.4f}\n")
        f.write(f"  p-value: {r2_test['p_value']:.6f}\n")
        f.write(f"  Cohen's d: {r2_effect:.4f}\n")
        f.write(f"  Significant: {'YES' if r2_test['significant'] else 'NO'}\n\n")
        
        f.write("RMSE Comparison:\n")
        f.write(f"  SimpleLSTM mean: {lstm_rmse.mean():.4f}\n")
        f.write(f"  ST-GAT mean: {st_gat_rmse.mean():.4f}\n")
        f.write(f"  Difference: {rmse_test['mean_diff']:.4f}\n")
        f.write(f"  t-statistic: {rmse_test['t_statistic']:.4f}\n")
        f.write(f"  p-value: {rmse_test['p_value']:.6f}\n")
        f.write(f"  Cohen's d: {rmse_effect:.4f}\n")
        f.write(f"  Significant: {'YES' if rmse_test['significant'] else 'NO'}\n\n")
        
        f.write("Directional Accuracy Comparison:\n")
        f.write(f"  SimpleLSTM mean: {lstm_dir.mean()*100:.2f}%\n")
        f.write(f"  ST-GAT mean: {st_gat_dir.mean()*100:.2f}%\n")
        f.write(f"  Difference: {dir_test['mean_diff']*100:.2f}%\n")
        f.write(f"  t-statistic: {dir_test['t_statistic']:.4f}\n")
        f.write(f"  p-value: {dir_test['p_value']:.6f}\n")
        f.write(f"  Cohen's d: {dir_effect:.4f}\n")
        f.write(f"  Significant: {'YES' if dir_test['significant'] else 'NO'}\n")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    stocks = results['ST-GAT']['per_stock']['stock'].values
    
    # R² comparison
    ax = axes[0]
    x = np.arange(len(stocks))
    width = 0.35
    
    ax.bar(x - width/2, st_gat_r2, width, label='ST-GAT', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, lstm_r2, width, label='SimpleLSTM', alpha=0.8, color='coral')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Stock')
    ax.set_ylabel('R² Score')
    ax.set_title(f'R² Comparison\n(p={r2_test["p_value"]:.4f})')
    ax.set_xticks(x)
    ax.set_xticklabels(stocks)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # RMSE comparison
    ax = axes[1]
    ax.bar(x - width/2, st_gat_rmse, width, label='ST-GAT', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, lstm_rmse, width, label='SimpleLSTM', alpha=0.8, color='coral')
    ax.set_xlabel('Stock')
    ax.set_ylabel('RMSE')
    ax.set_title(f'RMSE Comparison\n(p={rmse_test["p_value"]:.4f})')
    ax.set_xticks(x)
    ax.set_xticklabels(stocks)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Directional accuracy comparison
    ax = axes[2]
    ax.bar(x - width/2, st_gat_dir*100, width, label='ST-GAT', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, lstm_dir*100, width, label='SimpleLSTM', alpha=0.8, color='coral')
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Stock')
    ax.set_ylabel('Directional Accuracy (%)')
    ax.set_title(f'Dir Acc Comparison\n(p={dir_test["p_value"]:.4f})')
    ax.set_xticks(x)
    ax.set_xticklabels(stocks)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'significance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {results_dir / 'significance_tests.csv'}")
    print(f"✓ Saved: {results_dir / 'simplelstm_vs_stgat_test.txt'}")
    print(f"✓ Saved: {results_dir / 'significance_comparison.png'}")
    print()
    
    # Final summary
    print("="*70)
    print("SUMMARY OF FINDINGS")
    print("="*70)
    print()
    
    print("KEY RESULT: SimpleLSTM vs ST-GAT")
    print("-"*70)
    print(f"SimpleLSTM R²: {lstm_r2.mean():.4f}")
    print(f"ST-GAT R²: {st_gat_r2.mean():.4f}")
    print(f"Improvement: {((lstm_r2.mean() / st_gat_r2.mean()) - 1) * 100:.1f}%")
    print(f"Significance: p={r2_test['p_value']:.6f}")
    print(f"Effect size (Cohen's d): {r2_effect:.4f} (large)" if abs(r2_effect) > 0.8 else f"Effect size (Cohen's d): {r2_effect:.4f}")
    print()
    
    if r2_test['significant']:
        print("✓ SimpleLSTM is SIGNIFICANTLY better than ST-GAT (p < 0.05)")
        print()
        print("CONCLUSION:")
        print("  Graph structure does NOT improve volatility prediction.")
        print("  Simple temporal models outperform graph attention networks.")
        print("  This is a valid negative result for publication.")
    else:
        print("✗ No significant difference found")
    
    print()
    print("="*70)
    print("✓ STATISTICAL SIGNIFICANCE TESTING COMPLETE!")
    print("="*70)
    print()
    print("Next: Final research report")
    print("  python scripts/create_final_report_phase7.py")
    print()


if __name__ == "__main__":
    main()
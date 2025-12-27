"""
Phase 6: Extract and Analyze Attention Weights

Systematically extracts attention from trained model and performs:
1. Attention weight extraction across test set
2. Entropy calculation and hypothesis testing
3. Critical pathway identification
4. Temporal dynamics analysis

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
from datetime import datetime

from models.simplified_st_gat import SimplifiedSTGAT
from utils.temporal_dataset import TemporalGraphDataset
from utils.graph_dataloader import TemporalGraphDataLoader
from utils.attention_analyzer import AttentionAnalyzer


def extract_attention_from_model(
    model,
    features,
    edge_index,
    device='cpu'
):
    """
    Extract attention weights from model forward pass.
    
    Works with MultiHeadGATLayer structure:
    - model.gat is a MultiHeadGATLayer with 4 heads
    - Each head stores attention_weights after forward pass
    - We average attention across heads
    """
    batch_size, num_nodes, seq_len, feat_dim = features.shape
    
    # Store attention per timestep
    attention_per_timestep = []
    
    model.eval()
    with torch.no_grad():
        # Process each timestep through GAT
        for t in range(seq_len):
            x_t = features[:, :, t, :]
            x_t_flat = x_t.reshape(batch_size * num_nodes, feat_dim)
            
            # Forward through GAT (MultiHeadGATLayer)
            # This will cause each head to store attention in self.attention_weights
            _ = model.gat(x_t_flat, edge_index)
            
            # Extract attention from all heads
            head_attentions = []
            for head in model.gat.attention_heads:
                if hasattr(head, 'attention_weights'):
                    head_attentions.append(head.attention_weights.cpu())
            
            if len(head_attentions) > 0:
                # Average attention across all heads
                avg_attention = torch.stack(head_attentions).mean(dim=0)
                attention_per_timestep.append(avg_attention)
    
    return attention_per_timestep


def extract_all_attention_weights(
    model,
    dataloader,
    edge_index,
    device='cpu'
):
    """
    Extract attention for entire dataset.
    
    Returns:
        List of attention weights per sample [num_samples, num_edges]
    """
    all_attention = []
    all_targets = []
    
    print("Extracting attention weights from test set...")
    
    sample_count = 0
    
    for batch_idx, (features, edge_idx, targets) in enumerate(dataloader):
        features = features.to(device)
        
        batch_size = features.shape[0]
        
        for b in range(batch_size):
            # Single sample
            sample_features = features[b:b+1]
            
            # Extract attention
            attention_per_t = extract_attention_from_model(
                model, sample_features, edge_index, device
            )
            
            # Check if we got attention
            if len(attention_per_t) == 0:
                print(f"  Warning: No attention extracted for sample {sample_count}")
                continue
            
            # Average across timesteps
            avg_attention = torch.stack(attention_per_t).mean(dim=0)
            
            all_attention.append(avg_attention)
            all_targets.append(targets[b].mean().item())
            sample_count += 1
        
        if (batch_idx + 1) % 5 == 0:
            print(f"  Processed batch {batch_idx + 1}/{len(dataloader)}, total samples: {sample_count}")
    
    print(f"✓ Extracted attention for {len(all_attention)} samples")
    
    return all_attention, all_targets


def main():
    """Main Phase 6 execution"""
    print("="*70)
    print("PHASE 6: ATTENTION ANALYSIS & INTERPRETABILITY")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup
    stock_names = ['ALB', 'APTV', 'F', 'GM', 'MGA', 'SQM', 'TSLA']
    edge_index = torch.tensor([
        [0, 0, 0, 4, 4, 4, 1, 1, 1, 0, 0, 4, 5, 5, 5],
        [6, 2, 3, 6, 2, 3, 2, 3, 6, 4, 1, 1, 6, 2, 3]
    ], dtype=torch.long)
    
    # Load test dataset
    print("Step 1/5: Loading test dataset...")
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
    
    # Load optimized model
    print("Step 2/5: Loading optimized model...")
    model = SimplifiedSTGAT(
        num_nodes=7, num_edges=15, input_features=15,
        gat_hidden_dim=64, gat_heads=4, lstm_hidden_dim=64,
        temporal_window=20, output_dim=1, dropout=0.3, device='cpu'
    )
    
    checkpoint = torch.load(
        'checkpoints/r2_optimization/AdaptiveHuber/best_model.pt',
        map_location='cpu',
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded from epoch {checkpoint['epoch']}")
    print()
    
    # Extract attention
    print("Step 3/5: Extracting attention weights...")
    attention_weights, volatility_levels = extract_all_attention_weights(
        model, test_loader, edge_index, device='cpu'
    )
    print()
    
    # Initialize analyzer
    print("Step 4/5: Analyzing attention patterns...")
    analyzer = AttentionAnalyzer(
        model=model,
        edge_index=edge_index,
        stock_names=stock_names,
        device='cpu'
    )
    
    # Store extracted data
    analyzer.attention_weights = attention_weights
    analyzer.volatility_levels = volatility_levels
    analyzer.timestamps = list(range(len(attention_weights)))
    
    # Compute entropy for each sample
    analyzer.attention_entropy = [
        analyzer.compute_attention_entropy(attn)
        for attn in attention_weights
    ]
    
    print(f"✓ Computed entropy for {len(analyzer.attention_entropy)} samples")
    print()
    
    # Analysis 1: Critical edges
    print("Analysis 1: Identifying critical supply chain edges...")
    avg_attention = torch.stack(attention_weights).mean(dim=0)
    critical_edges = analyzer.identify_critical_edges(avg_attention, top_k=5)
    
    print("\nTop 5 Most Important Supply Chain Links:")
    print("-"*70)
    for i, (source, target, weight) in enumerate(critical_edges, 1):
        print(f"{i}. {source} → {target}: {weight:.4f}")
    print()
    
    # Analysis 2: Entropy hypothesis testing
    print("Analysis 2: Testing entropy-volatility hypothesis...")
    hypothesis_result = analyzer.test_entropy_hypothesis()
    
    print("\nHypothesis: Attention entropy increases before volatility events")
    print("-"*70)
    if 'error' not in hypothesis_result:
        print(f"Mean entropy before events: {hypothesis_result['mean_entropy_before_events']:.4f}")
        print(f"Mean entropy (normal):      {hypothesis_result['mean_entropy_normal']:.4f}")
        print(f"Difference:                 {hypothesis_result['difference']:+.4f}")
        print(f"t-statistic:                {hypothesis_result['t_statistic']:.4f}")
        print(f"p-value:                    {hypothesis_result['p_value']:.4f}")
        
        if hypothesis_result['significant']:
            print("✓ SIGNIFICANT: Hypothesis supported (p < 0.05)")
        else:
            print("✗ NOT SIGNIFICANT: Insufficient evidence (p ≥ 0.05)")
    else:
        print(f"Error: {hypothesis_result['error']}")
    print()
    
    # Analysis 3: Attention distribution
    print("Analysis 3: Analyzing attention distribution...")
    
    # Per-edge statistics
    attention_matrix = torch.stack(attention_weights).numpy()  # [samples, edges]
    
    edge_stats = []
    for edge_idx in range(edge_index.shape[1]):
        source_idx = edge_index[0, edge_idx].item()
        target_idx = edge_index[1, edge_idx].item()
        
        edge_attention = attention_matrix[:, edge_idx]
        
        edge_stats.append({
            'edge': f"{stock_names[source_idx]}→{stock_names[target_idx]}",
            'source': stock_names[source_idx],
            'target': stock_names[target_idx],
            'mean_attention': edge_attention.mean(),
            'std_attention': edge_attention.std(),
            'max_attention': edge_attention.max(),
            'min_attention': edge_attention.min()
        })
    
    edge_stats_df = pd.DataFrame(edge_stats).sort_values('mean_attention', ascending=False)
    
    print("\nAttention Distribution by Edge:")
    print("-"*70)
    print(edge_stats_df.to_string(index=False))
    print()
    
    # Save results
    print("Step 5/5: Saving results...")
    results_dir = Path("results/phase6_attention")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save attention data
    analyzer._save_attention_data(results_dir)
    
    # Save edge statistics
    edge_stats_df.to_csv(results_dir / 'edge_attention_stats.csv', index=False)
    
    # Save hypothesis test results
    if 'error' not in hypothesis_result:
        with open(results_dir / 'hypothesis_test.txt', 'w') as f:
            f.write("HYPOTHESIS TEST RESULTS\n")
            f.write("="*70 + "\n\n")
            f.write("Hypothesis: Attention entropy increases before volatility events\n\n")
            f.write(f"Mean entropy before events: {hypothesis_result['mean_entropy_before_events']:.4f}\n")
            f.write(f"Mean entropy (normal):      {hypothesis_result['mean_entropy_normal']:.4f}\n")
            f.write(f"Difference:                 {hypothesis_result['difference']:+.4f}\n")
            f.write(f"t-statistic:                {hypothesis_result['t_statistic']:.4f}\n")
            f.write(f"p-value:                    {hypothesis_result['p_value']:.4f}\n")
            f.write(f"\nResult: {'SIGNIFICANT' if hypothesis_result['significant'] else 'NOT SIGNIFICANT'}\n")
    
    print(f"✓ Results saved to: {results_dir}")
    print()
    
    print("="*70)
    print("✓ PHASE 6 ATTENTION EXTRACTION COMPLETE!")
    print("="*70)
    print()
    print("Next: Run visualization script to create attention plots")
    print("  python scripts/visualize_attention_phase6.py")
    print()


if __name__ == "__main__":
    main()
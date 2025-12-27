"""
Phase 6: Visualize Attention Patterns

Creates comprehensive visualizations of attention dynamics:
1. Attention heatmap (which edges are important)
2. Entropy over time (early warning signal)
3. Critical pathways (supply chain analysis)
4. Temporal attention dynamics
5. Attention-volatility correlation

Author: EV Supply Chain GAT Team
Date: December 2024
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime


def load_attention_data(results_dir: Path):
    """Load extracted attention data"""
    attention_df = pd.read_csv(results_dir / 'attention_data.csv')
    edge_stats_df = pd.read_csv(results_dir / 'edge_attention_stats.csv')
    
    return attention_df, edge_stats_df


def create_attention_heatmap(attention_df: pd.DataFrame, save_path: Path):
    """
    Create heatmap showing attention distribution across edges and samples.
    """
    print("Creating attention heatmap...")
    
    # Extract attention columns
    attention_cols = [col for col in attention_df.columns if col.startswith('attn_')]
    
    # Get subset of samples for visualization (every 10th sample)
    sample_indices = np.arange(0, len(attention_df), max(1, len(attention_df) // 50))
    
    # Extract attention matrix
    attention_matrix = attention_df.loc[sample_indices, attention_cols].values.T
    
    # Create edge labels (clean up)
    edge_labels = [col.replace('attn_', '').replace('_', '→') for col in attention_cols]
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    sns.heatmap(
        attention_matrix,
        xticklabels=[f"T{i}" for i in sample_indices],
        yticklabels=edge_labels,
        cmap='YlOrRd',
        cbar_kws={'label': 'Attention Weight'},
        ax=ax,
        vmin=0,
        vmax=attention_matrix.max()
    )
    
    ax.set_title('Attention Weights Across Supply Chain Edges Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sample Index (Time)', fontsize=12)
    ax.set_ylabel('Supply Chain Edge', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_path}")


def create_entropy_timeseries(attention_df: pd.DataFrame, save_path: Path):
    """
    Plot attention entropy over time with volatility overlay.
    """
    print("Creating entropy time series...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot 1: Entropy over time
    ax1.plot(
        attention_df['sample_idx'],
        attention_df['attention_entropy'],
        color='steelblue',
        linewidth=1,
        alpha=0.7
    )
    
    # Add rolling mean
    window = 20
    rolling_entropy = attention_df['attention_entropy'].rolling(window=window).mean()
    ax1.plot(
        attention_df['sample_idx'],
        rolling_entropy,
        color='darkblue',
        linewidth=2,
        label=f'{window}-sample rolling mean'
    )
    
    ax1.set_ylabel('Attention Entropy', fontsize=11)
    ax1.set_title('Attention Entropy Over Time', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Volatility over time
    ax2.plot(
        attention_df['sample_idx'],
        attention_df['volatility_level'],
        color='coral',
        linewidth=1,
        alpha=0.7
    )
    
    rolling_vol = attention_df['volatility_level'].rolling(window=window).mean()
    ax2.plot(
        attention_df['sample_idx'],
        rolling_vol,
        color='darkred',
        linewidth=2,
        label=f'{window}-sample rolling mean'
    )
    
    ax2.set_xlabel('Sample Index (Time)', fontsize=11)
    ax2.set_ylabel('Volatility Level', fontsize=11)
    ax2.set_title('Actual Volatility Over Time', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_path}")


def create_entropy_volatility_scatter(attention_df: pd.DataFrame, save_path: Path):
    """
    Scatter plot showing relationship between entropy and future volatility.
    """
    print("Creating entropy-volatility correlation plot...")
    
    # Shift volatility forward to see if entropy predicts future volatility
    attention_df['future_volatility'] = attention_df['volatility_level'].shift(-5)
    
    # Remove NaN
    plot_df = attention_df.dropna()
    
    # Compute correlation
    corr, p_value = pearsonr(plot_df['attention_entropy'], plot_df['future_volatility'])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        plot_df['attention_entropy'],
        plot_df['future_volatility'],
        c=plot_df['sample_idx'],
        cmap='viridis',
        alpha=0.6,
        s=30
    )
    
    # Add regression line
    z = np.polyfit(plot_df['attention_entropy'], plot_df['future_volatility'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(plot_df['attention_entropy'].min(), plot_df['attention_entropy'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label='Linear fit')
    
    ax.set_xlabel('Attention Entropy', fontsize=12)
    ax.set_ylabel('Future Volatility (t+5)', fontsize=12)
    ax.set_title(
        f'Attention Entropy vs Future Volatility\nCorrelation: {corr:.3f} (p={p_value:.4f})',
        fontsize=13,
        fontweight='bold'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sample Index (Time)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_path}")
    
    return corr, p_value


def create_critical_pathways_plot(edge_stats_df: pd.DataFrame, save_path: Path):
    """
    Visualize most important supply chain pathways.
    """
    print("Creating critical pathways visualization...")
    
    # Get top 10 edges
    top_edges = edge_stats_df.nlargest(10, 'mean_attention')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Horizontal bar chart
    y_pos = np.arange(len(top_edges))
    
    bars = ax.barh(
        y_pos,
        top_edges['mean_attention'],
        xerr=top_edges['std_attention'],
        color='steelblue',
        alpha=0.7,
        edgecolor='darkblue'
    )
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_edges['edge'])
    ax.set_xlabel('Mean Attention Weight', fontsize=12)
    ax.set_title('Top 10 Critical Supply Chain Pathways', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_edges['mean_attention'])):
        ax.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_path}")


def create_attention_network_graph(edge_stats_df: pd.DataFrame, save_path: Path):
    """
    Create interactive network graph with edge thickness based on attention.
    """
    print("Creating attention network graph...")
    
    # Parse edges
    edges_data = []
    for _, row in edge_stats_df.iterrows():
        source, target = row['edge'].split('→')
        edges_data.append({
            'source': source,
            'target': target,
            'weight': row['mean_attention']
        })
    
    # Create NetworkX graph
    G = nx.DiGraph()
    for edge in edges_data:
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create edge traces
    edge_traces = []
    for edge in G.edges(data=True):
        source, target, data = edge
        weight = data['weight']
        
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        
        # Edge line
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=weight * 50, color='rgba(125,125,125,0.5)'),
            hoverinfo='text',
            text=f'{source}→{target}: {weight:.4f}',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Count in/out edges
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        node_text.append(f'{node}<br>In: {in_degree}, Out: {out_degree}')
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=30,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        ),
        text=[node for node in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo='text',
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        title='Supply Chain Attention Network (Edge Width = Attention Strength)',
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=1000,
        height=800
    )
    
    fig.write_html(str(save_path))
    
    print(f"✓ Saved: {save_path}")


def create_edge_attention_distribution(attention_df: pd.DataFrame, save_path: Path):
    """
    Box plots showing attention distribution for each edge.
    """
    print("Creating edge attention distribution plot...")
    
    # Extract attention columns
    attention_cols = [col for col in attention_df.columns if col.startswith('attn_')]
    
    # Prepare data for box plot
    edge_labels = [col.replace('attn_', '').replace('_', '→') for col in attention_cols]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Box plot
    bp = ax.boxplot(
        [attention_df[col].values for col in attention_cols],
        labels=edge_labels,
        patch_artist=True,
        showmeans=True
    )
    
    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Supply Chain Edge', fontsize=12)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title('Attention Distribution Across Supply Chain Edges', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_path}")


def main():
    """Main visualization execution"""
    print("="*70)
    print("PHASE 6: ATTENTION VISUALIZATION")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load data
    print("Loading attention data...")
    results_dir = Path("results/phase6_attention")
    attention_df, edge_stats_df = load_attention_data(results_dir)
    
    print(f"✓ Loaded {len(attention_df)} samples")
    print(f"✓ Loaded {len(edge_stats_df)} edges")
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    print("-"*70)
    
    # 1. Attention heatmap
    create_attention_heatmap(
        attention_df,
        results_dir / 'attention_heatmap.png'
    )
    
    # 2. Entropy time series
    create_entropy_timeseries(
        attention_df,
        results_dir / 'entropy_timeseries.png'
    )
    
    # 3. Entropy-volatility correlation
    corr, p_value = create_entropy_volatility_scatter(
        attention_df,
        results_dir / 'entropy_volatility_correlation.png'
    )
    
    # 4. Critical pathways
    create_critical_pathways_plot(
        edge_stats_df,
        results_dir / 'critical_pathways.png'
    )
    
    # 5. Attention network graph
    create_attention_network_graph(
        edge_stats_df,
        results_dir / 'attention_network.html'
    )
    
    # 6. Edge attention distribution
    create_edge_attention_distribution(
        attention_df,
        results_dir / 'edge_attention_distribution.png'
    )
    
    print("-"*70)
    print()
    
    # Summary
    print("="*70)
    print("VISUALIZATION SUMMARY")
    print("="*70)
    print()
    
    print(f"Entropy-Volatility Correlation: {corr:.4f} (p={p_value:.4f})")
    if p_value < 0.05:
        print("  → SIGNIFICANT: Entropy predicts future volatility ✓")
    else:
        print("  → Not significant: Weak predictive power")
    print()
    
    print("Top 5 Critical Pathways:")
    for i, row in edge_stats_df.nlargest(5, 'mean_attention').iterrows():
        print(f"  {i+1}. {row['edge']}: {row['mean_attention']:.4f}")
    print()
    
    print("Generated Visualizations:")
    print(f"  1. {results_dir / 'attention_heatmap.png'}")
    print(f"  2. {results_dir / 'entropy_timeseries.png'}")
    print(f"  3. {results_dir / 'entropy_volatility_correlation.png'}")
    print(f"  4. {results_dir / 'critical_pathways.png'}")
    print(f"  5. {results_dir / 'attention_network.html'}")
    print(f"  6. {results_dir / 'edge_attention_distribution.png'}")
    print()
    
    print("="*70)
    print("✓ PHASE 6 VISUALIZATION COMPLETE!")
    print("="*70)
    print()
    print("Next: Review visualizations and proceed to Phase 7 (Hypothesis Testing)")
    print()


if __name__ == "__main__":
    main()
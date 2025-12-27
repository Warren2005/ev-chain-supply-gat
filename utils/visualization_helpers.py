"""
Helper functions for visualization creation

Provides utility functions for:
- Color schemes
- Layout templates
- Data transformations
- Plot styling

Author: EV Supply Chain GAT Team
Date: December 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import plotly.graph_objects as go


def get_stock_tier_mapping() -> Dict[str, str]:
    """
    Get supply chain tier for each stock.
    
    Returns:
        Dictionary mapping stock ticker to tier
    """
    return {
        'ALB': 'Raw Materials',
        'SQM': 'Raw Materials',
        'APTV': 'Components',
        'MGA': 'Components',
        'F': 'OEMs',
        'GM': 'OEMs',
        'TSLA': 'OEMs'
    }


def get_tier_colors() -> Dict[str, str]:
    """
    Get color scheme for supply chain tiers.
    
    Returns:
        Dictionary mapping tier to color
    """
    return {
        'Raw Materials': '#2E7D32',  # Green
        'Components': '#1976D2',     # Blue
        'OEMs': '#C62828'            # Red
    }


def get_stock_colors() -> Dict[str, str]:
    """
    Get individual colors for each stock.
    
    Returns:
        Dictionary mapping stock ticker to color
    """
    return {
        'ALB': '#66BB6A',
        'SQM': '#81C784',
        'APTV': '#42A5F5',
        'MGA': '#64B5F6',
        'F': '#EF5350',
        'GM': '#E57373',
        'TSLA': '#FF5252'
    }


def create_plotly_template() -> go.layout.Template:
    """
    Create consistent Plotly template for all visualizations.
    
    Returns:
        Plotly template with styling
    """
    template = go.layout.Template()
    
    template.layout = go.Layout(
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return template


def format_metric_name(metric: str) -> str:
    """
    Format metric names for display.
    
    Args:
        metric: Metric key (e.g., 'r2', 'rmse')
    
    Returns:
        Formatted string (e.g., 'R² Score', 'RMSE')
    """
    mapping = {
        'r2': 'R² Score',
        'rmse': 'RMSE',
        'mae': 'MAE',
        'mse': 'MSE',
        'mape': 'MAPE',
        'directional_accuracy': 'Directional Accuracy',
        'correlation': 'Correlation'
    }
    
    return mapping.get(metric, metric.replace('_', ' ').title())


def calculate_prediction_errors(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Calculate various error metrics.
    
    Args:
        predictions: Predicted values [num_samples, num_stocks]
        targets: Actual values [num_samples, num_stocks]
    
    Returns:
        Dictionary of error metrics
    """
    errors = predictions - targets
    
    return {
        'absolute_error': np.abs(errors),
        'squared_error': errors ** 2,
        'relative_error': errors / (np.abs(targets) + 1e-8),
        'directional_correct': (np.sign(predictions) == np.sign(targets)).astype(float)
    }


def create_summary_statistics_table(
    predictions: np.ndarray,
    targets: np.ndarray,
    stock_names: List[str]
) -> pd.DataFrame:
    """
    Create summary statistics table for visualization.
    
    Args:
        predictions: Predicted values
        targets: Actual values
        stock_names: List of stock tickers
    
    Returns:
        DataFrame with summary statistics
    """
    stats = []
    
    for i, stock in enumerate(stock_names):
        preds = predictions[:, i]
        acts = targets[:, i]
        
        stats.append({
            'Stock': stock,
            'Pred Mean': preds.mean(),
            'Pred Std': preds.std(),
            'Actual Mean': acts.mean(),
            'Actual Std': acts.std(),
            'Mean Error': (preds - acts).mean(),
            'RMSE': np.sqrt(((preds - acts) ** 2).mean()),
            'Correlation': np.corrcoef(preds, acts)[0, 1]
        })
    
    return pd.DataFrame(stats)


def add_3d_grid(fig: go.Figure, x_range: Tuple, y_range: Tuple, z_range: Tuple):
    """
    Add subtle 3D grid to figure.
    
    Args:
        fig: Plotly figure
        x_range: (min, max) for x-axis
        y_range: (min, max) for y-axis
        z_range: (min, max) for z-axis
    """
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                range=x_range,
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1
            ),
            yaxis=dict(
                range=y_range,
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1
            ),
            zaxis=dict(
                range=z_range,
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1
            )
        )
    )


def create_legend_html(tier_colors: Dict[str, str]) -> str:
    """
    Create HTML legend for static exports.
    
    Args:
        tier_colors: Dictionary mapping tier to color
    
    Returns:
        HTML string for legend
    """
    html = '<div style="font-family: Arial; padding: 10px;">\n'
    html += '<h3>Supply Chain Tiers</h3>\n'
    html += '<ul style="list-style-type: none; padding: 0;">\n'
    
    for tier, color in tier_colors.items():
        html += f'  <li><span style="color: {color}; font-size: 20px;">●</span> {tier}</li>\n'
    
    html += '</ul>\n</div>'
    
    return html
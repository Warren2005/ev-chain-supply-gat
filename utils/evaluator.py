"""
Model Evaluation Suite for ST-GAT

Comprehensive evaluation including:
- Test set metrics
- Per-stock performance
- Temporal error analysis
- 3D embedding visualizations
- Attention weight extraction

Author: EV Supply Chain GAT Team
Date: December 2024
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


class STGATEvaluator:
    """
    Comprehensive evaluator for ST-GAT model.
    
    Provides:
    - Test set evaluation
    - Detailed metrics computation
    - Per-stock and temporal analysis
    - 3D visualization of learned representations
    - Attention weight extraction and analysis
    
    Attributes:
        model (nn.Module): Trained ST-GAT model
        test_loader (DataLoader): Test data loader
        device (str): Device for evaluation
        stock_names (List[str]): List of stock tickers
        logger (logging.Logger): Logger instance
    """
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        stock_names: List[str],
        device: str = "cpu"
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained ST-GAT model
            test_loader: DataLoader for test data
            stock_names: List of stock ticker names
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.stock_names = stock_names
        self.device = device
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Storage for results
        self.predictions = None
        self.targets = None
        self.features_embedded = None
        self.attention_weights = None
        
        self.logger.info(f"Evaluator initialized for {len(stock_names)} stocks")
    
    def _setup_logger(self) -> logging.Logger:
        """Configure logging."""
        logger = logging.getLogger(f"{__name__}.STGATEvaluator")
        logger.setLevel(logging.INFO)
        
        if logger.handlers:
            logger.handlers.clear()
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info("Starting test set evaluation...")
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, edge_index, targets in self.test_loader:
                # Move to device
                features = features.to(self.device)
                edge_index = edge_index.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(features, edge_index)
                predictions = predictions.squeeze(-1)
                
                # Store results
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate all batches
        self.predictions = torch.cat(all_predictions, dim=0)
        self.targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        metrics = self._compute_metrics(self.predictions, self.targets)
        
        self.logger.info("Test set evaluation complete")
        self._print_metrics(metrics)
        
        return metrics
    
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            predictions: Model predictions [num_samples, num_nodes]
            targets: Ground truth targets [num_samples, num_nodes]
        
        Returns:
            Dictionary with metrics
        """
        # MSE
        mse = torch.mean((predictions - targets) ** 2).item()
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # MAE
        mae = torch.mean(torch.abs(predictions - targets)).item()
        
        # MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        epsilon = 1e-8
        mape = torch.mean(
            torch.abs((targets - predictions) / (targets + epsilon))
        ).item() * 100
        
        # R² score
        ss_res = torch.sum((targets - predictions) ** 2).item()
        ss_tot = torch.sum((targets - targets.mean()) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Directional accuracy (did we predict the direction of change correctly?)
        # Compare sign of predictions vs targets
        correct_direction = (
            torch.sign(predictions) == torch.sign(targets)
        ).float().mean().item()
        
        # Correlation
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        correlation = torch.corrcoef(
            torch.stack([pred_flat, target_flat])
        )[0, 1].item()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'directional_accuracy': correct_direction,
            'correlation': correlation
        }
    
    def _print_metrics(self, metrics: Dict[str, float]) -> None:
        """Print metrics in formatted table."""
        print("\n" + "="*70)
        print("TEST SET EVALUATION METRICS")
        print("="*70)
        print(f"MSE:                    {metrics['mse']:.6f}")
        print(f"RMSE:                   {metrics['rmse']:.6f}")
        print(f"MAE:                    {metrics['mae']:.6f}")
        print(f"MAPE:                   {metrics['mape']:.2f}%")
        print(f"R² Score:               {metrics['r2']:.4f}")
        print(f"Directional Accuracy:   {metrics['directional_accuracy']*100:.2f}%")
        print(f"Correlation:            {metrics['correlation']:.4f}")
        print("="*70 + "\n")
    
    def compute_per_stock_metrics(self) -> pd.DataFrame:
        """
        Compute metrics for each stock individually.
        
        Returns:
            DataFrame with per-stock metrics
        """
        if self.predictions is None:
            raise ValueError("Run evaluate() first")
        
        self.logger.info("Computing per-stock metrics...")
        
        stock_metrics = []
        
        for stock_idx, stock_name in enumerate(self.stock_names):
            # Extract predictions and targets for this stock
            stock_preds = self.predictions[:, stock_idx]
            stock_targets = self.targets[:, stock_idx]
            
            # Compute metrics
            metrics = self._compute_metrics(
                stock_preds.unsqueeze(-1),
                stock_targets.unsqueeze(-1)
            )
            
            stock_metrics.append({
                'stock': stock_name,
                'mse': metrics['mse'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2'],
                'directional_accuracy': metrics['directional_accuracy'],
                'correlation': metrics['correlation']
            })
        
        df = pd.DataFrame(stock_metrics)
        
        self.logger.info("Per-stock metrics computed")
        
        return df
    
    def extract_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract learned embeddings from GAT layers for visualization.
        
        This extracts the internal representations learned by the model
        which can be visualized in 3D space using dimensionality reduction.
        
        Returns:
            Tuple of (gat_embeddings, lstm_embeddings)
                - gat_embeddings: [num_samples, num_nodes, gat_hidden_dim]
                - lstm_embeddings: [num_samples, num_nodes, lstm_hidden_dim]
        """
        self.logger.info("Extracting learned embeddings...")
        
        gat_embeddings = []
        lstm_embeddings = []
        
        with torch.no_grad():
            for features, edge_index, _ in self.test_loader:
                features = features.to(self.device)
                edge_index = edge_index.to(self.device)
                
                batch_size, num_nodes, seq_len, feat_dim = features.shape
                
                # Extract GAT embeddings (after GAT processing)
                gat_outputs = []
                
                for t in range(seq_len):
                    x_t = features[:, :, t, :]
                    x_t_flat = x_t.reshape(batch_size * num_nodes, feat_dim)
                    
                    # Pass through GAT layers
                    h = x_t_flat
                    for gat_layer in self.model.gat_layers_list:
                        h = gat_layer(h, edge_index)
                    
                    h = h.reshape(batch_size, num_nodes, self.model.gat_hidden_dim)
                    gat_outputs.append(h)
                
                # Take last timestep's GAT output
                gat_output_last = gat_outputs[-1].cpu().numpy()
                gat_embeddings.append(gat_output_last)
                
                # Extract LSTM embeddings
                gat_output_seq = torch.stack(gat_outputs, dim=2)
                lstm_output, _ = self.model.lstm(gat_output_seq)
                lstm_embeddings.append(lstm_output.cpu().numpy())
        
        # Concatenate all batches
        gat_embeddings = np.concatenate(gat_embeddings, axis=0)
        lstm_embeddings = np.concatenate(lstm_embeddings, axis=0)
        
        self.logger.info(
            f"Extracted embeddings: GAT {gat_embeddings.shape}, "
            f"LSTM {lstm_embeddings.shape}"
        )
        
        return gat_embeddings, lstm_embeddings
    
    def visualize_embeddings_3d(
        self,
        save_dir: Path,
        method: str = 'tsne'
    ) -> None:
        """
        Create 3D visualization of learned embeddings.
        
        Uses t-SNE or PCA to reduce high-dimensional embeddings to 3D
        for visualization. This shows how the model clusters similar
        stocks/patterns in representation space.
        
        Args:
            save_dir: Directory to save visualizations
            method: Dimensionality reduction method ('tsne' or 'pca')
        """
        self.logger.info(f"Creating 3D embedding visualization ({method})...")
        
        # Extract embeddings
        gat_embeddings, lstm_embeddings = self.extract_embeddings()
        
        # Use LSTM embeddings for visualization (final learned representations)
        # Reshape: [num_samples, num_nodes, hidden_dim] -> [num_samples*num_nodes, hidden_dim]
        num_samples, num_nodes, hidden_dim = lstm_embeddings.shape
        embeddings_flat = lstm_embeddings.reshape(-1, hidden_dim)
        
        # Create labels for coloring
        stock_labels = np.repeat(self.stock_names, num_samples)
        sample_indices = np.tile(np.arange(num_samples), num_nodes)
        
        # Reduce to 3D
        if method == 'tsne':
            reducer = TSNE(n_components=3, random_state=42, perplexity=30)
        else:  # pca
            reducer = PCA(n_components=3, random_state=42)
        
        embeddings_3d = reducer.fit_transform(embeddings_flat)
        
        # Create interactive 3D plot with Plotly
        fig = go.Figure()
        
        # Add trace for each stock
        for stock_name in self.stock_names:
            mask = stock_labels == stock_name
            
            fig.add_trace(go.Scatter3d(
                x=embeddings_3d[mask, 0],
                y=embeddings_3d[mask, 1],
                z=embeddings_3d[mask, 2],
                mode='markers',
                name=stock_name,
                marker=dict(
                    size=3,
                    opacity=0.6
                ),
                text=[f"{stock_name}<br>Sample {i}" for i in sample_indices[mask]],
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title=f'3D {method.upper()} Visualization of Learned Embeddings',
            scene=dict(
                xaxis_title=f'{method.upper()} Component 1',
                yaxis_title=f'{method.upper()} Component 2',
                zaxis_title=f'{method.upper()} Component 3',
            ),
            width=1000,
            height=800,
            showlegend=True
        )
        
        # Save
        save_path = save_dir / f'embeddings_3d_{method}.html'
        fig.write_html(str(save_path))
        
        self.logger.info(f"3D embedding visualization saved to {save_path}")
    
    def plot_prediction_vs_actual_3d(self, save_dir: Path) -> None:
        """
        Create 3D scatter plot of predictions vs actual values over time.
        
        3D axes: Actual Value, Predicted Value, Time
        This shows model performance evolution across the test period.
        
        Args:
            save_dir: Directory to save visualization
        """
        self.logger.info("Creating 3D prediction vs actual plot...")
        
        if self.predictions is None:
            raise ValueError("Run evaluate() first")
        
        # Create figure
        fig = go.Figure()
        
        # Add trace for each stock
        num_samples = self.predictions.shape[0]
        time_indices = np.arange(num_samples)
        
        for stock_idx, stock_name in enumerate(self.stock_names):
            preds = self.predictions[:, stock_idx].numpy()
            actuals = self.targets[:, stock_idx].numpy()
            
            # Calculate errors for coloring
            errors = np.abs(preds - actuals)
            
            fig.add_trace(go.Scatter3d(
                x=actuals,
                y=preds,
                z=time_indices,
                mode='markers',
                name=stock_name,
                marker=dict(
                    size=3,
                    color=errors,
                    colorscale='Viridis',
                    showscale=(stock_idx == 0),  # Show colorbar only once
                    colorbar=dict(title="Absolute Error"),
                    opacity=0.6
                ),
                text=[
                    f"{stock_name}<br>"
                    f"Actual: {a:.4f}<br>"
                    f"Pred: {p:.4f}<br>"
                    f"Error: {e:.4f}<br>"
                    f"Time: {t}"
                    for a, p, e, t in zip(actuals, preds, errors, time_indices)
                ],
                hoverinfo='text'
            ))
        
        # Add perfect prediction line (y=x)
        min_val = min(self.targets.min().item(), self.predictions.min().item())
        max_val = max(self.targets.max().item(), self.predictions.max().item())
        
        fig.add_trace(go.Scatter3d(
            x=[min_val, max_val],
            y=[min_val, max_val],
            z=[0, num_samples],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=True
        ))
        
        fig.update_layout(
            title='3D Prediction vs Actual Values Over Time',
            scene=dict(
                xaxis_title='Actual Volatility',
                yaxis_title='Predicted Volatility',
                zaxis_title='Time (Test Sample Index)',
            ),
            width=1000,
            height=800,
            showlegend=True
        )
        
        # Save
        save_path = save_dir / 'predictions_3d.html'
        fig.write_html(str(save_path))
        
        self.logger.info(f"3D prediction plot saved to {save_path}")
    
    def save_results(
        self,
        save_dir: Path,
        metrics: Dict[str, float],
        per_stock_df: pd.DataFrame
    ) -> None:
        """
        Save evaluation results to files.
        
        Args:
            save_dir: Directory to save results
            metrics: Overall metrics dictionary
            per_stock_df: Per-stock metrics DataFrame
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save overall metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(save_dir / 'test_metrics.csv', index=False)
        
        # Save per-stock metrics
        per_stock_df.to_csv(save_dir / 'per_stock_metrics.csv', index=False)
        
        # Save predictions and targets
        results_df = pd.DataFrame({
            f'pred_{stock}': self.predictions[:, i].numpy()
            for i, stock in enumerate(self.stock_names)
        })
        
        for i, stock in enumerate(self.stock_names):
            results_df[f'actual_{stock}'] = self.targets[:, i].numpy()
        
        results_df.to_csv(save_dir / 'predictions_detailed.csv', index=False)
        
        self.logger.info(f"Results saved to {save_dir}")
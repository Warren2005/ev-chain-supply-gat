"""
Advanced 3D Visualizations for ST-GAT Model

Creates publication-quality 3D visualizations including:
- Interactive supply chain graph with attention weights
- 3D loss landscape with training trajectory
- Temporal evolution of predictions
- Stock clustering by learned embeddings
- Attention flow through supply chain

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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import networkx as nx


class AdvancedVisualizer:
    """
    Advanced 3D visualization suite for ST-GAT analysis.
    
    Creates interactive 3D visualizations to understand:
    - Model behavior across stocks
    - Supply chain attention dynamics
    - Temporal prediction patterns
    - Training optimization landscape
    """
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        stock_names: List[str],
        edge_index: torch.Tensor,
        device: str = "cpu"
    ):
        """
        Initialize visualizer.
        
        Args:
            model: Trained ST-GAT model
            test_loader: Test data loader
            stock_names: List of stock tickers
            edge_index: Graph edge indices [2, num_edges]
            device: Device for computation
        """
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.stock_names = stock_names
        self.edge_index = edge_index
        self.device = device
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Storage
        self.predictions = None
        self.targets = None
        self.embeddings = None
        
        self.logger.info("AdvancedVisualizer initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger(f"{__name__}.AdvancedVisualizer")
        logger.setLevel(logging.INFO)
        
        if logger.handlers:
            logger.handlers.clear()
        
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def extract_predictions_and_embeddings(self):
        """Extract predictions and internal embeddings from model"""
        self.logger.info("Extracting predictions and embeddings...")
        
        all_predictions = []
        all_targets = []
        all_embeddings = []
        
        with torch.no_grad():
            for features, edge_index, targets in self.test_loader:
                features = features.to(self.device)
                edge_index = edge_index.to(self.device)
                
                # Get predictions
                predictions = self.model(features, edge_index)
                predictions = predictions.squeeze(-1)
                
                # Store
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                
                # Extract embeddings (last LSTM hidden state)
                # This requires accessing intermediate layers
                batch_size, num_nodes, seq_len, feat_dim = features.shape
                
                # Process through GAT
                gat_outputs = []
                for t in range(seq_len):
                    x_t = features[:, :, t, :]
                    x_t_flat = x_t.reshape(batch_size * num_nodes, feat_dim)
                    h_t = self.model.gat(x_t_flat, edge_index)
                    h_t = h_t.reshape(batch_size, num_nodes, -1)
                    gat_outputs.append(h_t)
                
                gat_output = torch.stack(gat_outputs, dim=2)
                gat_output = gat_output.permute(1, 0, 2, 3)
                
                # LSTM embeddings (final hidden state)
                node_embeddings = []
                for node_idx in range(num_nodes):
                    node_seq = gat_output[node_idx].transpose(0, 1)
                    lstm_out, (h_n, c_n) = self.model.lstm(node_seq)
                    node_embeddings.append(h_n[-1])  # Last layer hidden state
                
                embeddings = torch.stack(node_embeddings, dim=1)
                all_embeddings.append(embeddings.cpu())
        
        # Concatenate all batches
        self.predictions = torch.cat(all_predictions, dim=0)
        self.targets = torch.cat(all_targets, dim=0)
        self.embeddings = torch.cat(all_embeddings, dim=0)
        
        self.logger.info(
            f"Extracted: predictions {self.predictions.shape}, "
            f"embeddings {self.embeddings.shape}"
        )
    
    def create_3d_supply_chain_graph(self, save_path: Path):
        """
        Create interactive 3D supply chain graph with nodes positioned in 3D space.
        
        Uses force-directed layout in 3D to visualize supply chain structure.
        Node colors represent tier (raw materials, components, OEMs).
        Edge thickness represents attention importance (future work).
        """
        self.logger.info("Creating 3D supply chain graph...")
        
        # Define stock tiers for coloring
        stock_tiers = {
            'ALB': 'Raw Materials',
            'SQM': 'Raw Materials',
            'APTV': 'Components',
            'MGA': 'Components',
            'F': 'OEMs',
            'GM': 'OEMs',
            'TSLA': 'OEMs'
        }
        
        tier_colors = {
            'Raw Materials': '#2E7D32',  # Green
            'Components': '#1976D2',     # Blue
            'OEMs': '#C62828'            # Red
        }
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        for idx, stock in enumerate(self.stock_names):
            G.add_node(idx, name=stock, tier=stock_tiers[stock])
        
        # Add edges
        edge_index_np = self.edge_index.cpu().numpy()
        for i in range(edge_index_np.shape[1]):
            source = edge_index_np[0, i]
            target = edge_index_np[1, i]
            G.add_edge(source, target)
        
        # 3D spring layout
        pos_3d = nx.spring_layout(G, dim=3, k=1.5, iterations=50)
        
        # Extract positions
        x_nodes = [pos_3d[i][0] for i in range(len(self.stock_names))]
        y_nodes = [pos_3d[i][1] for i in range(len(self.stock_names))]
        z_nodes = [pos_3d[i][2] for i in range(len(self.stock_names))]
        
        # Create edges for plotting
        edge_x = []
        edge_y = []
        edge_z = []
        
        for edge in G.edges():
            x0, y0, z0 = pos_3d[edge[0]]
            x1, y1, z1 = pos_3d[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_z += [z0, z1, None]
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='#888', width=2),
            hoverinfo='none',
            name='Supply Chain Links'
        ))
        
        # Add nodes by tier
        for tier, color in tier_colors.items():
            tier_indices = [i for i, s in enumerate(self.stock_names) 
                           if stock_tiers[s] == tier]
            
            fig.add_trace(go.Scatter3d(
                x=[x_nodes[i] for i in tier_indices],
                y=[y_nodes[i] for i in tier_indices],
                z=[z_nodes[i] for i in tier_indices],
                mode='markers+text',
                marker=dict(size=20, color=color, line=dict(color='white', width=2)),
                text=[self.stock_names[i] for i in tier_indices],
                textposition="top center",
                textfont=dict(size=12, color='black'),
                name=tier,
                hovertext=[
                    f"{self.stock_names[i]}<br>Tier: {tier}<br>Node {i}"
                    for i in tier_indices
                ],
                hoverinfo='text'
            ))
        
        # Update layout
        fig.update_layout(
            title='3D EV Supply Chain Graph Structure',
            scene=dict(
                xaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
                zaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
                bgcolor='white'
            ),
            showlegend=True,
            width=1000,
            height=800,
            hovermode='closest'
        )
        
        # Save
        fig.write_html(str(save_path))
        self.logger.info(f"3D supply chain graph saved: {save_path}")
    
    def create_3d_temporal_evolution(self, save_path: Path, num_samples: int = 50):
        """
        Create 3D visualization showing how predictions evolve over time.
        
        3D axes: Time, Actual Value, Predicted Value
        Shows trajectory of predictions vs actuals for each stock.
        """
        self.logger.info("Creating 3D temporal evolution...")
        
        if self.predictions is None:
            self.extract_predictions_and_embeddings()
        
        # Limit samples for clarity
        n_samples = min(num_samples, self.predictions.shape[0])
        
        fig = go.Figure()
        
        # Add trace for each stock
        for stock_idx, stock in enumerate(self.stock_names):
            preds = self.predictions[:n_samples, stock_idx].numpy()
            actuals = self.targets[:n_samples, stock_idx].numpy()
            time_indices = np.arange(n_samples)
            
            # Actual values trajectory
            fig.add_trace(go.Scatter3d(
                x=time_indices,
                y=actuals,
                z=preds,
                mode='markers+lines',
                name=f'{stock}',
                marker=dict(size=4),
                line=dict(width=2),
                text=[f"{stock}<br>Time: {t}<br>Actual: {a:.3f}<br>Pred: {p:.3f}"
                      for t, a, p in zip(time_indices, actuals, preds)],
                hoverinfo='text'
            ))
        
        # Add perfect prediction plane (z = y)
        y_range = [self.targets[:n_samples].min().item(), 
                   self.targets[:n_samples].max().item()]
        
        fig.add_trace(go.Surface(
            x=[[0, n_samples], [0, n_samples]],
            y=[[y_range[0], y_range[0]], [y_range[1], y_range[1]]],
            z=[[y_range[0], y_range[1]], [y_range[0], y_range[1]]],
            colorscale=[[0, 'rgba(255,0,0,0.1)'], [1, 'rgba(255,0,0,0.1)']],
            showscale=False,
            name='Perfect Prediction',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title='3D Temporal Evolution: Predictions vs Actuals Over Time',
            scene=dict(
                xaxis_title='Time (Test Sample Index)',
                yaxis_title='Actual Volatility',
                zaxis_title='Predicted Volatility',
            ),
            width=1200,
            height=800,
            showlegend=True
        )
        
        fig.write_html(str(save_path))
        self.logger.info(f"3D temporal evolution saved: {save_path}")
    
    def create_3d_embedding_clusters(self, save_path: Path, method: str = 'tsne'):
        """
        Create 3D visualization of learned stock embeddings.
        
        Shows how the model internally represents different stocks
        in a 3D embedding space after dimensionality reduction.
        """
        self.logger.info(f"Creating 3D embedding clusters ({method})...")
        
        if self.embeddings is None:
            self.extract_predictions_and_embeddings()
        
        # Reshape embeddings: [num_samples, num_nodes, hidden_dim] -> [num_samples*num_nodes, hidden_dim]
        num_samples, num_nodes, hidden_dim = self.embeddings.shape
        embeddings_flat = self.embeddings.reshape(-1, hidden_dim).numpy()
        
        # Create labels
        stock_labels = np.repeat(self.stock_names, num_samples)
        time_labels = np.tile(np.arange(num_samples), num_nodes)
        
        # Reduce to 3D
        if method == 'tsne':
            reducer = TSNE(n_components=3, random_state=42, perplexity=30)
        else:  # pca
            reducer = PCA(n_components=3, random_state=42)
        
        embeddings_3d = reducer.fit_transform(embeddings_flat)
        
        # Create figure
        fig = go.Figure()
        
        # Add trace for each stock
        for stock in self.stock_names:
            mask = stock_labels == stock
            
            fig.add_trace(go.Scatter3d(
                x=embeddings_3d[mask, 0],
                y=embeddings_3d[mask, 1],
                z=embeddings_3d[mask, 2],
                mode='markers',
                name=stock,
                marker=dict(size=3, opacity=0.6),
                text=[f"{stock}<br>Sample {t}" for t in time_labels[mask]],
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title=f'3D Stock Embeddings ({method.upper()}) - Learned Representations',
            scene=dict(
                xaxis_title=f'{method.upper()} Component 1',
                yaxis_title=f'{method.upper()} Component 2',
                zaxis_title=f'{method.upper()} Component 3',
            ),
            width=1000,
            height=800,
            showlegend=True
        )
        
        fig.write_html(str(save_path))
        self.logger.info(f"3D embedding clusters saved: {save_path}")
    
    def create_3d_loss_landscape(
    self,
    train_history: dict,
    save_path: Path
):
        """
        Create 3D loss landscape showing training trajectory.
        
        Visualizes how the model navigated the loss landscape during training.
        """
        self.logger.info("Creating 3D loss landscape...")
        
        epochs = np.arange(len(train_history['train_losses']))
        train_losses = np.array(train_history['train_losses'])
        val_losses = np.array(train_history['val_losses'])
        
        # Create 3D surface plot
        fig = go.Figure()
        
        # Training trajectory
        fig.add_trace(go.Scatter3d(
            x=epochs,
            y=train_losses,
            z=val_losses,
            mode='markers+lines',
            name='Training Trajectory',
            marker=dict(
                size=4,
                color=epochs,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Epoch")
            ),
            line=dict(width=3, color='blue'),
            text=[f"Epoch {e}<br>Train: {t:.4f}<br>Val: {v:.4f}"
                for e, t, v in zip(epochs, train_losses, val_losses)],
            hoverinfo='text'
        ))
        
        # Mark start
        fig.add_trace(go.Scatter3d(
            x=[epochs[0]], y=[train_losses[0]], z=[val_losses[0]],
            mode='markers',
            marker=dict(size=15, color='green', symbol='diamond'),  # Changed from 'diamond' - this is OK
            name='Start',
            showlegend=True
        ))
        
        # Mark best
        best_epoch = np.argmin(val_losses)
        fig.add_trace(go.Scatter3d(
            x=[epochs[best_epoch]], y=[train_losses[best_epoch]], z=[val_losses[best_epoch]],
            mode='markers',
            marker=dict(size=15, color='gold', symbol='diamond-open'),  # Changed from 'star' to 'diamond-open'
            name='Best Model',
            showlegend=True
        ))
        
        # Mark end
        fig.add_trace(go.Scatter3d(
            x=[epochs[-1]], y=[train_losses[-1]], z=[val_losses[-1]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='square'),  # Changed from 'square' - this is OK
            name='Final',
            showlegend=True
        ))
        
        fig.update_layout(
            title='3D Loss Landscape: Training Optimization Trajectory',
            scene=dict(
                xaxis_title='Epoch',
                yaxis_title='Training Loss',
                zaxis_title='Validation Loss',
            ),
            width=1000,
            height=800
        )
        
        fig.write_html(str(save_path))
        self.logger.info(f"3D loss landscape saved: {save_path}")
    
    def create_performance_3d_scatter(self, save_path: Path):
        """
        Create 3D scatter showing R², directional accuracy, and correlation.
        
        Each stock is a point in 3D space defined by its performance metrics.
        """
        self.logger.info("Creating 3D performance scatter...")
        
        # Load per-stock metrics
        metrics_df = pd.read_csv('results/final_evaluation/per_stock_metrics.csv')
        
        fig = go.Figure()
        
        # Define stock tiers
        stock_tiers = {
            'ALB': 'Raw Materials', 'SQM': 'Raw Materials',
            'APTV': 'Components', 'MGA': 'Components',
            'F': 'OEMs', 'GM': 'OEMs', 'TSLA': 'OEMs'
        }
        
        tier_colors = {
            'Raw Materials': 'green',
            'Components': 'blue',
            'OEMs': 'red'
        }
        
        # Add each stock
        for _, row in metrics_df.iterrows():
            stock = row['stock']
            tier = stock_tiers[stock]
            
            fig.add_trace(go.Scatter3d(
                x=[row['r2']],
                y=[row['directional_accuracy']],
                z=[row['correlation']],
                mode='markers+text',
                marker=dict(size=15, color=tier_colors[tier]),
                text=[stock],
                textposition="top center",
                name=stock,
                hovertext=f"{stock}<br>Tier: {tier}<br>R²: {row['r2']:.3f}<br>Dir Acc: {row['directional_accuracy']*100:.1f}%<br>Corr: {row['correlation']:.3f}",
                hoverinfo='text'
            ))
        
        # Add reference planes
        # R² = 0 plane (baseline)
        fig.add_trace(go.Surface(
            x=[[0, 0], [0, 0]],
            y=[[0, 1], [0, 1]],
            z=[[-1, -1], [1, 1]],
            colorscale=[[0, 'rgba(200,200,200,0.3)'], [1, 'rgba(200,200,200,0.3)']],
            showscale=False,
            name='R²=0 (Baseline)',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title='3D Performance Space: Stock-Level Metrics',
            scene=dict(
                xaxis_title='R² Score',
                yaxis_title='Directional Accuracy',
                zaxis_title='Correlation',
            ),
            width=1000,
            height=800,
            showlegend=True
        )
        
        fig.write_html(str(save_path))
        self.logger.info(f"3D performance scatter saved: {save_path}")
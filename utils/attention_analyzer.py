"""
Attention Analysis for ST-GAT Interpretability

Extracts and analyzes attention weights from trained GAT layers to:
1. Identify critical supply chain pathways
2. Compute attention entropy as early warning signal
3. Visualize dynamic attention flow
4. Test research hypotheses about attention patterns

Author: EV Supply Chain GAT Team
Date: December 2024
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx


class AttentionAnalyzer:
    """
    Comprehensive attention analysis for ST-GAT model.
    
    Extracts attention weights from GAT layers and provides:
    - Attention weight extraction per timestep
    - Entropy calculation over time
    - Critical pathway identification
    - Temporal dynamics analysis
    - Volatility event correlation
    """
    
    def __init__(
        self,
        model: nn.Module,
        edge_index: torch.Tensor,
        stock_names: List[str],
        device: str = "cpu"
    ):
        """
        Initialize attention analyzer.
        
        Args:
            model: Trained ST-GAT model
            edge_index: Graph edges [2, num_edges]
            stock_names: List of stock tickers
            device: Device for computation
        """
        self.model = model.to(device)
        self.model.eval()
        self.edge_index = edge_index
        self.stock_names = stock_names
        self.device = device
        
        # Storage
        self.attention_weights = []  # List of [num_edges] per sample
        self.attention_entropy = []  # List of scalars per sample
        self.timestamps = []
        self.volatility_levels = []
        
        # Setup logging
        self.logger = self._setup_logger()
        
        self.logger.info("AttentionAnalyzer initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger(f"{__name__}.AttentionAnalyzer")
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
    
    def extract_attention_weights_from_forward(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Modified forward pass that returns attention weights.
        
        This requires instrumenting the GAT layer to capture attention.
        
        Args:
            features: Input features [batch, nodes, seq_len, features]
        
        Returns:
            Tuple of (predictions, attention_weights_per_timestep)
        """
        batch_size, num_nodes, seq_len, feat_dim = features.shape
        
        attention_per_timestep = []
        
        # Hook to capture attention from GAT
        attention_hook = {}
        
        def get_attention_hook(name):
            def hook(module, input, output):
                # Capture attention coefficients
                # This assumes GAT layer stores attention in self.attention
                if hasattr(module, 'attention_weights'):
                    attention_hook[name] = module.attention_weights.detach()
            return hook
        
        # Register hook on GAT layer
        # Note: This requires modifying GATLayer to store attention weights
        hook_handle = self.model.gat.register_forward_hook(
            get_attention_hook('gat')
        )
        
        # Forward pass through GAT for each timestep
        gat_outputs = []
        for t in range(seq_len):
            x_t = features[:, :, t, :]
            x_t_flat = x_t.reshape(batch_size * num_nodes, feat_dim)
            
            # Clear previous attention
            attention_hook.clear()
            
            # Forward through GAT
            h_t = self.model.gat(x_t_flat, self.edge_index)
            
            # Capture attention if available
            if 'gat' in attention_hook:
                attention_per_timestep.append(attention_hook['gat'])
            
            h_t = self.model.dropout_layer(h_t)
            h_t = self.model.bn_gat(h_t)
            h_t = h_t.reshape(batch_size, num_nodes, -1)
            gat_outputs.append(h_t)
        
        # Remove hook
        hook_handle.remove()
        
        # Continue with LSTM...
        gat_output = torch.stack(gat_outputs, dim=2)
        gat_output = gat_output.permute(1, 0, 2, 3)
        
        lstm_outputs = []
        for node_idx in range(num_nodes):
            node_seq = gat_output[node_idx].transpose(0, 1)
            lstm_out, _ = self.model.lstm(node_seq)
            lstm_last = lstm_out[-1]
            lstm_last = self.model.bn_lstm(lstm_last)
            lstm_outputs.append(lstm_last)
        
        lstm_output = torch.stack(lstm_outputs, dim=0).transpose(0, 1)
        
        # Residual
        residual = features[:, :, -1, :]
        if self.model.residual_proj is not None:
            residual = self.model.residual_proj(residual)
        
        output = lstm_output + residual
        output = self.model.dropout_layer(output)
        predictions = self.model.output_layer(output)
        
        return predictions, attention_per_timestep
    
    def compute_attention_entropy(
        self,
        attention_weights: torch.Tensor,
        normalize: bool = True
    ) -> float:
        """
        Compute Shannon entropy of attention distribution.
        
        Higher entropy = attention more dispersed (uncertain)
        Lower entropy = attention more focused (confident)
        
        Hypothesis: Entropy increases before volatility events.
        
        Args:
            attention_weights: Attention coefficients [num_edges]
            normalize: Whether to normalize to [0,1]
        
        Returns:
            Entropy value
        """
        # Ensure positive and sum to 1
        attn = attention_weights.cpu().numpy()
        attn = np.abs(attn)
        attn = attn / (attn.sum() + 1e-10)
        
        # Shannon entropy
        ent = entropy(attn, base=2)
        
        if normalize:
            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(attn))
            ent = ent / max_entropy if max_entropy > 0 else 0
        
        return ent
    
    def identify_critical_edges(
        self,
        attention_weights: torch.Tensor,
        top_k: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Identify most important supply chain edges by attention weight.
        
        Args:
            attention_weights: Attention coefficients [num_edges]
            top_k: Number of top edges to return
        
        Returns:
            List of (source_stock, target_stock, attention_weight)
        """
        attn = attention_weights.cpu().numpy()
        
        # Get top-k indices
        top_indices = np.argsort(attn)[-top_k:][::-1]
        
        critical_edges = []
        for idx in top_indices:
            source_idx = self.edge_index[0, idx].item()
            target_idx = self.edge_index[1, idx].item()
            
            source_stock = self.stock_names[source_idx]
            target_stock = self.stock_names[target_idx]
            weight = attn[idx]
            
            critical_edges.append((source_stock, target_stock, weight))
        
        return critical_edges
    
    def analyze_temporal_attention_decay(
        self,
        attention_per_timestep: List[torch.Tensor]
    ) -> Dict[str, float]:
        """
        Analyze how attention changes across temporal window.
        
        Do recent timesteps get more attention than distant ones?
        
        Args:
            attention_per_timestep: List of attention weights per timestep
        
        Returns:
            Dictionary with decay statistics
        """
        # Average attention strength per timestep
        avg_attention = [attn.mean().item() for attn in attention_per_timestep]
        
        # Compute linear trend (decay rate)
        timesteps = np.arange(len(avg_attention))
        coeffs = np.polyfit(timesteps, avg_attention, deg=1)
        decay_rate = coeffs[0]  # Slope
        
        # Recent vs distant ratio
        recent_avg = np.mean(avg_attention[-5:])  # Last 5 timesteps
        distant_avg = np.mean(avg_attention[:5])  # First 5 timesteps
        recency_ratio = recent_avg / (distant_avg + 1e-10)
        
        return {
            'decay_rate': decay_rate,
            'recency_ratio': recency_ratio,
            'recent_attention': recent_avg,
            'distant_attention': distant_avg
        }
    
    def extract_attention_for_dataset(
        self,
        dataloader,
        save_path: Optional[Path] = None
    ):
        """
        Extract attention weights for entire dataset.
        
        Args:
            dataloader: DataLoader with test data
            save_path: Optional path to save results
        """
        self.logger.info("Extracting attention weights from dataset...")
        
        self.attention_weights = []
        self.attention_entropy = []
        self.timestamps = []
        
        sample_idx = 0
        
        with torch.no_grad():
            for features, edge_index, targets in dataloader:
                features = features.to(self.device)
                
                batch_size = features.shape[0]
                
                for b in range(batch_size):
                    # Single sample
                    sample_features = features[b:b+1]
                    
                    # Extract attention (requires modified forward)
                    # For now, we'll use a simplified approach
                    # In practice, need to modify GATLayer to return attention
                    
                    # Placeholder: Random attention for demonstration
                    # TODO: Replace with actual attention extraction
                    num_edges = self.edge_index.shape[1]
                    fake_attention = torch.rand(num_edges)
                    fake_attention = F.softmax(fake_attention, dim=0)
                    
                    self.attention_weights.append(fake_attention)
                    
                    # Compute entropy
                    ent = self.compute_attention_entropy(fake_attention)
                    self.attention_entropy.append(ent)
                    
                    # Store volatility level (target)
                    volatility = targets[b].mean().item()
                    self.volatility_levels.append(volatility)
                    
                    self.timestamps.append(sample_idx)
                    sample_idx += 1
        
        self.logger.info(f"Extracted attention for {len(self.attention_weights)} samples")
        
        if save_path:
            self._save_attention_data(save_path)
    
    def _save_attention_data(self, save_dir: Path):
        """Save extracted attention data"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as DataFrame
        data = {
            'sample_idx': self.timestamps,
            'attention_entropy': self.attention_entropy,
            'volatility_level': self.volatility_levels
        }
        
        # Add per-edge attention
        attention_array = torch.stack(self.attention_weights).numpy()
        for edge_idx in range(attention_array.shape[1]):
            source_idx = self.edge_index[0, edge_idx].item()
            target_idx = self.edge_index[1, edge_idx].item()
            edge_name = f"attn_{self.stock_names[source_idx]}_{self.stock_names[target_idx]}"
            data[edge_name] = attention_array[:, edge_idx]
        
        df = pd.DataFrame(data)
        df.to_csv(save_dir / 'attention_data.csv', index=False)
        
        self.logger.info(f"Saved attention data to: {save_dir / 'attention_data.csv'}")
    
    def test_entropy_hypothesis(self) -> Dict:
        """
        Test hypothesis: Attention entropy increases before volatility events.
        
        Returns:
            Dictionary with test results
        """
        self.logger.info("Testing entropy-volatility hypothesis...")
        
        # Define volatility events (top 25% volatility)
        vol_threshold = np.percentile(self.volatility_levels, 75)
        high_vol_events = np.array(self.volatility_levels) > vol_threshold
        
        # Compute entropy before events (1-5 timesteps ahead)
        entropy_before_events = []
        entropy_normal = []
        
        for i in range(5, len(self.attention_entropy)):
            if high_vol_events[i]:
                # Entropy 1-5 steps before
                entropy_before_events.extend(self.attention_entropy[i-5:i])
            else:
                entropy_normal.append(self.attention_entropy[i])
        
        # Statistical test
        from scipy.stats import ttest_ind
        
        if len(entropy_before_events) > 0 and len(entropy_normal) > 0:
            t_stat, p_value = ttest_ind(entropy_before_events, entropy_normal)
            
            mean_before = np.mean(entropy_before_events)
            mean_normal = np.mean(entropy_normal)
            
            result = {
                'hypothesis': 'Attention entropy increases before volatility events',
                'mean_entropy_before_events': mean_before,
                'mean_entropy_normal': mean_normal,
                'difference': mean_before - mean_normal,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'n_before_events': len(entropy_before_events),
                'n_normal': len(entropy_normal)
            }
            
            self.logger.info(f"Hypothesis test: p={p_value:.4f}, significant={result['significant']}")
            
            return result
        else:
            return {'error': 'Insufficient data for hypothesis testing'}
    
    def create_attention_heatmap(
        self,
        sample_indices: List[int],
        save_path: Path
    ):
        """
        Create heatmap showing attention weights over multiple samples.
        
        Args:
            sample_indices: Which samples to visualize
            save_path: Path to save heatmap
        """
        # Extract attention for selected samples
        attention_matrix = []
        for idx in sample_indices:
            if idx < len(self.attention_weights):
                attention_matrix.append(self.attention_weights[idx].numpy())
        
        attention_matrix = np.array(attention_matrix)
        
        # Create edge labels
        edge_labels = []
        for edge_idx in range(self.edge_index.shape[1]):
            source_idx = self.edge_index[0, edge_idx].item()
            target_idx = self.edge_index[1, edge_idx].item()
            edge_labels.append(
                f"{self.stock_names[source_idx]}→{self.stock_names[target_idx]}"
            )
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(
            attention_matrix.T,
            xticklabels=[f"Sample {i}" for i in sample_indices],
            yticklabels=edge_labels,
            cmap='YlOrRd',
            cbar_kws={'label': 'Attention Weight'},
            ax=ax
        )
        
        ax.set_title('Attention Weights Across Samples')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Supply Chain Edge')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.logger.info(f"Saved attention heatmap: {save_path}")
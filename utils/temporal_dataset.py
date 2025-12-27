"""
Temporal Graph Dataset for EV Supply Chain Volatility Prediction

This module implements a custom PyTorch Dataset for handling temporal graph data.
It creates sliding windows of node features and pairs them with volatility targets.

Author: EV Supply Chain GAT Team
Date: December 2024
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TemporalGraphDataset(Dataset):
    """
    PyTorch Dataset for temporal graph data with sliding windows.
    
    This dataset creates temporal windows from time-series graph data where:
    - Each sample is a window of T consecutive days
    - Features are normalized node attributes over time
    - Targets are next-day volatility predictions
    
    The dataset handles:
    - Loading processed parquet files
    - Creating sliding temporal windows
    - Pairing features with targets (next-day volatility)
    - Providing graph structure (edge_index)
    
    Data Flow:
        Parquet file → DataFrame → Sliding windows → (features, edge_index, target)
    
    Attributes:
        data_path (Path): Path to processed parquet file
        edge_index (torch.Tensor): Graph edge indices [2, num_edges]
        window_size (int): Number of days in temporal window
        target_feature (str): Name of feature to predict
        stock_col (str): Name of stock identifier column
        num_nodes (int): Number of nodes in graph
        features_df (pd.DataFrame): Loaded feature data
        samples (List): Pre-computed sliding window samples
        logger (logging.Logger): Logger instance
    """
    
    # Default configuration
    DEFAULT_WINDOW_SIZE = 20
    DEFAULT_TARGET_FEATURE = "garch_vol"  # Updated to match real data
    DEFAULT_STOCK_COL = "ticker"  # Updated to match real data
    DEFAULT_NUM_NODES = 8
    DEFAULT_NUM_EDGES = 21
    
    def __init__(
        self,
        data_path: str,
        edge_index: torch.Tensor,
        window_size: int = DEFAULT_WINDOW_SIZE,
        target_feature: str = DEFAULT_TARGET_FEATURE,
        stock_col: str = DEFAULT_STOCK_COL,
        num_nodes: int = DEFAULT_NUM_NODES,
        stride: int = 1
    ):
        """
        Initialize the Temporal Graph Dataset.
        
        Args:
            data_path: Path to processed parquet file (e.g., 'train_features.parquet')
            edge_index: Graph edge indices as torch.Tensor [2, num_edges]
            window_size: Number of days in each temporal window (default: 20)
            target_feature: Name of feature to predict (default: 'garch_vol')
            stock_col: Name of stock identifier column (default: 'ticker')
            num_nodes: Number of nodes in graph (default: 8)
            stride: Step size between consecutive windows (default: 1)
        """
        super().__init__()
        
        self.data_path = Path(data_path)
        self.edge_index = edge_index
        self.window_size = window_size
        self.target_feature = target_feature
        self.stock_col = stock_col
        self.num_nodes = num_nodes
        self.stride = stride
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Validate inputs
        self._validate_inputs()
        
        # Load data
        self.features_df = self._load_data()
        
        # Create sliding window samples
        self.samples = self._create_windows()
        
        self.logger.info(
            f"Dataset initialized: {len(self.samples)} samples, "
            f"window={window_size}, stride={stride}"
        )
    
    def _setup_logger(self) -> logging.Logger:
        """
        Configure logging for this dataset.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"{__name__}.TemporalGraphDataset")
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
    
    def _validate_inputs(self) -> None:
        """
        Validate initialization parameters.
        
        Raises:
            ValueError: If inputs are invalid
            FileNotFoundError: If data file doesn't exist
        """
        # Check file exists
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Check window size
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        
        # Check stride
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        
        # Check edge_index shape
        if self.edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape [2, num_edges], "
                f"got {self.edge_index.shape}"
            )
        
        # Check num_nodes
        if self.num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {self.num_nodes}")
    
    def _load_data(self) -> pd.DataFrame:
        """
        Load processed features from parquet file.
        
        Returns:
            DataFrame with columns: [date, ticker, feature1, feature2, ...]
        
        Raises:
            ValueError: If data format is invalid
        """
        self.logger.info(f"Loading data from {self.data_path}")
        
        df = pd.read_parquet(self.data_path)
        
        # Validate required columns
        required_cols = ['date', self.stock_col, self.target_feature]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {df.columns.tolist()}"
            )
        
        # Sort by date and stock
        df = df.sort_values(['date', self.stock_col]).reset_index(drop=True)
        
        self.logger.info(
            f"Loaded {len(df)} rows, {len(df.columns)} columns, "
            f"{df['date'].nunique()} unique dates"
        )
        
        return df
    
    def _create_windows(self) -> list:
        """
        Create sliding temporal windows from the dataset.
        
        Each window contains:
        - Features: [num_nodes, window_size, num_features]
        - Target: [num_nodes] (next-day volatility for each node)
        
        Returns:
            List of tuples: (start_idx, end_idx, target_idx)
        """
        # Get unique dates
        dates = sorted(self.features_df['date'].unique())
        num_dates = len(dates)
        
        samples = []
        
        # Create sliding windows
        # We need window_size days + 1 day for target
        for i in range(0, num_dates - self.window_size, self.stride):
            # Window: days i to i+window_size-1
            # Target: day i+window_size
            window_start_idx = i
            window_end_idx = i + self.window_size
            target_idx = i + self.window_size
            
            # Make sure target exists
            if target_idx < num_dates:
                samples.append((window_start_idx, window_end_idx, target_idx))
        
        self.logger.info(
            f"Created {len(samples)} sliding windows "
            f"(window_size={self.window_size}, stride={self.stride})"
        )
        
        return samples
    
    def _get_features_for_window(
        self, 
        window_start: int, 
        window_end: int
    ) -> torch.Tensor:
        """
        Extract features for a temporal window.
        
        Args:
            window_start: Start index in date array
            window_end: End index in date array (exclusive)
        
        Returns:
            Features tensor [num_nodes, window_size, num_features]
        """
        dates = sorted(self.features_df['date'].unique())
        window_dates = dates[window_start:window_end]
        
        # Get data for these dates
        window_df = self.features_df[
            self.features_df['date'].isin(window_dates)
        ].copy()
        
        # Get feature columns (exclude date, ticker)
        feature_cols = [
            col for col in window_df.columns 
            if col not in ['date', self.stock_col]
        ]
        
        # Reshape to [num_nodes, window_size, num_features]
        features = []
        stocks = sorted(window_df[self.stock_col].unique())
        
        for stock in stocks:
            stock_data = window_df[window_df[self.stock_col] == stock][feature_cols].values
            features.append(stock_data)
        
        features = np.stack(features, axis=0)  # [num_nodes, window_size, num_features]
        
        return torch.FloatTensor(features)
    
    def _get_target_for_date(self, target_idx: int) -> torch.Tensor:
        """
        Extract target values for a specific date.
        
        Args:
            target_idx: Index in date array
        
        Returns:
            Target tensor [num_nodes]
        """
        dates = sorted(self.features_df['date'].unique())
        target_date = dates[target_idx]
        
        # Get data for this date
        target_df = self.features_df[
            self.features_df['date'] == target_date
        ].copy()
        
        # Extract target feature
        targets = []
        stocks = sorted(target_df[self.stock_col].unique())
        
        for stock in stocks:
            stock_target = target_df[
                target_df[self.stock_col] == stock
            ][self.target_feature].values[0]
            targets.append(stock_target)
        
        return torch.FloatTensor(targets)
    
    def __len__(self) -> int:
        """
        Get number of samples in dataset.
        
        Returns:
            Number of sliding window samples
        """
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (features, edge_index, target):
                - features: [num_nodes, window_size, num_features]
                - edge_index: [2, num_edges]
                - target: [num_nodes]
        """
        window_start, window_end, target_idx = self.samples[idx]
        
        # Get features for window
        features = self._get_features_for_window(window_start, window_end)
        
        # Get target (next-day volatility)
        target = self._get_target_for_date(target_idx)
        
        return features, self.edge_index, target
    
    def get_stats(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        return {
            "num_samples": len(self.samples),
            "window_size": self.window_size,
            "stride": self.stride,
            "num_nodes": self.num_nodes,
            "num_edges": self.edge_index.shape[1],
            "num_dates": self.features_df['date'].nunique(),
            "num_features": len([
                col for col in self.features_df.columns 
                if col not in ['date', self.stock_col]
            ]),
            "target_feature": self.target_feature,
            "stock_col": self.stock_col,
            "data_path": str(self.data_path)
        }
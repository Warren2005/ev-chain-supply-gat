"""
Custom DataLoader for Temporal Graph Data

This module implements custom batching for graph data where each batch contains
multiple temporal windows but the same graph structure.

Key Challenge:
- Standard DataLoader stacks tensors: [batch, ...]
- Our data: features=[num_nodes, seq_len, features], edge_index=[2, num_edges]
- Need to keep graph structure constant while batching temporal windows

Author: EV Supply Chain GAT Team
Date: December 2024
"""

import logging
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader


def temporal_graph_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for temporal graph data.
    
    Standard PyTorch DataLoader can't handle graph data properly because:
    1. Edge_index is the same across all samples (same graph structure)
    2. Features have shape [num_nodes, seq_len, features] not [features]
    
    This function:
    - Stacks features into [batch, num_nodes, seq_len, features]
    - Returns single edge_index (same for all samples)
    - Stacks targets into [batch, num_nodes]
    
    Args:
        batch: List of (features, edge_index, target) tuples from dataset
    
    Returns:
        Tuple of (batched_features, edge_index, batched_targets):
            - batched_features: [batch_size, num_nodes, seq_len, features]
            - edge_index: [2, num_edges] (same for all samples)
            - batched_targets: [batch_size, num_nodes]
    
    Example:
        >>> # Each sample: features=[7,20,15], edge_index=[2,12], target=[7]
        >>> # Batch of 4: features=[4,7,20,15], edge_index=[2,12], target=[4,7]
    """
    # Extract features, edge_index, and targets
    features_list = [item[0] for item in batch]
    edge_indices = [item[1] for item in batch]
    targets_list = [item[2] for item in batch]
    
    # Stack features: [batch, num_nodes, seq_len, features]
    batched_features = torch.stack(features_list, dim=0)
    
    # Edge index is same for all samples - just use first one
    edge_index = edge_indices[0]
    
    # Verify all edge indices are identical (sanity check)
    for i, edge_idx in enumerate(edge_indices):
        if not torch.equal(edge_idx, edge_index):
            raise ValueError(
                f"Sample {i} has different edge_index! "
                f"All samples must have the same graph structure."
            )
    
    # Stack targets: [batch, num_nodes]
    batched_targets = torch.stack(targets_list, dim=0)
    
    return batched_features, edge_index, batched_targets


class TemporalGraphDataLoader:
    """
    Wrapper class for creating DataLoaders with temporal graph data.
    
    This class simplifies the creation of train/val/test DataLoaders with
    the correct custom collate function for graph data.
    
    Key Features:
    - Automatic custom collate function
    - Configurable batch size, shuffle, num_workers
    - Logging of DataLoader configuration
    
    Attributes:
        batch_size (int): Number of temporal windows per batch
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of parallel data loading workers
        drop_last (bool): Whether to drop last incomplete batch
        logger (logging.Logger): Logger instance
    """
    
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_NUM_WORKERS = 0  # 0 for single process (safer for development)
    
    def __init__(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        shuffle: bool = True,
        num_workers: int = DEFAULT_NUM_WORKERS,
        drop_last: bool = False
    ):
        """
        Initialize DataLoader wrapper.
        
        Args:
            batch_size: Number of samples per batch (default: 16)
            shuffle: Whether to shuffle data (default: True for training)
            num_workers: Number of subprocesses for data loading (default: 0)
            drop_last: Drop last incomplete batch (default: False)
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        
        # Setup logging
        self.logger = self._setup_logger()
        
        self.logger.info(
            f"DataLoader configured: batch_size={batch_size}, "
            f"shuffle={shuffle}, num_workers={num_workers}, drop_last={drop_last}"
        )
    
    def _setup_logger(self) -> logging.Logger:
        """
        Configure logging for this DataLoader.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"{__name__}.TemporalGraphDataLoader")
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
    
    def create_dataloader(self, dataset) -> DataLoader:
        """
        Create a DataLoader for the given dataset.
        
        Args:
            dataset: TemporalGraphDataset instance
        
        Returns:
            Configured DataLoader with custom collate function
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            collate_fn=temporal_graph_collate,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.logger.info(
            f"Created DataLoader: {len(dataset)} samples, "
            f"{len(dataloader)} batches"
        )
        
        return dataloader
    
    @staticmethod
    def create_train_val_test_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test DataLoaders.
        
        Convenience method to create all three DataLoaders with appropriate
        settings:
        - Train: shuffle=True
        - Val/Test: shuffle=False
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            batch_size: Batch size for all loaders
            num_workers: Number of workers for all loaders
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Training loader - with shuffling
        train_wrapper = TemporalGraphDataLoader(
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True  # Drop last batch for consistent batch size
        )
        train_loader = train_wrapper.create_dataloader(train_dataset)
        
        # Validation loader - no shuffling
        val_wrapper = TemporalGraphDataLoader(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False
        )
        val_loader = val_wrapper.create_dataloader(val_dataset)
        
        # Test loader - no shuffling
        test_wrapper = TemporalGraphDataLoader(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False
        )
        test_loader = test_wrapper.create_dataloader(test_dataset)
        
        return train_loader, val_loader, test_loader
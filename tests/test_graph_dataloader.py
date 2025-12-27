"""
Unit tests for Temporal Graph DataLoader

Run with: pytest tests/test_graph_dataloader.py -v
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch
import pandas as pd
import numpy as np

from utils.temporal_dataset import TemporalGraphDataset
from utils.graph_dataloader import (
    temporal_graph_collate,
    TemporalGraphDataLoader
)


class TestTemporalGraphCollate:
    """Test suite for custom collate function"""
    
    def test_collate_basic(self):
        """Test basic collate functionality"""
        # Create mock batch
        batch_size = 4
        num_nodes = 7
        seq_len = 20
        num_features = 15
        num_edges = 12
        
        batch = []
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        for _ in range(batch_size):
            features = torch.randn(num_nodes, seq_len, num_features)
            target = torch.randn(num_nodes)
            batch.append((features, edge_index, target))
        
        # Collate
        batched_features, batched_edge_index, batched_targets = temporal_graph_collate(batch)
        
        # Check shapes
        assert batched_features.shape == (batch_size, num_nodes, seq_len, num_features)
        assert batched_edge_index.shape == (2, num_edges)
        assert batched_targets.shape == (batch_size, num_nodes)
    
    def test_collate_preserves_edge_index(self):
        """Test that edge_index is preserved correctly"""
        batch_size = 3
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        batch = [
            (torch.randn(3, 20, 15), edge_index, torch.randn(3))
            for _ in range(batch_size)
        ]
        
        _, batched_edge_index, _ = temporal_graph_collate(batch)
        
        assert torch.equal(batched_edge_index, edge_index)
    
    def test_collate_different_edge_index_raises_error(self):
        """Test that different edge indices raise an error"""
        edge_index1 = torch.tensor([[0, 1], [1, 0]])
        edge_index2 = torch.tensor([[0, 2], [2, 0]])  # Different!
        
        batch = [
            (torch.randn(3, 20, 15), edge_index1, torch.randn(3)),
            (torch.randn(3, 20, 15), edge_index2, torch.randn(3))
        ]
        
        with pytest.raises(ValueError, match="different edge_index"):
            temporal_graph_collate(batch)
    
    def test_collate_single_sample(self):
        """Test collate with batch size 1"""
        features = torch.randn(7, 20, 15)
        edge_index = torch.randint(0, 7, (2, 12))
        target = torch.randn(7)
        
        batch = [(features, edge_index, target)]
        
        batched_features, batched_edge_index, batched_targets = temporal_graph_collate(batch)
        
        assert batched_features.shape == (1, 7, 20, 15)
        assert batched_targets.shape == (1, 7)


class TestTemporalGraphDataLoader:
    """Test suite for DataLoader wrapper"""
    
    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a small dataset for testing"""
        # Create synthetic data
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        stocks = ['TSLA', 'F', 'GM', 'RIVN', 'MGA', 'APTV', 'ALB']
        
        data = []
        for date in dates:
            for stock in stocks:
                row = {
                    'date': date,
                    'ticker': stock,
                    'garch_vol': np.random.rand(),
                    'log_return': np.random.randn() * 0.02,
                    'realized_vol': np.random.rand() * 0.03,
                }
                # Add more features to reach 15 total
                for i in range(12):
                    row[f'feature_{i}'] = np.random.randn()
                data.append(row)
        
        df = pd.DataFrame(data)
        parquet_path = tmp_path / "test_data.parquet"
        df.to_parquet(parquet_path)
        
        # Create dataset
        edge_index = torch.randint(0, 7, (2, 12))
        dataset = TemporalGraphDataset(
            data_path=str(parquet_path),
            edge_index=edge_index,
            window_size=20
        )
        
        return dataset
    
    def test_dataloader_creation(self, sample_dataset):
        """Test basic DataLoader creation"""
        wrapper = TemporalGraphDataLoader(batch_size=4)
        dataloader = wrapper.create_dataloader(sample_dataset)
        
        assert dataloader is not None
        assert dataloader.batch_size == 4
        assert len(dataloader) > 0
    
    def test_dataloader_iteration(self, sample_dataset):
        """Test iterating through DataLoader"""
        wrapper = TemporalGraphDataLoader(batch_size=4)
        dataloader = wrapper.create_dataloader(sample_dataset)
        
        batch_count = 0
        for features, edge_index, targets in dataloader:
            batch_count += 1
            
            # Check shapes
            assert features.ndim == 4  # [batch, nodes, seq, features]
            assert edge_index.ndim == 2  # [2, edges]
            assert targets.ndim == 2  # [batch, nodes]
            
            # Check batch dimension
            assert features.shape[0] <= 4  # At most batch_size
            assert targets.shape[0] <= 4
        
        assert batch_count > 0
    
    def test_dataloader_batch_consistency(self, sample_dataset):
        """Test that batches have consistent shapes"""
        wrapper = TemporalGraphDataLoader(batch_size=8)
        dataloader = wrapper.create_dataloader(sample_dataset)
        
        first_batch = next(iter(dataloader))
        features, edge_index, targets = first_batch
        
        num_nodes = features.shape[1]
        seq_len = features.shape[2]
        num_features = features.shape[3]
        
        # All batches should have same node/seq/feature dimensions
        for feat, _, targ in dataloader:
            assert feat.shape[1] == num_nodes
            assert feat.shape[2] == seq_len
            assert feat.shape[3] == num_features
            assert targ.shape[1] == num_nodes
    
    def test_shuffle_flag(self, sample_dataset):
        """Test shuffle flag affects order"""
        # Without shuffle
        wrapper_no_shuffle = TemporalGraphDataLoader(batch_size=4, shuffle=False)
        loader_no_shuffle = wrapper_no_shuffle.create_dataloader(sample_dataset)
        
        first_run = [features[0, 0, 0, 0].item() for features, _, _ in loader_no_shuffle]
        second_run = [features[0, 0, 0, 0].item() for features, _, _ in loader_no_shuffle]
        
        # Should be identical without shuffle
        assert first_run == second_run
    
    def test_create_train_val_test_loaders(self, sample_dataset):
        """Test creating all three loaders at once"""
        train_loader, val_loader, test_loader = TemporalGraphDataLoader.create_train_val_test_loaders(
            sample_dataset,
            sample_dataset,
            sample_dataset,
            batch_size=4
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        # Verify all loaders work
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0
        
        # Verify we can get batches from each
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        test_batch = next(iter(test_loader))
        
        assert len(train_batch) == 3  # (features, edge_index, targets)
        assert len(val_batch) == 3
        assert len(test_batch) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
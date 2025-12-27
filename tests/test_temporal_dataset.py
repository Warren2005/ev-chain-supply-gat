"""
Unit tests for Temporal Graph Dataset

Run with: pytest tests/test_temporal_dataset.py -v
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


class TestTemporalGraphDataset:
    """Test suite for TemporalGraphDataset"""
    
    @pytest.fixture
    def sample_data_path(self, tmp_path):
        """Create sample parquet file for testing"""
        # Create synthetic data
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        stocks = ['TSLA', 'F', 'GM', 'RIVN', 'MGA', 'APTV', 'ALB', 'LTHM']
        
        data = []
        for date in dates:
            for stock in stocks:
                row = {
                    'date': date,
                    'ticker': stock,  # Changed from 'stock' to 'ticker'
                    'garch_vol': np.random.rand(),  # Changed from 'garch_volatility' to 'garch_vol'
                    'log_return': np.random.randn() * 0.02,
                    'realized_vol': np.random.rand() * 0.03,  # Changed to match real data
                    'rsi': np.random.rand() * 100,
                    'volume_shock': np.random.randn(),
                    'vix': 15 + np.random.randn() * 5
                }
                data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save to parquet
        parquet_path = tmp_path / "test_features.parquet"
        df.to_parquet(parquet_path)
        
        return parquet_path
    
    @pytest.fixture
    def sample_edge_index(self):
        """Create sample edge index"""
        # 8 nodes, 21 edges
        edge_index = torch.randint(0, 8, (2, 21))
        return edge_index
    
    def test_initialization(self, sample_data_path, sample_edge_index):
        """Test dataset initializes correctly"""
        dataset = TemporalGraphDataset(
            data_path=str(sample_data_path),
            edge_index=sample_edge_index,
            window_size=20,
            num_nodes=8
        )
        
        assert dataset is not None
        assert dataset.window_size == 20
        assert dataset.num_nodes == 8
        assert len(dataset) > 0
    
    def test_invalid_file_path(self, sample_edge_index):
        """Test error handling for invalid file path"""
        with pytest.raises(FileNotFoundError):
            TemporalGraphDataset(
                data_path="nonexistent.parquet",
                edge_index=sample_edge_index
            )
    
    def test_invalid_window_size(self, sample_data_path, sample_edge_index):
        """Test error handling for invalid window size"""
        with pytest.raises(ValueError):
            TemporalGraphDataset(
                data_path=str(sample_data_path),
                edge_index=sample_edge_index,
                window_size=0
            )
    
    def test_invalid_edge_index_shape(self, sample_data_path):
        """Test error handling for invalid edge_index shape"""
        bad_edge_index = torch.randint(0, 8, (3, 21))  # Wrong first dimension
        
        with pytest.raises(ValueError):
            TemporalGraphDataset(
                data_path=str(sample_data_path),
                edge_index=bad_edge_index
            )
    
    def test_sample_shape(self, sample_data_path, sample_edge_index):
        """Test that samples have correct shape"""
        dataset = TemporalGraphDataset(
            data_path=str(sample_data_path),
            edge_index=sample_edge_index,
            window_size=20,
            num_nodes=8
        )
        
        features, edge_index, target = dataset[0]
        
        # Check shapes
        assert features.shape[0] == 8  # num_nodes
        assert features.shape[1] == 20  # window_size
        assert features.shape[2] > 0  # num_features
        assert edge_index.shape[0] == 2
        assert target.shape[0] == 8  # num_nodes
    
    def test_multiple_samples(self, sample_data_path, sample_edge_index):
        """Test accessing multiple samples"""
        dataset = TemporalGraphDataset(
            data_path=str(sample_data_path),
            edge_index=sample_edge_index,
            window_size=20,
            num_nodes=8
        )
        
        # Get multiple samples
        sample1 = dataset[0]
        sample2 = dataset[1]
        
        # Shapes should be consistent
        assert sample1[0].shape == sample2[0].shape
        assert sample1[2].shape == sample2[2].shape
        
        # Edge index should be same
        assert torch.equal(sample1[1], sample2[1])
    
    def test_sliding_window_count(self, sample_data_path, sample_edge_index):
        """Test that correct number of windows is created"""
        dataset = TemporalGraphDataset(
            data_path=str(sample_data_path),
            edge_index=sample_edge_index,
            window_size=20,
            num_nodes=8,
            stride=1
        )
        
        # 50 dates, window=20, stride=1
        # Should have 50 - 20 = 30 windows
        assert len(dataset) == 30
    
    def test_stride_parameter(self, sample_data_path, sample_edge_index):
        """Test stride parameter works correctly"""
        dataset_stride1 = TemporalGraphDataset(
            data_path=str(sample_data_path),
            edge_index=sample_edge_index,
            window_size=20,
            stride=1
        )
        
        dataset_stride5 = TemporalGraphDataset(
            data_path=str(sample_data_path),
            edge_index=sample_edge_index,
            window_size=20,
            stride=5
        )
        
        # Stride=5 should have ~1/5 the samples
        assert len(dataset_stride5) < len(dataset_stride1)
        assert len(dataset_stride5) == 6  # (50-20)/5 = 6
    
    def test_no_nan_values(self, sample_data_path, sample_edge_index):
        """Test that samples don't contain NaN values"""
        dataset = TemporalGraphDataset(
            data_path=str(sample_data_path),
            edge_index=sample_edge_index,
            window_size=20
        )
        
        features, _, target = dataset[0]
        
        assert not torch.isnan(features).any()
        assert not torch.isnan(target).any()
    
    def test_get_stats(self, sample_data_path, sample_edge_index):
        """Test get_stats method"""
        dataset = TemporalGraphDataset(
            data_path=str(sample_data_path),
            edge_index=sample_edge_index,
            window_size=20
        )
        
        stats = dataset.get_stats()
        
        assert isinstance(stats, dict)
        assert 'num_samples' in stats
        assert 'window_size' in stats
        assert 'num_nodes' in stats
        assert stats['window_size'] == 20
        assert stats['num_nodes'] == 8
    
    def test_iteration(self, sample_data_path, sample_edge_index):
        """Test dataset can be iterated"""
        dataset = TemporalGraphDataset(
            data_path=str(sample_data_path),
            edge_index=sample_edge_index,
            window_size=20
        )
        
        count = 0
        for features, edge_index, target in dataset:
            count += 1
            assert features.shape[0] == 8
            assert target.shape[0] == 8
        
        assert count == len(dataset)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Unit tests for ST-GAT Trainer

Run with: pytest tests/test_trainer.py -v
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

from models.st_gat import STGAT
from utils.temporal_dataset import TemporalGraphDataset
from utils.graph_dataloader import TemporalGraphDataLoader
from utils.trainer import STGATTrainer


class TestSTGATTrainer:
    """Test suite for Trainer"""
    
    @pytest.fixture
    def sample_datasets(self, tmp_path):
        """Create small datasets for testing"""
        dates = pd.date_range('2020-01-01', periods=60, freq='D')
        stocks = ['TSLA', 'F', 'GM', 'RIVN', 'MGA', 'APTV', 'ALB']
        
        data = []
        for date in dates:
            for stock in stocks:
                row = {
                    'date': date,
                    'ticker': stock,
                    'garch_vol': np.random.rand() * 0.02,
                }
                # Add 14 more features to reach 15 total
                for i in range(14):
                    row[f'feature_{i}'] = np.random.randn() * 0.1
                data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save train and val
        train_path = tmp_path / "train.parquet"
        val_path = tmp_path / "val.parquet"
        
        df.to_parquet(train_path)
        df.to_parquet(val_path)
        
        # Create datasets
        edge_index = torch.randint(0, 7, (2, 12))
        
        train_dataset = TemporalGraphDataset(
            data_path=str(train_path),
            edge_index=edge_index,
            window_size=20
        )
        
        val_dataset = TemporalGraphDataset(
            data_path=str(val_path),
            edge_index=edge_index,
            window_size=20
        )
        
        return train_dataset, val_dataset
    
    @pytest.fixture
    def sample_model(self):
        """Create small model for testing"""
        model = STGAT(
            num_nodes=7,
            num_edges=12,
            input_features=15,
            gat_hidden_dim=32,  # Smaller for testing
            gat_heads=2,
            gat_layers=1,
            lstm_hidden_dim=32,
            lstm_layers=1,
            temporal_window=20
        )
        return model
    
    def test_trainer_initialization(self, sample_model, sample_datasets):
        """Test trainer initializes correctly"""
        train_dataset, val_dataset = sample_datasets
        
        train_loader, val_loader, _ = TemporalGraphDataLoader.create_train_val_test_loaders(
            train_dataset, val_dataset, val_dataset,
            batch_size=4
        )
        
        trainer = STGATTrainer(
            model=sample_model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=1e-3,
            device='cpu'
        )
        
        assert trainer is not None
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None
    
    def test_single_epoch_training(self, sample_model, sample_datasets):
        """Test single epoch of training"""
        train_dataset, val_dataset = sample_datasets
        
        train_loader, val_loader, _ = TemporalGraphDataLoader.create_train_val_test_loaders(
            train_dataset, val_dataset, val_dataset,
            batch_size=4
        )
        
        trainer = STGATTrainer(
            model=sample_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu'
        )
        
        # Train one epoch
        train_loss = trainer.train_epoch()
        
        assert isinstance(train_loss, float)
        assert train_loss > 0
        assert not np.isnan(train_loss)
        assert not np.isinf(train_loss)
    
    def test_validation(self, sample_model, sample_datasets):
        """Test validation loop"""
        train_dataset, val_dataset = sample_datasets
        
        train_loader, val_loader, _ = TemporalGraphDataLoader.create_train_val_test_loaders(
            train_dataset, val_dataset, val_dataset,
            batch_size=4
        )
        
        trainer = STGATTrainer(
            model=sample_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu'
        )
        
        val_loss, metrics = trainer.validate_epoch()
        
        assert isinstance(val_loss, float)
        assert val_loss > 0
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
    
    def test_multiple_epochs(self, sample_model, sample_datasets):
        """Test training for multiple epochs"""
        train_dataset, val_dataset = sample_datasets
        
        train_loader, val_loader, _ = TemporalGraphDataLoader.create_train_val_test_loaders(
            train_dataset, val_dataset, val_dataset,
            batch_size=4
        )
        
        trainer = STGATTrainer(
            model=sample_model,
            train_loader=train_loader,
            val_loader=val_loader,
            early_stopping_patience=5,
            device='cpu'
        )
        
        history = trainer.train(num_epochs=3, verbose=False)
        
        assert len(history['train_losses']) == 3
        assert len(history['val_losses']) == 3
        assert trainer.best_val_loss > 0
    
    def test_checkpointing(self, sample_model, sample_datasets, tmp_path):
        """Test model checkpointing"""
        train_dataset, val_dataset = sample_datasets
        
        train_loader, val_loader, _ = TemporalGraphDataLoader.create_train_val_test_loaders(
            train_dataset, val_dataset, val_dataset,
            batch_size=4
        )
        
        trainer = STGATTrainer(
            model=sample_model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            device='cpu'
        )
        
        # Train and save
        trainer.train(num_epochs=2, verbose=False)
        
        # Check checkpoint exists
        checkpoint_path = tmp_path / "checkpoints" / "best_model.pt"
        assert checkpoint_path.exists()
    
    def test_checkpoint_loading(self, sample_model, sample_datasets, tmp_path):
        """Test loading from checkpoint"""
        train_dataset, val_dataset = sample_datasets
        
        train_loader, val_loader, _ = TemporalGraphDataLoader.create_train_val_test_loaders(
            train_dataset, val_dataset, val_dataset,
            batch_size=4
        )
        
        # Train and save
        trainer1 = STGATTrainer(
            model=sample_model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            device='cpu'
        )
        trainer1.train(num_epochs=2, verbose=False)
        
        # Store best epoch and loss
        original_best_epoch = trainer1.current_epoch  # Might be 1 or 2
        original_best_loss = trainer1.best_val_loss
        
        # Load checkpoint into new model
        new_model = STGAT(
            num_nodes=7, num_edges=12, input_features=15,
            gat_hidden_dim=32, gat_heads=2, gat_layers=1,
            lstm_hidden_dim=32, lstm_layers=1, temporal_window=20
        )
        
        trainer2 = STGATTrainer(
            model=new_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu'
        )
        
        checkpoint_path = tmp_path / "checkpoints" / "best_model.pt"
        trainer2.load_checkpoint(str(checkpoint_path))
        
        # Verify checkpoint was loaded correctly
        assert trainer2.best_val_loss == original_best_loss
        assert trainer2.current_epoch > 0  # Should have loaded some epoch
        assert len(trainer2.train_losses) > 0  # Should have history


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
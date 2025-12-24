"""
Unit tests for FeatureEngineer

Tests all functionality of the feature engineering module.

Run with: pytest tests/test_feature_engineering.py -v
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path

from utils.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class"""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for test data"""
        temp_path = tempfile.mkdtemp()
        market_dir = Path(temp_path) / "market_data"
        macro_dir = Path(temp_path) / "macro_data"
        output_dir = Path(temp_path) / "output"
        
        market_dir.mkdir(parents=True)
        macro_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        yield {
            'market': str(market_dir),
            'macro': str(macro_dir),
            'output': str(output_dir)
        }
        
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def engineer(self, temp_dirs):
        """Create a FeatureEngineer instance with temp directories"""
        return FeatureEngineer(
            market_data_dir=temp_dirs['market'],
            macro_data_dir=temp_dirs['macro'],
            output_dir=temp_dirs['output']
        )
    
    @pytest.fixture
    def sample_stock_data(self):
        """Generate sample stock data for testing"""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='B')
        
        # Generate realistic price data
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))
        
        df = pd.DataFrame({
            'date': dates,
            'open': prices * 0.98,
            'high': prices * 1.02,
            'low': prices * 0.97,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        
        return df
    
    def test_initialization(self, temp_dirs):
        """Test that FeatureEngineer initializes correctly"""
        engineer = FeatureEngineer(
            market_data_dir=temp_dirs['market'],
            macro_data_dir=temp_dirs['macro'],
            output_dir=temp_dirs['output']
        )
        
        assert engineer.market_data_dir.exists()
        assert engineer.output_dir.exists()
        assert engineer.logger is not None
        assert engineer.VOLATILITY_WINDOW == 20
        assert engineer.MIN_PERIODS == 10
    
    def test_load_stock_data_success(self, engineer, temp_dirs, sample_stock_data):
        """Test successful loading of stock data"""
        # Save sample data
        filepath = Path(temp_dirs['market']) / "TSLA_prices.csv"
        sample_stock_data.to_csv(filepath, index=False)
        
        # Load data
        df = engineer.load_stock_data('TSLA')
        
        assert df is not None
        assert len(df) == 100
        assert 'date' in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['date'])
    
    def test_load_stock_data_file_not_found(self, engineer):
        """Test handling of missing file"""
        df = engineer.load_stock_data('NONEXISTENT')
        
        assert df is None
    
    def test_calculate_log_returns(self, engineer, sample_stock_data):
        """Test log return calculation"""
        df = engineer.calculate_log_returns(sample_stock_data)
        
        assert 'log_return' in df.columns
        assert df['log_return'].isna().sum() == 1  # First row should be NaN
        assert len(df) == 100
        
        # Check that log returns are reasonable
        assert df['log_return'].abs().max() < 0.5  # No extreme returns in sample data
    
    def test_calculate_log_returns_formula(self, engineer):
        """Test that log return formula is correct"""
        # Simple test case
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=3),
            'close': [100.0, 110.0, 105.0]
        })
        
        result = engineer.calculate_log_returns(df)
        
        # ln(110/100) ≈ 0.0953
        assert abs(result['log_return'].iloc[1] - np.log(110/100)) < 1e-6
        
        # ln(105/110) ≈ -0.0465
        assert abs(result['log_return'].iloc[2] - np.log(105/110)) < 1e-6
    
    def test_calculate_realized_volatility(self, engineer, sample_stock_data):
        """Test realized volatility calculation"""
        # First calculate returns
        df = engineer.calculate_log_returns(sample_stock_data)
        
        # Then calculate volatility
        df = engineer.calculate_realized_volatility(df, window=20)
        
        assert 'realized_vol' in df.columns
        assert df['realized_vol'].notna().sum() > 0
        
        # Volatility should be positive
        assert (df['realized_vol'].dropna() >= 0).all()
    
    def test_calculate_realized_volatility_custom_window(self, engineer, sample_stock_data):
        """Test volatility calculation with custom window"""
        df = engineer.calculate_log_returns(sample_stock_data)
        df = engineer.calculate_realized_volatility(df, window=10)
        
        # Should have more non-null values with smaller window
        assert df['realized_vol'].notna().sum() >= 90
    
    def test_calculate_basic_features(self, engineer, temp_dirs, sample_stock_data):
        """Test end-to-end basic feature calculation"""
        # Save sample data
        filepath = Path(temp_dirs['market']) / "TSLA_prices.csv"
        sample_stock_data.to_csv(filepath, index=False)
        
        # Calculate features
        result = engineer.calculate_basic_features('TSLA')
        
        assert result is not None
        assert 'log_return' in result.columns
        assert 'realized_vol' in result.columns
        assert 'ticker' in result.columns
        assert result['ticker'].iloc[0] == 'TSLA'
    
    def test_calculate_basic_features_missing_file(self, engineer):
        """Test handling when stock file doesn't exist"""
        result = engineer.calculate_basic_features('NONEXISTENT')
        
        assert result is None
    
    def test_process_all_stocks(self, engineer, temp_dirs, sample_stock_data):
        """Test processing multiple stocks"""
        # Create data for 3 stocks
        for ticker in ['TSLA', 'AAPL', 'F']:
            filepath = Path(temp_dirs['market']) / f"{ticker}_prices.csv"
            sample_stock_data.to_csv(filepath, index=False)
        
        # Process all
        results = engineer.process_all_stocks(['TSLA', 'AAPL', 'F'])
        
        assert len(results) == 3
        assert 'TSLA' in results
        assert 'AAPL' in results
        assert 'F' in results
        
        # Check stats
        assert len(engineer.processing_stats['stocks_processed']) == 3
        assert len(engineer.processing_stats['stocks_failed']) == 0
    
    def test_validate_features_success(self, engineer, sample_stock_data):
        """Test validation of good quality features"""
        df = engineer.calculate_log_returns(sample_stock_data)
        df = engineer.calculate_realized_volatility(df)
        df['ticker'] = 'TSLA'
        
        is_valid, issues = engineer.validate_features(df, 'TSLA')
        
        assert is_valid is True
        assert len(issues) == 0
    
    def test_validate_features_missing_columns(self, engineer):
        """Test validation fails with missing columns"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'close': [100.0] * 10
        })
        
        is_valid, issues = engineer.validate_features(df, 'TEST')
        
        assert is_valid is False
        assert any('missing' in str(issue).lower() for issue in issues)
    
    def test_validate_features_infinite_values(self, engineer):
        """Test validation detects infinite values"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'ticker': ['TSLA'] * 10,
            'log_return': [0.01, 0.02, np.inf, 0.03, 0.01] * 2,
            'realized_vol': [0.2] * 10
        })
        
        is_valid, issues = engineer.validate_features(df, 'TSLA')
        
        assert is_valid is False
        assert any('infinite' in str(issue).lower() for issue in issues)
    
    def test_validate_features_negative_volatility(self, engineer):
        """Test validation detects negative volatility"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'ticker': ['TSLA'] * 10,
            'log_return': [0.01] * 10,
            'realized_vol': [-0.1, 0.2, 0.3, 0.2, 0.1] * 2
        })
        
        is_valid, issues = engineer.validate_features(df, 'TSLA')
        
        assert is_valid is False
        assert any('negative' in str(issue).lower() for issue in issues)
    
    def test_save_features(self, engineer, temp_dirs, sample_stock_data):
        """Test saving features to Parquet"""
        # Create features for 2 stocks
        for ticker in ['TSLA', 'AAPL']:
            filepath = Path(temp_dirs['market']) / f"{ticker}_prices.csv"
            sample_stock_data.to_csv(filepath, index=False)
        
        results = engineer.process_all_stocks(['TSLA', 'AAPL'])
        
        # Save features
        success = engineer.save_features(results, "test_features.parquet")
        
        assert success is True
        
        # Verify file exists and can be loaded
        filepath = Path(temp_dirs['output']) / "test_features.parquet"
        assert filepath.exists()
        
        loaded = pd.read_parquet(filepath)
        assert len(loaded) == 200  # 100 rows × 2 stocks
        assert 'ticker' in loaded.columns
        assert set(loaded['ticker'].unique()) == {'TSLA', 'AAPL'}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
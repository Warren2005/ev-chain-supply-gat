"""
Unit tests for MacroDataCollector

Tests all functionality of the macro data collection module.

Run with: pytest tests/test_macro_data.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from utils.macro_data import MacroDataCollector


class TestMacroDataCollector:
    """Test suite for MacroDataCollector class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def collector(self, temp_dir):
        """Create a MacroDataCollector instance with temp directory"""
        return MacroDataCollector(output_dir=temp_dir)
    
    @pytest.fixture
    def sample_indicator_data(self):
        """Generate sample indicator data for testing"""
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='B')
        
        df = pd.DataFrame({
            'date': dates,
            'vix': np.random.uniform(10, 40, len(dates))
        })
        
        return df
    
    def test_initialization(self, temp_dir):
        """Test that MacroDataCollector initializes correctly"""
        collector = MacroDataCollector(output_dir=temp_dir)
        
        assert collector.output_dir == Path(temp_dir)
        assert collector.output_dir.exists()
        assert collector.logger is not None
        assert len(collector.MACRO_TICKERS) == 5
        assert 'vix' in collector.MACRO_TICKERS
    
    def test_macro_tickers_defined(self, collector):
        """Test that all required macro tickers are defined"""
        expected_tickers = ['vix', 'treasury_10y', 'lithium_proxy', 'copper', 'dxy']
        
        for ticker in expected_tickers:
            assert ticker in collector.MACRO_TICKERS
            assert isinstance(collector.MACRO_TICKERS[ticker], str)
    
    @patch('utils.macro_data.yf.download')
    def test_download_indicator_success(self, mock_download, collector):
        """Test successful indicator download"""
        # Create mock data
        dates = pd.date_range('2020-01-01', periods=100)
        mock_data = pd.DataFrame({
            'Close': np.random.uniform(15, 30, 100)
        }, index=dates)
        mock_download.return_value = mock_data
        
        # Download indicator
        df = collector.download_indicator(
            'vix',
            '^VIX',
            '2020-01-01',
            '2020-12-31'
        )
        
        # Assertions
        assert df is not None
        assert len(df) == 100
        assert 'date' in df.columns
        assert 'vix' in df.columns
    
    @patch('utils.macro_data.yf.download')
    def test_download_indicator_empty_response(self, mock_download, collector):
        """Test handling of empty data response"""
        mock_download.return_value = pd.DataFrame()
        
        df = collector.download_indicator(
            'vix',
            '^VIX',
            '2020-01-01',
            '2020-12-31'
        )
        
        assert df is None
    
    @patch('utils.macro_data.yf.download')
    def test_download_indicator_retry_logic(self, mock_download, collector):
        """Test retry logic on failures"""
        # Fail twice, then succeed
        dates = pd.date_range('2020-01-01', periods=50)
        success_data = pd.DataFrame({
            'Close': [20.0] * 50
        }, index=dates)
        
        mock_download.side_effect = [
            Exception("Network error"),
            Exception("Timeout"),
            success_data
        ]
        
        df = collector.download_indicator(
            'vix',
            '^VIX',
            '2020-01-01',
            '2020-12-31',
            max_retries=3
        )
        
        assert df is not None
        assert len(df) == 50
    
    def test_combine_indicators(self, collector):
        """Test combining multiple indicators"""
        # Create sample indicators
        dates = pd.date_range('2020-01-01', periods=100)
        
        indicators = {
            'vix': pd.DataFrame({
                'date': dates,
                'vix': np.random.uniform(10, 40, 100)
            }),
            'treasury_10y': pd.DataFrame({
                'date': dates,
                'treasury_10y': np.random.uniform(0.5, 3.0, 100)
            })
        }
        
        combined = collector.combine_indicators(indicators)
        
        assert combined is not None
        assert len(combined) == 100
        assert 'date' in combined.columns
        assert 'vix' in combined.columns
        assert 'treasury_10y' in combined.columns
    
    def test_combine_indicators_empty(self, collector):
        """Test combining with empty dictionary"""
        combined = collector.combine_indicators({})
        
        assert combined.empty
    
    def test_forward_fill_missing(self, collector):
        """Test forward filling missing values"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'vix': [20.0, np.nan, np.nan, 25.0, np.nan, 30.0, np.nan, np.nan, np.nan, 35.0]
        })
        
        filled = collector.forward_fill_missing(df, max_consecutive_nulls=3)
        
        # Check that values were forward filled
        assert filled['vix'].isna().sum() < df['vix'].isna().sum()
    
    def test_align_with_dates(self, collector):
        """Test aligning macro data with specific dates"""
        # Macro data (may have gaps)
        macro_dates = pd.date_range('2020-01-01', periods=100, freq='B')
        df = pd.DataFrame({
            'date': macro_dates,
            'vix': np.random.uniform(10, 40, 100)
        })
        
        # Target dates (e.g., from market data)
        target_dates = pd.date_range('2020-01-01', periods=50, freq='B')
        
        aligned = collector.align_with_dates(df, target_dates)
        
        assert len(aligned) == 50
        assert 'date' in aligned.columns
        assert 'vix' in aligned.columns
    
    def test_validate_data_success(self, collector):
        """Test validation of good quality data"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=200),
            'vix': np.random.uniform(10, 40, 200),
            'treasury_10y': np.random.uniform(0.5, 3.0, 200)
        })
        
        is_valid, issues = collector.validate_data(df)
        
        assert is_valid is True
        assert len(issues) == 0
    
    def test_validate_data_missing_date_column(self, collector):
        """Test validation fails without date column"""
        df = pd.DataFrame({
            'vix': [20.0] * 100
        })
        
        is_valid, issues = collector.validate_data(df)
        
        assert is_valid is False
        assert any("date" in str(issue).lower() for issue in issues)
    
    def test_validate_data_high_missing_percentage(self, collector):
        """Test validation fails with high missing data"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'vix': [np.nan] * 100  # 100% missing
        })
        
        is_valid, issues = collector.validate_data(df, max_missing_pct=10.0)
        
        assert is_valid is False
        assert any("missing" in str(issue).lower() for issue in issues)
    
    def test_validate_data_vix_out_of_range(self, collector):
        """Test validation detects VIX values out of range"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'vix': [150.0] * 100  # VIX > 100 is unrealistic
        })
        
        is_valid, issues = collector.validate_data(df)
        
        assert is_valid is False
        assert any("vix" in str(issue).lower() for issue in issues)
    
    def test_save_indicator(self, collector, sample_indicator_data):
        """Test saving individual indicator"""
        result = collector._save_indicator(sample_indicator_data, 'vix')
        
        assert result is True
        
        # Check file exists
        filepath = collector.output_dir / "vix.csv"
        assert filepath.exists()
        
        # Verify content
        loaded = pd.read_csv(filepath)
        assert len(loaded) == len(sample_indicator_data)
    
    def test_save_combined(self, collector):
        """Test saving combined indicators"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'vix': np.random.uniform(10, 40, 100),
            'treasury_10y': np.random.uniform(0.5, 3.0, 100)
        })
        
        result = collector.save_combined(df, "test_macro.csv")
        
        assert result is True
        
        # Check file exists
        filepath = collector.output_dir / "test_macro.csv"
        assert filepath.exists()


class TestIntegration:
    """Integration tests (may hit real APIs)"""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_real_download_vix(self):
        """Test actual VIX download from Yahoo Finance"""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = MacroDataCollector(output_dir=temp_dir)
            
            df = collector.download_indicator(
                'vix',
                '^VIX',
                '2023-01-01',
                '2023-01-31'
            )
            
            assert df is not None
            assert len(df) > 0
            assert 'vix' in df.columns
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_real_download_all_indicators(self):
        """Test downloading all real indicators"""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = MacroDataCollector(output_dir=temp_dir)
            
            indicators = collector.download_all_indicators(
                '2023-01-01',
                '2023-01-31'
            )
            
            # Should have at least some indicators
            assert len(indicators) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
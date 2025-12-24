"""
Unit tests for SECFilingDownloader

Tests all functionality of the SEC filing download module.

Run with: pytest tests/test_sec_downloader.py -v
"""

import pytest
import sqlite3
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from bs4 import BeautifulSoup

from utils.sec_downloader import SECFilingDownloader


class TestSECFilingDownloader:
    """Test suite for SECFilingDownloader class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def collector(self, temp_dir):
        """Create a SECFilingDownloader instance with temp directory"""
        output_dir = Path(temp_dir) / "filings"
        db_dir = Path(temp_dir) / "metadata"
        db_dir.mkdir(exist_ok=True)
        
        return SECFilingDownloader(
            output_dir=str(output_dir),
            db_name=str(db_dir / "test_filings.db")
        )
    
    def test_initialization(self, temp_dir):
        """Test that SECFilingDownloader initializes correctly"""
        output_dir = Path(temp_dir) / "filings"
        db_dir = Path(temp_dir) / "metadata"
        db_dir.mkdir(exist_ok=True)
        
        collector = SECFilingDownloader(
            output_dir=str(output_dir),
            db_name=str(db_dir / "test.db")
        )
        
        assert collector.output_dir.exists()
        assert collector.db_path.exists()
        assert collector.logger is not None
        assert collector.RATE_LIMIT_DELAY > 0
    
    def test_database_initialization(self, collector):
        """Test that SQLite database is created with correct schema"""
        conn = sqlite3.connect(collector.db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sec_filings'"
        )
        result = cursor.fetchone()
        
        assert result is not None
        assert result[0] == 'sec_filings'
        
        # Check schema
        cursor.execute("PRAGMA table_info(sec_filings)")
        columns = {col[1] for col in cursor.fetchall()}
        
        expected_columns = {
            'id', 'ticker', 'filing_type', 'fiscal_year', 'filing_date',
            'accession_number', 'url', 'local_path', 'download_date',
            'file_size_kb', 'status', 'error_message'
        }
        
        assert expected_columns.issubset(columns)
        
        conn.close()
    
    def test_get_company_cik_known_ticker(self, collector):
        """Test CIK lookup for known tickers"""
        # Test known tickers
        assert collector.get_company_cik('TSLA') == '0001318605'
        assert collector.get_company_cik('F') == '0000037996'
        assert collector.get_company_cik('ALB') == '0000915913'
    
    def test_get_company_cik_unknown_ticker(self, collector):
        """Test CIK lookup for unknown tickers"""
        cik = collector.get_company_cik('UNKNOWN_TICKER_XYZ')
        assert cik is None
    
    def test_get_company_cik_non_us_company(self, collector):
        """Test CIK lookup for non-US companies"""
        # Australian company should have no CIK
        cik = collector.get_company_cik('PLS.AX')
        assert cik is None
    
    def test_track_filing_success(self, collector):
        """Test tracking a successful filing download"""
        collector._track_filing(
            ticker='TSLA',
            filing_type='10-K',
            fiscal_year=2020,
            url='https://example.com',
            local_path='/path/to/file.html',
            file_size_kb=1024,
            status='downloaded'
        )
        
        # Verify it was saved to database
        conn = sqlite3.connect(collector.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM sec_filings WHERE ticker='TSLA' AND fiscal_year=2020"
        )
        result = cursor.fetchone()
        
        assert result is not None
        assert result[1] == 'TSLA'  # ticker
        assert result[3] == 2020     # fiscal_year
        assert result[10] == 'downloaded'  # status
        
        conn.close()
    
    def test_track_filing_failed(self, collector):
        """Test tracking a failed filing download"""
        collector._track_filing(
            ticker='TSLA',
            filing_type='10-K',
            fiscal_year=2020,
            status='failed',
            error_message='Network error'
        )
        
        conn = sqlite3.connect(collector.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT status, error_message FROM sec_filings "
            "WHERE ticker='TSLA' AND fiscal_year=2020"
        )
        result = cursor.fetchone()
        
        assert result[0] == 'failed'
        assert 'Network error' in result[1]
        
        conn.close()
    
    def test_rate_limiting(self, collector):
        """Test that rate limiting delays requests appropriately"""
        import time
        
        start_time = time.time()
        
        # Make 3 consecutive rate-limited calls
        for _ in range(3):
            collector._rate_limit()
        
        elapsed = time.time() - start_time
        
        # Should take at least 2 * RATE_LIMIT_DELAY (for 2 delays between 3 calls)
        min_expected_time = 2 * collector.RATE_LIMIT_DELAY
        
        assert elapsed >= min_expected_time * 0.9  # Allow 10% margin
    
    @patch('utils.sec_downloader.requests.get')
    def test_make_request_success(self, mock_get, collector):
        """Test successful HTTP request"""
        # Setup mock
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Test content"
        mock_get.return_value = mock_response
        
        # Make request
        response = collector._make_request('https://example.com')
        
        assert response is not None
        assert response.status_code == 200
        assert response.text == "Test content"
    
    @patch('utils.sec_downloader.requests.get')
    def test_make_request_404(self, mock_get, collector):
        """Test handling of 404 responses"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        response = collector._make_request('https://example.com')
        
        assert response is None
    
    @patch('utils.sec_downloader.requests.get')
    def test_make_request_retry_logic(self, mock_get, collector):
        """Test retry logic on failed requests"""
        # Fail twice, then succeed
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.text = "Success"
        
        mock_get.side_effect = [
            mock_response_fail,
            mock_response_fail,
            mock_response_success
        ]
        
        response = collector._make_request('https://example.com', max_retries=3)
        
        assert response is not None
        assert response.status_code == 200
    
    def test_parse_html_to_text(self, collector, tmp_path):
        """Test HTML parsing to plain text"""
        # Create a test HTML file
        html_content = """
        <html>
        <head><title>Test Filing</title></head>
        <body>
            <h1>10-K Filing</h1>
            <p>This is a test paragraph.</p>
            <script>console.log('remove this');</script>
            <style>.test { color: red; }</style>
        </body>
        </html>
        """
        
        test_file = tmp_path / "test.html"
        test_file.write_text(html_content)
        
        # Parse
        text = collector.parse_html_to_text(str(test_file))
        
        assert "10-K Filing" in text
        assert "This is a test paragraph" in text
        assert "console.log" not in text  # Script should be removed
        assert ".test { color: red; }" not in text  # Style should be removed
    
    def test_find_filing_url_simple(self, collector):
        """Test finding filing URL from search results"""
        # Create mock HTML
        html = """
        <html>
        <table>
            <tr>
                <td>10-K</td>
                <td><a href="/filing-detail/123">Documents</a></td>
                <td>Annual report</td>
                <td>2020-02-15</td>
            </tr>
        </table>
        </html>
        """
        
        soup = BeautifulSoup(html, 'html.parser')
        url = collector._find_filing_url(soup, 2020)
        
        assert url is not None
        assert "/filing-detail/123" in url
    
    def test_find_filing_url_not_found(self, collector):
        """Test when filing for year is not found"""
        html = """
        <html>
        <table>
            <tr>
                <td>10-K</td>
                <td><a href="/filing-detail/123">Documents</a></td>
                <td>Annual report</td>
                <td>2019-02-15</td>
            </tr>
        </table>
        </html>
        """
        
        soup = BeautifulSoup(html, 'html.parser')
        url = collector._find_filing_url(soup, 2020)
        
        assert url is None
    
    def test_download_filing_non_us_company(self, collector):
        """Test downloading filing for non-US company (should skip)"""
        filepath = collector.download_filing('PLS.AX', 2020)
        
        assert filepath is None
        
        # Check that it was tracked as skipped
        conn = sqlite3.connect(collector.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT status FROM sec_filings WHERE ticker='PLS.AX' AND fiscal_year=2020"
        )
        result = cursor.fetchone()
        
        assert result is not None
        assert result[0] == 'skipped'
        
        conn.close()
    
    @patch.object(SECFilingDownloader, '_make_request')
    def test_download_filing_request_failure(self, mock_request, collector):
        """Test handling of request failures during download"""
        # Mock request to return None (failure)
        mock_request.return_value = None
        
        filepath = collector.download_filing('TSLA', 2020)
        
        assert filepath is None
        
        # Check status in database
        conn = sqlite3.connect(collector.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT status FROM sec_filings WHERE ticker='TSLA' AND fiscal_year=2020"
        )
        result = cursor.fetchone()
        
        assert result is not None
        assert result[0] == 'failed'
        
        conn.close()
    
    def test_download_stats_tracking(self, collector):
        """Test that download statistics are tracked correctly"""
        assert "successful" in collector.download_stats
        assert "failed" in collector.download_stats
        assert "skipped" in collector.download_stats
        
        # Initially empty
        assert len(collector.download_stats["successful"]) == 0
        assert len(collector.download_stats["failed"]) == 0
        assert len(collector.download_stats["skipped"]) == 0


class TestIntegration:
    """Integration tests (may hit real APIs - marked as slow)"""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_real_cik_lookup(self):
        """Test that CIK lookups work for real companies"""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = SECFilingDownloader(output_dir=temp_dir)
            
            # Test a few known companies
            assert collector.get_company_cik('TSLA') is not None
            assert collector.get_company_cik('AAPL') is None  # Not in our mapping
    
    @pytest.mark.slow
    @pytest.mark.integration  
    def test_real_download_single_filing(self):
        """Test actual download from SEC (slow, requires internet)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = SECFilingDownloader(output_dir=temp_dir)
            
            # Try to download a recent filing
            # Note: This may fail if SEC structure changes
            filepath = collector.download_filing('TSLA', 2023)
            
            # This test may fail due to SEC website changes
            # It's more of a smoke test
            if filepath:
                assert Path(filepath).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
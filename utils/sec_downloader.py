"""
SEC Filing Downloader for EV Supply Chain GAT Project

Downloads 10-K filings from SEC EDGAR and tracks them in a SQLite database.
Respects SEC rate limits (10 requests per second).

Author: EV Supply Chain GAT Team
Date: December 2025
"""

import time
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class SECFilingDownloader:
    """
    Downloads SEC 10-K filings from EDGAR.
    
    Features:
    - Rate limiting (10 requests/second for SEC compliance)
    - SQLite tracking of downloads
    - HTML parsing to extract clean text
    - Retry logic for failed downloads
    
    Attributes:
        output_dir (Path): Directory where filings will be saved
        db_path (Path): Path to SQLite database
        logger (logging.Logger): Logger instance
    """
    
    # SEC EDGAR base URL
    EDGAR_BASE_URL = "https://www.sec.gov"
    
    # Rate limit: SEC allows 10 requests per second
    RATE_LIMIT_DELAY = 0.11  # 110ms between requests (slightly conservative)
    
    # User agent (required by SEC)
    USER_AGENT = "EV Supply Chain Research bot/1.0 (research@university.edu)"
    
    def __init__(
        self,
        output_dir: str = "data/raw/sec_filings",
        db_name: str = "sec_filings.db"
    ):
        """
        Initialize the SEC Filing Downloader.
        
        Args:
            output_dir: Directory to save downloaded filings
            db_name: Name of SQLite database file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Database path
        self.db_path = Path("data/metadata") / db_name
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Initialize database
        self._init_database()
        
        # Track last request time for rate limiting
        self._last_request_time = 0
        
        # Download statistics
        self.download_stats = {
            "successful": [],
            "failed": [],
            "skipped": []
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Configure logging for this class."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if logger.handlers:
            logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.output_dir / "sec_downloader.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _init_database(self) -> None:
        """Initialize SQLite database for tracking filings."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sec_filings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                filing_type TEXT NOT NULL,
                fiscal_year INTEGER NOT NULL,
                filing_date TEXT,
                accession_number TEXT,
                url TEXT,
                local_path TEXT,
                download_date TEXT,
                file_size_kb INTEGER,
                status TEXT,
                error_message TEXT,
                UNIQUE(ticker, filing_type, fiscal_year)
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_ticker_year 
            ON sec_filings(ticker, fiscal_year)
        ''')
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Database initialized at {self.db_path}")
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting (10 requests per second)."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        
        if time_since_last_request < self.RATE_LIMIT_DELAY:
            sleep_time = self.RATE_LIMIT_DELAY - time_since_last_request
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _make_request(self, url: str, max_retries: int = 3) -> Optional[requests.Response]:
        """
        Make HTTP request with rate limiting and retries.
        
        Args:
            url: URL to request
            max_retries: Maximum number of retry attempts
        
        Returns:
            Response object or None if failed
        """
        headers = {
            'User-Agent': self.USER_AGENT,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
        
        for attempt in range(max_retries):
            try:
                # Enforce rate limit
                self._rate_limit()
                
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 404:
                    self.logger.warning(f"404 Not Found: {url}")
                    return None
                else:
                    self.logger.warning(
                        f"HTTP {response.status_code} for {url} "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    
            except Exception as e:
                self.logger.error(
                    f"Error requesting {url} (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
        
        return None
    
    def get_company_cik(self, ticker: str) -> Optional[str]:
        """
        Get CIK (Central Index Key) for a ticker symbol.
        
        Note: This is a simplified version. In production, you'd use
        the SEC company tickers JSON file.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            CIK string (10 digits, zero-padded) or None
        """
        # For MVP, we'll use a manual mapping for our 15 companies
        # In production, fetch from: https://www.sec.gov/files/company_tickers.json
        
        CIK_MAPPING = {
            'TSLA': '0001318605',
            'F': '0000037996',
            'GM': '0001467858',
            'RIVN': '0001874178',
            'PCRFY': '0000108312',  # Panasonic (may not have US 10-K)
            'ALB': '0000915913',
            'SQM': '0000101368',
            'LTHM': '0001602065',
            'LAC': '0001320658',
            'MP': '0001866671',
            'PLS.AX': None,  # Australian company, no SEC filings
            'MGA': '0000062996',
            'APTV': '0001521332',
        }
        
        cik = CIK_MAPPING.get(ticker)
        
        if cik is None:
            self.logger.warning(
                f"No CIK mapping found for {ticker}. "
                f"This company may not file with SEC."
            )
        
        return cik
    
    def download_filing(
        self,
        ticker: str,
        year: int,
        filing_type: str = "10-K"
    ) -> Optional[str]:
        """
        Download a single SEC filing.
        
        Args:
            ticker: Stock ticker symbol
            year: Fiscal year
            filing_type: Type of filing (default: "10-K")
        
        Returns:
            Path to downloaded file, or None if download failed
        """
        # Get CIK
        cik = self.get_company_cik(ticker)
        if cik is None:
            self._track_filing(
                ticker, filing_type, year,
                status="skipped",
                error_message="No CIK found (non-US company or not in mapping)"
            )
            return None
        
        self.logger.info(f"Downloading {filing_type} for {ticker} (CIK: {cik}) - Year {year}")
        
        # Search for filing
        # Note: This is simplified. In production, you'd use the SEC EDGAR API
        # to search for specific filings by date range
        
        # For MVP, we'll construct a simple search URL
        search_url = (
            f"{self.EDGAR_BASE_URL}/cgi-bin/browse-edgar?"
            f"action=getcompany&CIK={cik}&type={filing_type}&dateb=&owner=exclude&count=100"
        )
        
        # Make request
        response = self._make_request(search_url)
        
        if response is None:
            self._track_filing(
                ticker, filing_type, year,
                status="failed",
                error_message="Failed to fetch company filings page"
            )
            return None
        
        # Parse HTML to find filing for specific year
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find table with filings
        # Note: This is a simplified parser. Production code would be more robust
        filing_url = self._find_filing_url(soup, year)
        
        if filing_url is None:
            self._track_filing(
                ticker, filing_type, year,
                status="failed",
                error_message=f"No {filing_type} found for year {year}"
            )
            return None
        
        # Download the actual filing document
        full_url = f"{self.EDGAR_BASE_URL}{filing_url}"
        doc_response = self._make_request(full_url)
        
        if doc_response is None:
            self._track_filing(
                ticker, filing_type, year,
                status="failed",
                error_message="Failed to download filing document"
            )
            return None
        
        # Save to file
        ticker_dir = self.output_dir / ticker
        ticker_dir.mkdir(exist_ok=True)
        
        filename = f"{filing_type}_{year}.html"
        filepath = ticker_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(doc_response.text)
        
        file_size_kb = filepath.stat().st_size // 1024
        
        self.logger.info(
            f"Saved {ticker} {filing_type} {year} to {filepath} ({file_size_kb} KB)"
        )
        
        # Track in database
        self._track_filing(
            ticker, filing_type, year,
            url=full_url,
            local_path=str(filepath),
            file_size_kb=file_size_kb,
            status="downloaded"
        )
        
        return str(filepath)
    
    def _find_filing_url(self, soup: BeautifulSoup, year: int) -> Optional[str]:
        """
        Find the URL for a specific year's filing from the search results page.
        
        This is a simplified version for MVP. Production code would be more robust.
        
        Args:
            soup: BeautifulSoup object of the search results page
            year: Target fiscal year
        
        Returns:
            Relative URL to filing, or None if not found
        """
        # Find all rows in the filings table
        rows = soup.find_all('tr')
        
        for row in rows:
            cells = row.find_all('td')
            
            if len(cells) < 4:
                continue
            
            # Check if this row contains the filing date
            date_cell = cells[3].get_text(strip=True) if len(cells) > 3 else ""
            
            # Check if year matches
            if str(year) in date_cell:
                # Find the "Documents" link
                doc_link = cells[1].find('a', href=True)
                if doc_link:
                    # Get the filing detail page
                    detail_url = doc_link['href']
                    
                    # Now we need to get the actual HTML document
                    # For simplicity, we'll return the detail page URL
                    # and assume we can extract the document from there
                    return detail_url
        
        return None
    
    def _track_filing(
        self,
        ticker: str,
        filing_type: str,
        fiscal_year: int,
        filing_date: Optional[str] = None,
        accession_number: Optional[str] = None,
        url: Optional[str] = None,
        local_path: Optional[str] = None,
        file_size_kb: Optional[int] = None,
        status: str = "pending",
        error_message: Optional[str] = None
    ) -> None:
        """Track filing in SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO sec_filings 
            (ticker, filing_type, fiscal_year, filing_date, accession_number,
             url, local_path, download_date, file_size_kb, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticker, filing_type, fiscal_year, filing_date, accession_number,
            url, local_path, datetime.now().isoformat(), file_size_kb,
            status, error_message
        ))
        
        conn.commit()
        conn.close()
    
    def download_all_filings(
        self,
        ticker_list: List[str],
        year_range: Tuple[int, int],
        filing_type: str = "10-K"
    ) -> Dict[str, List[str]]:
        """
        Download filings for multiple tickers and years.
        
        Args:
            ticker_list: List of ticker symbols
            year_range: Tuple of (start_year, end_year) inclusive
            filing_type: Type of filing (default: "10-K")
        
        Returns:
            Dictionary mapping tickers to lists of downloaded file paths
        """
        results = {}
        start_year, end_year = year_range
        years = list(range(start_year, end_year + 1))
        
        total_downloads = len(ticker_list) * len(years)
        
        self.logger.info(
            f"Starting download of {filing_type} filings for {len(ticker_list)} tickers, "
            f"{len(years)} years ({total_downloads} total filings)"
        )
        
        # Reset stats
        self.download_stats = {"successful": [], "failed": [], "skipped": []}
        
        # Download with progress bar
        with tqdm(total=total_downloads, desc="Downloading SEC filings") as pbar:
            for ticker in ticker_list:
                results[ticker] = []
                
                for year in years:
                    filepath = self.download_filing(ticker, year, filing_type)
                    
                    if filepath:
                        results[ticker].append(filepath)
                        self.download_stats["successful"].append(f"{ticker}_{year}")
                    else:
                        # Check if skipped or failed
                        cik = self.get_company_cik(ticker)
                        if cik is None:
                            self.download_stats["skipped"].append(f"{ticker}_{year}")
                        else:
                            self.download_stats["failed"].append(f"{ticker}_{year}")
                    
                    pbar.update(1)
        
        # Print summary
        self._print_summary()
        
        return results
    
    def parse_html_to_text(self, filepath: str) -> str:
        """
        Parse HTML filing to extract plain text.
        
        Args:
            filepath: Path to HTML filing
        
        Returns:
            Extracted plain text
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style']):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error parsing {filepath}: {str(e)}")
            return ""
    
    def get_download_status(self) -> pd.DataFrame:
        """
        Get download status from database.
        
        Returns:
            DataFrame with all filing records
        """
        import pandas as pd
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM sec_filings", conn)
        conn.close()
        
        return df
    
    def _print_summary(self) -> None:
        """Print download summary statistics."""
        total = (
            len(self.download_stats["successful"]) +
            len(self.download_stats["failed"]) +
            len(self.download_stats["skipped"])
        )
        success_rate = (
            len(self.download_stats["successful"]) / total * 100
            if total > 0 else 0
        )
        
        print("\n" + "="*60)
        print("SEC FILING DOWNLOAD SUMMARY")
        print("="*60)
        print(f"Total attempted: {total}")
        print(f"Successful: {len(self.download_stats['successful'])} ({success_rate:.1f}%)")
        print(f"Failed: {len(self.download_stats['failed'])}")
        print(f"Skipped: {len(self.download_stats['skipped'])} (non-US companies)")
        print("="*60 + "\n")
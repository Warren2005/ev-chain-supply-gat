"""
Market Data Collector for EV Supply Chain GAT Project

This module provides functionality to download and validate stock market data
using yfinance. It handles errors gracefully, validates data quality, and
saves results in a structured format.

Author: EV Supply Chain GAT Team
Date: December 2025
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm


class MarketDataCollector:
    """
    Collects and validates stock market data from Yahoo Finance.
    
    Attributes:
        output_dir (Path): Directory where CSV files will be saved
        logger (logging.Logger): Logger instance for this class
    """
    
    def __init__(self, output_dir: str = "data/raw/market_data"):
        """
        Initialize the MarketDataCollector.
        
        Args:
            output_dir: Directory to save downloaded data (default: data/raw/market_data)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Track download statistics
        self.download_stats = {
            "successful": [],
            "failed": [],
            "warnings": []
        }
    
    def _setup_logger(self) -> logging.Logger:
        """
        Configure logging for this class.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.output_dir / "market_data_collector.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def download_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Download historical stock data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'TSLA')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            max_retries: Maximum number of download attempts (default: 3)
        
        Returns:
            DataFrame with OHLCV data, or None if download fails
        
        Raises:
            ValueError: If ticker is empty or dates are invalid
        """
        # Input validation
        if not ticker or not isinstance(ticker, str):
            raise ValueError(f"Invalid ticker: {ticker}")
        
        try:
            pd.to_datetime(start_date)
            pd.to_datetime(end_date)
        except Exception as e:
            raise ValueError(f"Invalid date format: {e}")
        
        # Download with retry logic
        for attempt in range(max_retries):
            try:
                self.logger.info(
                    f"Downloading {ticker} (attempt {attempt + 1}/{max_retries})"
                )
                
                # Download data
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                
                # Check if data was returned
                if df.empty:
                    self.logger.warning(f"No data returned for {ticker}")
                    if attempt < max_retries - 1:
                        continue
                    return None
                
                # Standardize column names (lowercase)
                df.columns = df.columns.str.lower()
                
                # Reset index to make date a column
                df.reset_index(inplace=True)
                # yfinance names the index 'Date', so rename it to lowercase 'date'
                if 'Date' in df.columns:
                    df.rename(columns={'Date': 'date'}, inplace=True)
                elif 'date' not in df.columns:
                    # If neither 'Date' nor 'date' exists, the first column should be the date
                    df.rename(columns={df.columns[0]: 'date'}, inplace=True)
                
                # Select and order relevant columns
                columns_to_keep = ['date', 'open', 'high', 'low', 'close', 'volume']
                
                # Check if all required columns exist
                missing_cols = set(columns_to_keep) - set(df.columns)
                if missing_cols:
                    self.logger.error(f"Missing columns for {ticker}: {missing_cols}")
                    return None
                
                df = df[columns_to_keep]
                
                self.logger.info(
                    f"Successfully downloaded {ticker}: {len(df)} rows"
                )
                return df
                
            except Exception as e:
                self.logger.error(
                    f"Error downloading {ticker} (attempt {attempt + 1}): {str(e)}"
                )
                if attempt == max_retries - 1:
                    return None
        
        return None
    
    def validate_data(
        self,
        df: pd.DataFrame,
        ticker: str,
        min_data_points: int = 100
    ) -> Tuple[bool, List[str]]:
        """
        Validate the quality of downloaded stock data.
        
        Args:
            df: DataFrame to validate
            ticker: Ticker symbol (for logging)
            min_data_points: Minimum required data points (default: 100)
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check 1: DataFrame is not None or empty
        if df is None or df.empty:
            issues.append("DataFrame is None or empty")
            return False, issues
        
        # Check 2: Minimum data points
        if len(df) < min_data_points:
            issues.append(
                f"Insufficient data: {len(df)} rows (minimum: {min_data_points})"
            )
        
        # Check 3: Required columns exist
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
            return False, issues
        
        # Check 4: No nulls in critical columns
        critical_cols = ['date', 'close']
        for col in critical_cols:
            null_count = df[col].isna().sum()
            if null_count > 0:
                issues.append(f"{col} has {null_count} null values")
        
        # Check 5: Prices are positive
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                negative_count = (df[col] <= 0).sum()
                issues.append(f"{col} has {negative_count} non-positive values")
        
        # Check 6: Volume is non-negative
        if (df['volume'] < 0).any():
            issues.append("Volume has negative values")
        
        # Check 7: High >= Low
        if (df['high'] < df['low']).any():
            issues.append("Some rows have high < low")
        
        # Check 8: Missing data percentage
        missing_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
        if missing_pct > 10:
            issues.append(f"High missing data: {missing_pct:.2f}%")
        
        # Determine if data is valid
        is_valid = len(issues) == 0
        
        if issues:
            self.logger.warning(f"Validation issues for {ticker}: {issues}")
        else:
            self.logger.info(f"Validation passed for {ticker}")
        
        return is_valid, issues
    
    def save_to_csv(self, df: pd.DataFrame, ticker: str) -> bool:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            ticker: Stock ticker (used for filename)
        
        Returns:
            True if save successful, False otherwise
        """
        try:
            filepath = self.output_dir / f"{ticker}_prices.csv"
            df.to_csv(filepath, index=False)
            self.logger.info(f"Saved {ticker} data to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving {ticker}: {str(e)}")
            return False
    
    def download_all_stocks(
        self,
        ticker_list: List[str],
        start_date: str,
        end_date: str,
        validate: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple stocks with progress tracking.
        
        Args:
            ticker_list: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            validate: Whether to validate data before saving (default: True)
        
        Returns:
            Dictionary mapping tickers to DataFrames (only successful downloads)
        """
        results = {}
        
        self.logger.info(
            f"Starting download for {len(ticker_list)} tickers "
            f"from {start_date} to {end_date}"
        )
        
        # Reset stats
        self.download_stats = {"successful": [], "failed": [], "warnings": []}
        
        # Download with progress bar
        for ticker in tqdm(ticker_list, desc="Downloading stocks"):
            df = self.download_stock_data(ticker, start_date, end_date)
            
            if df is not None:
                if validate:
                    is_valid, issues = self.validate_data(df, ticker)
                    
                    if is_valid:
                        self.save_to_csv(df, ticker)
                        results[ticker] = df
                        self.download_stats["successful"].append(ticker)
                    else:
                        self.download_stats["warnings"].append({
                            "ticker": ticker,
                            "issues": issues
                        })
                        # Still save but flag it
                        self.save_to_csv(df, ticker)
                        results[ticker] = df
                else:
                    self.save_to_csv(df, ticker)
                    results[ticker] = df
                    self.download_stats["successful"].append(ticker)
            else:
                self.download_stats["failed"].append(ticker)
        
        # Save download log
        self._save_download_log(ticker_list, start_date, end_date)
        
        # Print summary
        self._print_summary()
        
        return results
    
    def _save_download_log(
        self,
        ticker_list: List[str],
        start_date: str,
        end_date: str
    ) -> None:
        """Save download metadata to JSON file."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "date_range": {"start": start_date, "end": end_date},
            "total_tickers": len(ticker_list),
            "successful": self.download_stats["successful"],
            "failed": self.download_stats["failed"],
            "warnings": self.download_stats["warnings"]
        }
        
        log_file = self.output_dir / "download_log.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self.logger.info(f"Download log saved to {log_file}")
    
    def _print_summary(self) -> None:
        """Print download summary statistics."""
        total = (
            len(self.download_stats["successful"]) +
            len(self.download_stats["failed"])
        )
        success_rate = (
            len(self.download_stats["successful"]) / total * 100
            if total > 0 else 0
        )
        
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        print(f"Total tickers: {total}")
        print(f"Successful: {len(self.download_stats['successful'])} ({success_rate:.1f}%)")
        print(f"Failed: {len(self.download_stats['failed'])}")
        print(f"Warnings: {len(self.download_stats['warnings'])}")
        
        if self.download_stats['failed']:
            print(f"\nFailed tickers: {', '.join(self.download_stats['failed'])}")
        
        if self.download_stats['warnings']:
            print(f"\nWarnings issued for: {len(self.download_stats['warnings'])} tickers")
        
        print("="*60 + "\n")
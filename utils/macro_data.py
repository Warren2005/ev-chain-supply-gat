"""
Macro Data Collector for EV Supply Chain GAT Project

This module downloads macroeconomic indicators (VIX, Treasury yields, commodities)
that affect the entire supply chain. These features are shared across all nodes.

Author: EV Supply Chain GAT Team
Date: December 2025
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm


class MacroDataCollector:
    """
    Collects macroeconomic indicators from various sources.
    
    Features collected:
    - VIX (Volatility Index)
    - 10-Year Treasury Yield
    - Commodity prices (Lithium proxy, Copper)
    - US Dollar Index
    
    Attributes:
        output_dir (Path): Directory where data files will be saved
        logger (logging.Logger): Logger instance for this class
    """
    
    # Ticker symbols for macro indicators
    MACRO_TICKERS = {
        'vix': '^VIX',                    # CBOE Volatility Index
        'treasury_10y': '^TNX',           # 10-Year Treasury Yield
        'lithium_proxy': 'ALB',           # Albemarle as lithium proxy
        'copper': 'HG=F',                 # Copper Futures
        'dxy': 'DX-Y.NYB',               # US Dollar Index
    }
    
    def __init__(self, output_dir: str = "data/raw/macro_data"):
        """
        Initialize the MacroDataCollector.
        
        Args:
            output_dir: Directory to save downloaded data
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
        log_file = self.output_dir / "macro_data_collector.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def download_indicator(
        self,
        indicator_name: str,
        ticker: str,
        start_date: str,
        end_date: str,
        max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Download a single macroeconomic indicator.
        
        Args:
            indicator_name: Name of the indicator (e.g., 'vix', 'treasury_10y')
            ticker: Yahoo Finance ticker symbol
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            max_retries: Maximum number of download attempts
        
        Returns:
            DataFrame with date and indicator value, or None if download fails
        """
        for attempt in range(max_retries):
            try:
                self.logger.info(
                    f"Downloading {indicator_name} ({ticker}) - "
                    f"attempt {attempt + 1}/{max_retries}"
                )
                
                # Download data using Ticker().history() for consistency with market data
                stock = yf.Ticker(ticker)
                data = stock.history(start=start_date, end=end_date)
                
                # Check if data was returned
                if data.empty:
                    self.logger.warning(
                        f"No data returned for {indicator_name} ({ticker})"
                    )
                    if attempt < max_retries - 1:
                        continue
                    return None
                
                # Extract close price and create DataFrame
                # Reset index to get dates as a column
                data.reset_index(inplace=True)
                
                # Handle date column naming (could be 'Date' or 'index')
                if 'Date' in data.columns:
                    date_col = 'Date'
                elif 'index' in data.columns:
                    date_col = 'index'
                else:
                    date_col = data.columns[0]
                
                # Create clean DataFrame with just date and indicator value
                # Remove timezone info to keep dates clean
                df = pd.DataFrame({
                    'date': pd.to_datetime(data[date_col]).dt.tz_localize(None),
                    indicator_name: data['Close'].values
                })
                
                self.logger.info(
                    f"Successfully downloaded {indicator_name}: {len(df)} rows"
                )
                return df
                
            except Exception as e:
                self.logger.error(
                    f"Error downloading {indicator_name} (attempt {attempt + 1}): {str(e)}"
                )
                if attempt == max_retries - 1:
                    return None
        
        return None
    
    def download_all_indicators(
        self,
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Download all macro indicators.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        
        Returns:
            Dictionary mapping indicator names to DataFrames
        """
        results = {}
        
        self.logger.info(
            f"Starting download of {len(self.MACRO_TICKERS)} macro indicators "
            f"from {start_date} to {end_date}"
        )
        
        # Reset stats
        self.download_stats = {"successful": [], "failed": [], "warnings": []}
        
        # Download each indicator
        for indicator_name, ticker in tqdm(
            self.MACRO_TICKERS.items(),
            desc="Downloading macro indicators"
        ):
            df = self.download_indicator(
                indicator_name,
                ticker,
                start_date,
                end_date
            )
            
            if df is not None:
                results[indicator_name] = df
                self.download_stats["successful"].append(indicator_name)
                
                # Save individual indicator
                self._save_indicator(df, indicator_name)
            else:
                self.download_stats["failed"].append(indicator_name)
        
        # Print summary
        self._print_summary()
        
        return results
    
    def combine_indicators(
        self,
        indicators: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Combine all indicators into a single DataFrame.
        
        Args:
            indicators: Dictionary of indicator DataFrames
        
        Returns:
            Combined DataFrame with all indicators
        """
        if not indicators:
            self.logger.error("No indicators to combine")
            return pd.DataFrame()
        
        # Start with the first indicator
        first_indicator = list(indicators.keys())[0]
        combined = indicators[first_indicator].copy()
        
        # Merge all other indicators
        for indicator_name, df in list(indicators.items())[1:]:
            combined = pd.merge(
                combined,
                df,
                on='date',
                how='outer'
            )
        
        # Sort by date
        combined.sort_values('date', inplace=True)
        combined.reset_index(drop=True, inplace=True)
        
        self.logger.info(
            f"Combined {len(indicators)} indicators into single DataFrame: "
            f"{len(combined)} rows"
        )
        
        return combined
    
    def forward_fill_missing(
        self,
        df: pd.DataFrame,
        max_consecutive_nulls: int = 5
    ) -> pd.DataFrame:
        """
        Forward fill missing values in macro data.
        
        Macro indicators change slowly, so forward filling is acceptable.
        
        Args:
            df: DataFrame with potential missing values
            max_consecutive_nulls: Maximum consecutive nulls to fill
        
        Returns:
            DataFrame with missing values filled
        """
        df_filled = df.copy()
        
        # Forward fill, but limit consecutive fills
        for col in df_filled.columns:
            if col == 'date':
                continue
            
            # Forward fill
            df_filled[col] = df_filled[col].ffill(limit=max_consecutive_nulls)
        
        # Log remaining nulls
        null_counts = df_filled.isnull().sum()
        if null_counts.sum() > 0:
            self.logger.warning(
                f"Remaining null values after forward fill: {null_counts.to_dict()}"
            )
        
        return df_filled
    
    def align_with_dates(
        self,
        df: pd.DataFrame,
        target_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Align macro data with specific trading dates.
        
        Args:
            df: Macro indicators DataFrame
            target_dates: DatetimeIndex of target dates (e.g., from market data)
        
        Returns:
            DataFrame aligned to target dates
        """
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Create target DataFrame
        target_df = pd.DataFrame({'date': target_dates})
        
        # Merge and forward fill
        aligned = pd.merge(
            target_df,
            df,
            on='date',
            how='left'
        )
        
        # Forward fill missing values (macro data isn't available on all trading days)
        aligned = self.forward_fill_missing(aligned)
        
        self.logger.info(
            f"Aligned macro data to {len(aligned)} target dates"
        )
        
        return aligned
    
    def validate_data(
        self,
        df: pd.DataFrame,
        max_missing_pct: float = 10.0
    ) -> Tuple[bool, List[str]]:
        """
        Validate macro data quality.
        
        Args:
            df: DataFrame to validate
            max_missing_pct: Maximum acceptable missing data percentage
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check 1: DataFrame not empty
        if df is None or df.empty:
            issues.append("DataFrame is None or empty")
            return False, issues
        
        # Check 2: Date column exists
        if 'date' not in df.columns:
            issues.append("Missing 'date' column")
            return False, issues
        
        # Check 3: Check missing data percentage
        for col in df.columns:
            if col == 'date':
                continue
            
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            if missing_pct > max_missing_pct:
                issues.append(
                    f"{col} has {missing_pct:.2f}% missing data "
                    f"(threshold: {max_missing_pct}%)"
                )
        
        # Check 4: Reasonable value ranges
        if 'vix' in df.columns:
            if (df['vix'] < 0).any() or (df['vix'] > 100).any():
                issues.append("VIX has values outside reasonable range (0-100)")
        
        if 'treasury_10y' in df.columns:
            if (df['treasury_10y'] < -5).any() or (df['treasury_10y'] > 20).any():
                issues.append("Treasury yield has values outside reasonable range (-5% to 20%)")
        
        is_valid = len(issues) == 0
        
        if issues:
            self.logger.warning(f"Validation issues: {issues}")
        else:
            self.logger.info("Validation passed for macro data")
        
        return is_valid, issues
    
    def _save_indicator(self, df: pd.DataFrame, indicator_name: str) -> bool:
        """
        Save individual indicator to CSV.
        
        Args:
            df: DataFrame to save
            indicator_name: Name of the indicator
        
        Returns:
            True if save successful, False otherwise
        """
        try:
            filepath = self.output_dir / f"{indicator_name}.csv"
            df.to_csv(filepath, index=False)
            self.logger.info(f"Saved {indicator_name} to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving {indicator_name}: {str(e)}")
            return False
    
    def save_combined(
        self,
        df: pd.DataFrame,
        filename: str = "macro_indicators.csv"
    ) -> bool:
        """
        Save combined macro indicators to CSV.
        
        Args:
            df: Combined DataFrame
            filename: Output filename
        
        Returns:
            True if save successful, False otherwise
        """
        try:
            filepath = self.output_dir / filename
            df.to_csv(filepath, index=False)
            self.logger.info(f"Saved combined macro data to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving combined data: {str(e)}")
            return False
    
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
        print("MACRO DATA DOWNLOAD SUMMARY")
        print("="*60)
        print(f"Total indicators: {total}")
        print(f"Successful: {len(self.download_stats['successful'])} ({success_rate:.1f}%)")
        print(f"Failed: {len(self.download_stats['failed'])}")
        
        if self.download_stats['successful']:
            print(f"\nSuccessful: {', '.join(self.download_stats['successful'])}")
        
        if self.download_stats['failed']:
            print(f"\nFailed: {', '.join(self.download_stats['failed'])}")
        
        print("="*60 + "\n")
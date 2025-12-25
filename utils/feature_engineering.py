"""
Feature Engineering Module for EV Supply Chain GAT Project

This module calculates stock-specific and macro features for the GAT model.
Features are computed, validated, and saved in Parquet format for ML pipeline.

Stock-specific features (per node):
- Log returns
- Realized volatility (rolling)
- GARCH volatility (to be added)
- Volume shock
- Technical indicators

Author: EV Supply Chain GAT Team
Date: December 2025
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm
from arch import arch_model

from sklearn.preprocessing import StandardScaler
import pickle


class FeatureEngineer:
    """
    Calculates and validates features for the GAT model.
    
    This class handles:
    - Stock-specific feature calculation (returns, volatility, etc.)
    - Macro feature alignment
    - Feature normalization (train/val/test aware)
    - Data validation and quality checks
    
    Attributes:
        market_data_dir (Path): Directory containing raw market data
        macro_data_dir (Path): Directory containing macro indicators
        output_dir (Path): Directory for processed features
        logger (logging.Logger): Logger instance
    """
    
    # Feature calculation parameters
    VOLATILITY_WINDOW = 20  # Rolling window for realized volatility
    MIN_PERIODS = 10        # Minimum periods for rolling calculations
    
    # Date splits (from Phase 1 design)
    TRAIN_START = "2018-01-01"
    TRAIN_END = "2021-12-31"
    VAL_START = "2022-01-01"
    VAL_END = "2022-12-31"
    TEST_START = "2023-01-01"
    TEST_END = "2024-06-30"
    
    def __init__(
        self,
        market_data_dir: str = "data/raw/market_data",
        macro_data_dir: str = "data/raw/macro_data",
        output_dir: str = "data/processed"
    ):
        """
        Initialize the FeatureEngineer.
        
        Args:
            market_data_dir: Directory with raw market data CSVs
            macro_data_dir: Directory with macro indicator CSVs
            output_dir: Directory to save processed features
        """
        self.market_data_dir = Path(market_data_dir)
        self.macro_data_dir = Path(macro_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Track processing statistics
        self.processing_stats = {
            "stocks_processed": [],
            "stocks_failed": [],
            "features_calculated": []
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
        log_file = self.output_dir / "feature_engineering.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def load_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load market data for a single stock.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            DataFrame with OHLCV data, or None if loading fails
        """
        try:
            filepath = self.market_data_dir / f"{ticker}_prices.csv"
            
            if not filepath.exists():
                self.logger.error(f"File not found: {filepath}")
                return None
            
            df = pd.read_csv(filepath)
            
            # Ensure date column is datetime
            df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
            # Sort by date
            df.sort_values('date', inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            self.logger.info(f"Loaded {ticker}: {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {ticker}: {str(e)}")
            return None
        
    def load_macro_data(self) -> Optional[pd.DataFrame]:
        """
        Load combined macro indicators.
        
        Returns:
            DataFrame with macro features, or None if loading fails
        """
        try:
            filepath = self.macro_data_dir / "macro_indicators.csv"
            
            if not filepath.exists():
                self.logger.error(f"Macro data file not found: {filepath}")
                self.logger.error("Please run: python scripts/download_macro_data.py")
                return None
            
            df = pd.read_csv(filepath)
            
            # Ensure date column is datetime
            df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
            
            # Sort by date
            df.sort_values('date', inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            self.logger.info(f"Loaded macro data: {len(df)} rows, {len(df.columns)-1} indicators")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading macro data: {str(e)}")
            return None

    def align_macro_features(
        self,
        stock_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        max_forward_fill: int = 5
    ) -> pd.DataFrame:
        """
        Align macro features with stock trading dates.
        
        Macro indicators (VIX, yields) aren't published on all trading days,
        so we merge and forward-fill missing values.
        
        Args:
            stock_df: Stock features DataFrame with 'date' column
            macro_df: Macro indicators DataFrame with 'date' column
            max_forward_fill: Maximum days to forward fill (default: 5)
        
        Returns:
            DataFrame with stock and macro features combined
        """
        df = stock_df.copy()
        macro_copy = macro_df.copy()
        
        try:
            # CRITICAL FIX: Normalize dates to date-only (remove time component)
            df['date_only'] = df['date'].dt.normalize()
            macro_copy['date_only'] = macro_copy['date'].dt.normalize()
            
            # Merge on date_only instead of date
            merged = pd.merge(
                df,
                macro_copy.drop(columns=['date']),  # Drop original date, keep date_only
                on='date_only',
                how='left'
            )
            
            # Drop the temporary date_only column
            merged.drop(columns=['date_only'], inplace=True)
            
            # Get macro column names (everything except date and stock columns)
            stock_cols = set(stock_df.columns)
            macro_cols = [col for col in merged.columns if col not in stock_cols and col != 'date']
            
            # Forward fill macro values (they change slowly)
            # Limit forward fill to prevent stale data
            for col in macro_cols:
                merged[col] = merged[col].ffill(limit=max_forward_fill)
            
            self.logger.debug(
                f"Aligned macro features: {len(macro_cols)} indicators "
                f"across {len(merged)} dates"
            )
            
            # Check for remaining NaN in macro columns
            for col in macro_cols:
                nan_count = merged[col].isna().sum()
                if nan_count > 0:
                    self.logger.warning(
                        f"Macro feature '{col}' has {nan_count} NaN values after alignment"
                    )
            
            return merged
            
        except Exception as e:
            self.logger.error(f"Error aligning macro features: {str(e)}")
            return df
    
    def create_temporal_splits(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets temporally.
        
        This respects the time series nature - no random splitting.
        
        Args:
            df: Combined features DataFrame with 'date' column
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Split by date ranges
        train = df[
            (df['date'] >= self.TRAIN_START) & 
            (df['date'] <= self.TRAIN_END)
        ].copy()
        
        val = df[
            (df['date'] >= self.VAL_START) & 
            (df['date'] <= self.VAL_END)
        ].copy()
        
        test = df[
            (df['date'] >= self.TEST_START) & 
            (df['date'] <= self.TEST_END)
        ].copy()
        
        self.logger.info(
            f"Split data: Train={len(train)} rows, "
            f"Val={len(val)} rows, Test={len(test)} rows"
        )
        
        return train, val, test
    
    def fit_normalizer(
        self,
        train_df: pd.DataFrame,
        features_to_normalize: List[str] = None
    ) -> StandardScaler:
        """
        Fit StandardScaler on training data only.
        
        CRITICAL: This prevents data leakage by fitting only on train set.
        
        Args:
            train_df: Training data
            features_to_normalize: List of feature names to normalize
                If None, normalizes all numeric columns except date/ticker
        
        Returns:
            Fitted StandardScaler
        """
        # Default: normalize all numeric features except identifiers
        if features_to_normalize is None:
            exclude_cols = ['date', 'ticker', 'close', 'volume']
            features_to_normalize = [
                col for col in train_df.columns 
                if col not in exclude_cols and 
                pd.api.types.is_numeric_dtype(train_df[col])
            ]
        
        self.logger.info(f"Fitting normalizer on {len(features_to_normalize)} features")
        
        # Extract feature values from training data
        train_values = train_df[features_to_normalize].values
        
        # Fit scaler
        scaler = StandardScaler()
        scaler.fit(train_values)
        
        self.logger.info(
            f"Normalizer fitted on {len(train_df)} training samples"
        )
        
        # Store feature names for later reference
        scaler.feature_names_ = features_to_normalize
        
        return scaler
    
    def transform_features(
        self,
        df: pd.DataFrame,
        scaler: StandardScaler,
        suffix: str = "_norm"
    ) -> pd.DataFrame:
        """
        Normalize features using fitted scaler.
        
        Args:
            df: DataFrame to normalize
            scaler: Fitted StandardScaler
            suffix: Suffix to add to normalized column names (default: "_norm")
        
        Returns:
            DataFrame with normalized features added
        """
        result = df.copy()
        
        try:
            # Get feature names from scaler
            features = scaler.feature_names_
            
            # Extract values
            values = result[features].values
            
            # Transform
            normalized_values = scaler.transform(values)
            
            # Add normalized columns
            for i, feature in enumerate(features):
                result[f"{feature}{suffix}"] = normalized_values[:, i]
            
            self.logger.debug(
                f"Normalized {len(features)} features for {len(result)} rows"
            )
            
        except Exception as e:
            self.logger.error(f"Error normalizing features: {str(e)}")
        
        return result
    
    def save_normalizer(
        self,
        scaler: StandardScaler,
        filename: str = "feature_scaler.pkl"
    ) -> bool:
        """
        Save fitted scaler for later use.
        
        Args:
            scaler: Fitted StandardScaler
            filename: Output filename
        
        Returns:
            True if save successful
        """
        try:
            filepath = self.output_dir / filename
            
            with open(filepath, 'wb') as f:
                pickle.dump(scaler, f)
            
            self.logger.info(f"Saved normalizer to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving normalizer: {str(e)}")
            return False
    
    def load_normalizer(
        self,
        filename: str = "feature_scaler.pkl"
    ) -> Optional[StandardScaler]:
        """
        Load fitted scaler.
        
        Args:
            filename: Scaler filename
        
        Returns:
            Loaded StandardScaler, or None if loading fails
        """
        try:
            filepath = self.output_dir / filename
            
            if not filepath.exists():
                self.logger.error(f"Scaler file not found: {filepath}")
                return None
            
            with open(filepath, 'rb') as f:
                scaler = pickle.load(f)
            
            self.logger.info(f"Loaded normalizer from {filepath}")
            return scaler
            
        except Exception as e:
            self.logger.error(f"Error loading normalizer: {str(e)}")
            return None

    def process_and_normalize_all(
        self,
        ticker_list: List[str],
        save_splits: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Complete pipeline: process features, split, and normalize.
        
        This is the main method to prepare data for model training.
        
        Args:
            ticker_list: List of tickers to process
            save_splits: Whether to save train/val/test splits separately
        
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        # Step 1: Process all stock features
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: Processing stock features")
        self.logger.info("=" * 60)
        
        features_dict = self.process_all_stocks(ticker_list)
        
        if not features_dict:
            self.logger.error("No features processed. Aborting.")
            return {}
        
        # Step 2: Combine all stocks
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 2: Combining all stocks")
        self.logger.info("=" * 60)
        
        all_features = pd.concat(
            features_dict.values(),
            ignore_index=True
        )
        all_features.sort_values(['ticker', 'date'], inplace=True)
        
        self.logger.info(
            f"Combined {len(features_dict)} stocks: "
            f"{len(all_features)} total rows"
        )
        
        # Step 3: Create temporal splits
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 3: Creating temporal splits")
        self.logger.info("=" * 60)
        
        train_df, val_df, test_df = self.create_temporal_splits(all_features)
        
        # Step 4: Fit normalizer on training data
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 4: Fitting normalizer (train data only)")
        self.logger.info("=" * 60)
        
        scaler = self.fit_normalizer(train_df)
        self.save_normalizer(scaler)
        
        # Step 5: Normalize all splits
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 5: Normalizing all splits")
        self.logger.info("=" * 60)
        
        train_normalized = self.transform_features(train_df, scaler)
        val_normalized = self.transform_features(val_df, scaler)
        test_normalized = self.transform_features(test_df, scaler)
        
        # Step 6: Save splits
        if save_splits:
            self.logger.info("\n" + "=" * 60)
            self.logger.info("STEP 6: Saving normalized splits")
            self.logger.info("=" * 60)
            
            train_normalized.to_parquet(
                self.output_dir / "train_features.parquet",
                index=False
            )
            val_normalized.to_parquet(
                self.output_dir / "val_features.parquet",
                index=False
            )
            test_normalized.to_parquet(
                self.output_dir / "test_features.parquet",
                index=False
            )
            
            self.logger.info("Saved all splits to Parquet")
        
        # Print final summary
        self._print_normalization_summary(train_normalized, val_normalized, test_normalized)
        
        return {
            'train': train_normalized,
            'val': val_normalized,
            'test': test_normalized
        }
    
    def _print_normalization_summary(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """Print summary of normalized data."""
        print("\n" + "="*60)
        print("NORMALIZATION SUMMARY")
        print("="*60)
        
        # Count normalized features
        norm_cols = [col for col in train_df.columns if col.endswith('_norm')]
        
        print(f"Normalized features: {len(norm_cols)}")
        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_df)} rows ({train_df['ticker'].nunique()} stocks)")
        print(f"  Val:   {len(val_df)} rows ({val_df['ticker'].nunique()} stocks)")
        print(f"  Test:  {len(test_df)} rows ({test_df['ticker'].nunique()} stocks)")
        
        # Show sample statistics
        print(f"\nNormalized feature statistics (train set):")
        sample_features = norm_cols[:3]  # Show first 3
        print(train_df[sample_features].describe())
        
        print("="*60 + "\n")
        
            
    def calculate_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate log returns from close prices.
        
        Log returns are preferred over simple returns for:
        - Time-additivity
        - Better statistical properties
        - Symmetry around zero
        
        Args:
            df: DataFrame with 'close' column
        
        Returns:
            DataFrame with added 'log_return' column
        """
        df = df.copy()
        
        # Calculate log returns: ln(P_t / P_{t-1})
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # First row will be NaN (no previous price)
        self.logger.debug(f"Calculated log returns: {df['log_return'].notna().sum()} values")
        
        return df
    
    def calculate_realized_volatility(
        self,
        df: pd.DataFrame,
        window: int = None
    ) -> pd.DataFrame:
        """
        Calculate realized volatility using rolling standard deviation.
        
        Realized volatility is the rolling standard deviation of log returns,
        annualized to match market conventions.
        
        Args:
            df: DataFrame with 'log_return' column
            window: Rolling window size (default: VOLATILITY_WINDOW)
        
        Returns:
            DataFrame with added 'realized_vol' column
        """
        df = df.copy()
        window = window or self.VOLATILITY_WINDOW
        
        # Calculate rolling std of log returns
        # min_periods allows calculation with fewer data points initially
        rolling_std = df['log_return'].rolling(
            window=window,
            min_periods=self.MIN_PERIODS
        ).std()
        
        # Annualize: multiply by sqrt(252 trading days)
        df['realized_vol'] = rolling_std * np.sqrt(252)
        
        self.logger.debug(
            f"Calculated realized volatility: "
            f"{df['realized_vol'].notna().sum()} values (window={window})"
        )
        
        return df
    
    def calculate_garch_volatility(
        self,
        df: pd.DataFrame,
        p: int = 1,
        q: int = 1
    ) -> pd.DataFrame:
        """
        Calculate GARCH conditional volatility.
        
        GARCH(p,q) models time-varying volatility using:
        - p: autoregressive terms (past variances)
        - q: moving average terms (past shocks)
        
        GARCH(1,1) is the industry standard and captures volatility clustering.
        
        Args:
            df: DataFrame with 'log_return' column
            p: GARCH autoregressive order (default: 1)
            q: GARCH moving average order (default: 1)
        
        Returns:
            DataFrame with added 'garch_vol' column
        """
        df = df.copy()
        
        try:
            # Prepare returns (remove NaN, convert to percentages for stability)
            returns = df['log_return'].dropna() * 100
            
            # Need at least 100 observations for reliable GARCH estimation
            if len(returns) < 100:
                self.logger.warning(
                    f"Insufficient data for GARCH ({len(returns)} points). "
                    "Filling with NaN."
                )
                df['garch_vol'] = np.nan
                return df
            
            # Fit GARCH model
            # Suppress arch library warnings about convergence
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                model = arch_model(
                    returns,
                    vol='Garch',
                    p=p,
                    q=q,
                    rescale=False
                )
                
                # Fit with limited iterations to prevent hanging
                fitted_model = model.fit(disp='off', options={'maxiter': 100})
            
            # Extract conditional volatility
            # Convert back from percentage scale and annualize
            conditional_vol = fitted_model.conditional_volatility / 100
            conditional_vol_annualized = conditional_vol * np.sqrt(252)
            
            # Align with original dataframe (accounting for NaN in returns)
            df['garch_vol'] = np.nan
            
            # Find indices where we have returns
            valid_indices = df['log_return'].notna()
            
            # Assign GARCH values (starts from index 0 of conditional_vol)
            df.loc[valid_indices, 'garch_vol'] = conditional_vol_annualized.values
            
            self.logger.debug(
                f"Calculated GARCH({p},{q}) volatility: "
                f"{df['garch_vol'].notna().sum()} values"
            )
            
        except Exception as e:
            self.logger.warning(f"GARCH estimation failed: {str(e)}. Filling with NaN.")
            df['garch_vol'] = np.nan
        
        return df
    
    def calculate_rsi(
        self,
        df: pd.DataFrame,
        window: int = 14
    ) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI measures momentum and identifies overbought/oversold conditions.
        Values range from 0-100:
        - RSI > 70: Overbought (potential reversal down)
        - RSI < 30: Oversold (potential reversal up)
        
        Args:
            df: DataFrame with 'close' column
            window: RSI period (default: 14, industry standard)
        
        Returns:
            DataFrame with added 'rsi' column
        """
        df = df.copy()
        
        try:
            # Calculate price changes
            delta = df['close'].diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate rolling average of gains and losses
            avg_gains = gains.rolling(
                window=window,
                min_periods=self.MIN_PERIODS
            ).mean()
            
            avg_losses = losses.rolling(
                window=window,
                min_periods=self.MIN_PERIODS
            ).mean()
            
            # Calculate Relative Strength (RS)
            rs = avg_gains / avg_losses
            
            # Calculate RSI: 100 - (100 / (1 + RS))
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Handle division by zero (when avg_losses = 0, RSI = 100)
            df['rsi'] = df['rsi'].fillna(100)
            
            self.logger.debug(
                f"Calculated RSI: "
                f"{df['rsi'].notna().sum()} values (window={window})"
            )
            
        except Exception as e:
            self.logger.warning(f"RSI calculation failed: {str(e)}")
            df['rsi'] = np.nan
        
        return df
    

    def calculate_volume_shock(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate volume shock as z-score of volume.
        
        Volume shock identifies unusual trading activity, which often
        signals information flow or liquidity events that affect volatility.
        
        Args:
            df: DataFrame with 'volume' column
            window: Rolling window for mean/std calculation (default: 20)
        
        Returns:
            DataFrame with added 'volume_shock' column
        """
        df = df.copy()
        
        try:
            # Calculate rolling mean and std of volume
            rolling_mean = df['volume'].rolling(
                window=window,
                min_periods=self.MIN_PERIODS
            ).mean()
            
            rolling_std = df['volume'].rolling(
                window=window,
                min_periods=self.MIN_PERIODS
            ).std()
            
            # Calculate z-score: (volume - mean) / std
            # This tells us how many standard deviations above/below normal
            df['volume_shock'] = (df['volume'] - rolling_mean) / rolling_std
            
            # Replace infinite values with NaN (can happen if std = 0)
            df['volume_shock'] = df['volume_shock'].replace([np.inf, -np.inf], np.nan)
            
            self.logger.debug(
                f"Calculated volume shock: "
                f"{df['volume_shock'].notna().sum()} values (window={window})"
            )
            
        except Exception as e:
            self.logger.warning(f"Volume shock calculation failed: {str(e)}")
            df['volume_shock'] = np.nan
        
        return df
    
    def calculate_basic_features(
        self,
        ticker: str,
        include_macro: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Calculate basic stock-specific features for a single ticker.
        
        This method orchestrates the calculation of:
        - Log returns
        - Realized volatility
        - GARCH volatility
        - Volume shock
        - RSI
        - Macro features (optional)
        
        Args:
            ticker: Stock ticker symbol
            include_macro: Whether to include macro features (default: True)
        
        Returns:
            DataFrame with calculated features, or None if calculation fails
        """
        try:
            self.logger.info(f"Calculating features for {ticker}")
            
            # Load data
            df = self.load_stock_data(ticker)
            if df is None:
                return None
            
            # Validate required columns
            required_cols = ['date', 'close', 'volume']
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                self.logger.error(f"{ticker}: Missing columns {missing_cols}")
                return None
            
            # Calculate stock-specific features
            df = self.calculate_log_returns(df)
            df = self.calculate_realized_volatility(df)
            df = self.calculate_garch_volatility(df)
            df = self.calculate_volume_shock(df)
            df = self.calculate_rsi(df)
            
            # Add ticker column for tracking
            df['ticker'] = ticker
            
            # Align with macro features if requested
            if include_macro:
                macro_df = self.load_macro_data()
                if macro_df is not None:
                    df = self.align_macro_features(df, macro_df)
                else:
                    self.logger.warning(
                        f"Macro data not available for {ticker}. "
                        "Proceeding with stock features only."
                    )
            
            # Select relevant columns (stock + macro if available)
            base_cols = [
                'date', 'ticker', 'close', 'volume',
                'log_return', 'realized_vol', 'garch_vol',
                'volume_shock', 'rsi'
            ]
            
            # Add any macro columns that exist
            macro_cols = [col for col in df.columns if col not in base_cols]
            feature_cols = base_cols + macro_cols
            
            df = df[feature_cols]
            
            self.logger.info(
                f"Successfully calculated features for {ticker}: "
                f"{len(df)} rows, {len(feature_cols)} columns "
                f"({len(macro_cols)} macro features)"
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating features for {ticker}: {str(e)}")
            return None
    
    def process_all_stocks(
        self,
        ticker_list: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Process features for multiple stocks.
        
        Args:
            ticker_list: List of ticker symbols to process
        
        Returns:
            Dictionary mapping tickers to feature DataFrames
        """
        results = {}
        
        self.logger.info(
            f"Starting feature calculation for {len(ticker_list)} stocks"
        )
        
        # Reset stats
        self.processing_stats = {
            "stocks_processed": [],
            "stocks_failed": [],
            "features_calculated": [
                'log_return', 'realized_vol', 'garch_vol', 
                'volume_shock', 'rsi', 'macro_features'
            ]
        }
        
        # Process each stock with progress bar
        for ticker in tqdm(ticker_list, desc="Processing stocks"):
            df = self.calculate_basic_features(ticker)
            
            if df is not None:
                results[ticker] = df
                self.processing_stats["stocks_processed"].append(ticker)
            else:
                self.processing_stats["stocks_failed"].append(ticker)
        
        # Print summary
        self._print_summary()
        
        return results
    
    def validate_features(
        self,
        df: pd.DataFrame,
        ticker: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate feature quality for a stock.
        
        Args:
            df: DataFrame with calculated features
            ticker: Stock ticker (for logging)
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check 1: DataFrame not empty
        if df is None or df.empty:
            issues.append("DataFrame is None or empty")
            return False, issues
        
        # Check 2: Required columns exist
        required_cols = ['date', 'ticker', 'log_return', 'realized_vol']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
            return False, issues
        
        # Check 3: Check for excessive NaN values
        for col in ['log_return', 'realized_vol', 'garch_vol', 'volume_shock', 'rsi']:
            nan_pct = (df[col].isna().sum() / len(df)) * 100
            if nan_pct > 50:  # More than 50% NaN is problematic
                issues.append(f"{col} has {nan_pct:.1f}% NaN values")
        
        # Check 4: Check for infinite values
        for col in ['log_return', 'realized_vol', 'garch_vol', 'volume_shock', 'rsi']:
            if np.isinf(df[col]).any():
                issues.append(f"{col} contains infinite values")
        
        # Check 5: Volatility features should be positive
        for vol_col in ['realized_vol', 'garch_vol']:
            if vol_col in df.columns:
                if (df[vol_col] < 0).any():
                    issues.append(f"{vol_col} has negative values")
        
        # Check 6: RSI should be in valid range (0-100)
        if 'rsi' in df.columns:
            rsi_valid = df['rsi'].dropna()
            if len(rsi_valid) > 0:
                if (rsi_valid < 0).any() or (rsi_valid > 100).any():
                    issues.append("RSI has values outside valid range (0-100)")
        
        # Check 7: Log returns should be reasonable (-50% to +50% per day)
        if 'log_return' in df.columns:
            extreme_returns = df['log_return'].abs() > 0.5
            if extreme_returns.any():
                n_extreme = extreme_returns.sum()
                issues.append(
                    f"Found {n_extreme} extreme log returns (>50% daily change)"
                )
        
        is_valid = len(issues) == 0
        
        if issues:
            self.logger.warning(f"Validation issues for {ticker}: {issues}")
        else:
            self.logger.info(f"Validation passed for {ticker}")
        
        return is_valid, issues
    
    def save_features(
        self,
        features_dict: Dict[str, pd.DataFrame],
        filename: str = "stock_features.parquet"
    ) -> bool:
        """
        Save all features to a single Parquet file.
        
        Args:
            features_dict: Dictionary mapping tickers to DataFrames
            filename: Output filename
        
        Returns:
            True if save successful, False otherwise
        """
        try:
            # Combine all DataFrames
            all_features = pd.concat(
                features_dict.values(),
                ignore_index=True
            )
            
            # Sort by ticker and date
            all_features.sort_values(['ticker', 'date'], inplace=True)
            all_features.reset_index(drop=True, inplace=True)
            
            # Save to Parquet
            filepath = self.output_dir / filename
            all_features.to_parquet(filepath, index=False)
            
            self.logger.info(
                f"Saved features to {filepath}: "
                f"{len(all_features)} rows, "
                f"{len(all_features['ticker'].unique())} stocks"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving features: {str(e)}")
            return False
    
    def _print_summary(self) -> None:
        """Print processing summary statistics."""
        total = (
            len(self.processing_stats["stocks_processed"]) +
            len(self.processing_stats["stocks_failed"])
        )
        success_rate = (
            len(self.processing_stats["stocks_processed"]) / total * 100
            if total > 0 else 0
        )
        
        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)
        print(f"Total stocks: {total}")
        print(f"Successful: {len(self.processing_stats['stocks_processed'])} ({success_rate:.1f}%)")
        print(f"Failed: {len(self.processing_stats['stocks_failed'])}")
        
        if self.processing_stats['features_calculated']:
            print(f"\nFeatures calculated: {', '.join(self.processing_stats['features_calculated'])}")
        
        if self.processing_stats['stocks_failed']:
            print(f"\nFailed stocks: {', '.join(self.processing_stats['stocks_failed'])}")
        
        print("="*60 + "\n")
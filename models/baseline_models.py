"""
Baseline Models for Comparison with ST-GAT

Implements traditional and simpler models:
1. GARCH(1,1) - Traditional volatility forecasting
2. VAR - Vector AutoRegression (captures correlations but not graph)
3. Simple LSTM - Temporal model without graph structure
4. Persistence Model - Naive baseline (predict last value)

Author: EV Supply Chain GAT Team
Date: December 2024
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from arch import arch_model
from statsmodels.tsa.api import VAR as StatsVAR
import warnings
warnings.filterwarnings('ignore')


class GARCHBaseline:
    """
    GARCH(1,1) baseline for each stock independently.
    
    Traditional volatility forecasting model that captures:
    - Volatility clustering
    - Mean reversion
    - Leverage effects
    
    Does NOT capture:
    - Cross-stock dependencies
    - Supply chain relationships
    - Multi-stock dynamics
    """
    
    def __init__(self, stock_names):
        """
        Initialize GARCH models for each stock.
        
        Args:
            stock_names: List of stock tickers
        """
        self.stock_names = stock_names
        self.models = {}  # Store fitted GARCH models per stock
        self.logger = self._setup_logger()
        
        self.logger.info(f"GARCHBaseline initialized for {len(stock_names)} stocks")
    
    def _setup_logger(self):
        logger = logging.getLogger(f"{__name__}.GARCHBaseline")
        logger.setLevel(logging.INFO)
        if logger.handlers:
            logger.handlers.clear()
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def fit(self, train_targets: np.ndarray):
        """
        Fit GARCH(1,1) for each stock.
        
        Args:
            train_targets: Training volatility [num_samples, num_stocks]
        """
        self.logger.info("Fitting GARCH(1,1) models...")
        
        num_stocks = train_targets.shape[1]
        
        for stock_idx in range(num_stocks):
            stock = self.stock_names[stock_idx]
            y = train_targets[:, stock_idx] * 100  # Scale for GARCH stability
            
            try:
                # Fit GARCH(1,1)
                model = arch_model(y, vol='Garch', p=1, q=1, rescale=False)
                fitted = model.fit(disp='off', show_warning=False)
                self.models[stock] = fitted
                
                self.logger.info(f"  ✓ {stock}: GARCH fitted successfully")
            except Exception as e:
                self.logger.warning(f"  ✗ {stock}: GARCH fit failed: {e}")
                self.models[stock] = None
        
        self.logger.info("GARCH fitting complete")
    
    def predict(self, horizon: int = 1) -> np.ndarray:
        """
        Forecast volatility for next period.
        
        Args:
            horizon: Forecast horizon (default: 1)
        
        Returns:
            Predictions [num_stocks]
        """
        predictions = []
        
        for stock in self.stock_names:
            if self.models[stock] is not None:
                try:
                    forecast = self.models[stock].forecast(horizon=horizon)
                    pred = np.sqrt(forecast.variance.values[-1, 0]) / 100  # Rescale
                    predictions.append(pred)
                except:
                    predictions.append(0.0)  # Fallback
            else:
                predictions.append(0.0)
        
        return np.array(predictions)
    
    def evaluate(self, test_targets: np.ndarray) -> np.ndarray:
        """
        Rolling forecast on test set.
        
        Args:
            test_targets: Test volatility [num_samples, num_stocks]
        
        Returns:
            Predictions [num_samples, num_stocks]
        """
        num_samples, num_stocks = test_targets.shape
        predictions = np.zeros_like(test_targets)
        
        # Use last training prediction for all test samples
        # (GARCH doesn't do multi-step easily without refitting)
        for i in range(num_samples):
            predictions[i] = self.predict(horizon=1)
        
        return predictions


class VARBaseline:
    """
    Vector AutoRegression baseline.
    
    Captures:
    - Temporal dependencies
    - Cross-stock correlations
    - Linear dynamics
    
    Does NOT capture:
    - Non-linear relationships
    - Graph structure
    - Attention mechanisms
    """
    
    def __init__(self, stock_names, lag_order: int = 5):
        """
        Initialize VAR model.
        
        Args:
            stock_names: List of stock tickers
            lag_order: Number of lags to use
        """
        self.stock_names = stock_names
        self.lag_order = lag_order
        self.model = None
        self.fitted_model = None
        self.logger = self._setup_logger()
        
        self.logger.info(f"VARBaseline initialized: {len(stock_names)} stocks, lag={lag_order}")
    
    def _setup_logger(self):
        logger = logging.getLogger(f"{__name__}.VARBaseline")
        logger.setLevel(logging.INFO)
        if logger.handlers:
            logger.handlers.clear()
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def fit(self, train_targets: np.ndarray):
        """
        Fit VAR model.
        
        Args:
            train_targets: Training volatility [num_samples, num_stocks]
        """
        self.logger.info(f"Fitting VAR({self.lag_order}) model...")
        
        try:
            self.model = StatsVAR(train_targets)
            self.fitted_model = self.model.fit(maxlags=self.lag_order, ic='aic')
            
            self.logger.info(f"  ✓ VAR fitted with {self.fitted_model.k_ar} lags")
        except Exception as e:
            self.logger.error(f"  ✗ VAR fit failed: {e}")
            self.fitted_model = None
    
    def predict(self, last_obs: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Forecast next step.
        
        Args:
            last_obs: Last observations [lag_order, num_stocks]
            steps: Forecast steps
        
        Returns:
            Prediction [num_stocks]
        """
        if self.fitted_model is None:
            return np.zeros(len(self.stock_names))
        
        try:
            forecast = self.fitted_model.forecast(last_obs, steps=steps)
            return forecast[-1]  # Last step
        except:
            return np.zeros(len(self.stock_names))
    
    def evaluate(self, test_targets: np.ndarray, train_targets: np.ndarray) -> np.ndarray:
        """
        Rolling forecast on test set.
        
        Args:
            test_targets: Test volatility [num_samples, num_stocks]
            train_targets: Training volatility (for initialization)
        
        Returns:
            Predictions [num_samples, num_stocks]
        """
        num_samples, num_stocks = test_targets.shape
        predictions = np.zeros_like(test_targets)
        
        # Initialize with last training observations
        history = train_targets[-self.lag_order:].copy()
        
        for i in range(num_samples):
            # Predict
            pred = self.predict(history, steps=1)
            predictions[i] = pred
            
            # Update history (rolling window)
            if i < num_samples - 1:
                history = np.vstack([history[1:], test_targets[i]])
        
        return predictions


class SimpleLSTM(nn.Module):
    """
    Simple LSTM baseline (no graph structure).
    
    Captures:
    - Temporal dependencies
    - Non-linear dynamics
    - Individual stock patterns
    
    Does NOT capture:
    - Graph structure
    - Attention mechanisms
    - Supply chain relationships
    """
    
    def __init__(
        self,
        num_stocks: int,
        input_features: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2
    ):
        """
        Initialize Simple LSTM.
        
        Args:
            num_stocks: Number of stocks
            input_features: Input features per stock
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(SimpleLSTM, self).__init__()
        
        self.num_stocks = num_stocks
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        
        # Separate LSTM for each stock (no cross-stock information)
        self.lstms = nn.ModuleList([
            nn.LSTM(
                input_size=input_features,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            for _ in range(num_stocks)
        ])
        
        # Output layers
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, 1)
            for _ in range(num_stocks)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: [batch, num_stocks, seq_len, input_features]
        
        Returns:
            predictions: [batch, num_stocks, 1]
        """
        batch_size, num_stocks, seq_len, feat_dim = features.shape
        
        predictions = []
        
        for stock_idx in range(num_stocks):
            # Get stock sequence: [batch, seq_len, input_features]
            stock_seq = features[:, stock_idx, :, :]
            
            # LSTM
            lstm_out, _ = self.lstms[stock_idx](stock_seq)
            
            # Last timestep: [batch, hidden_dim]
            last_hidden = lstm_out[:, -1, :]
            
            # Dropout
            last_hidden = self.dropout(last_hidden)
            
            # Prediction
            pred = self.output_layers[stock_idx](last_hidden)  # [batch, 1]
            predictions.append(pred)
        
        # Stack: [batch, num_stocks, 1]
        predictions = torch.stack(predictions, dim=1)
        
        return predictions


class PersistenceBaseline:
    """
    Naive persistence model: predict last observed value.
    
    Simple baseline that assumes volatility doesn't change.
    Often surprisingly competitive for short-term forecasts.
    """
    
    def __init__(self, stock_names):
        self.stock_names = stock_names
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logger = logging.getLogger(f"{__name__}.PersistenceBaseline")
        logger.setLevel(logging.INFO)
        if logger.handlers:
            logger.handlers.clear()
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def fit(self, train_targets: np.ndarray):
        """No fitting needed for persistence model"""
        self.logger.info("Persistence model initialized (no training needed)")
    
    def evaluate(self, test_targets: np.ndarray, last_train_value: np.ndarray) -> np.ndarray:
        """
        Predict: just repeat last known value.
        
        Args:
            test_targets: Test targets [num_samples, num_stocks]
            last_train_value: Last training value [num_stocks]
        
        Returns:
            Predictions [num_samples, num_stocks]
        """
        num_samples = test_targets.shape[0]
        
        # For first prediction, use last training value
        # For subsequent, use previous actual value
        predictions = np.zeros_like(test_targets)
        predictions[0] = last_train_value
        
        for i in range(1, num_samples):
            predictions[i] = test_targets[i - 1]
        
        return predictions
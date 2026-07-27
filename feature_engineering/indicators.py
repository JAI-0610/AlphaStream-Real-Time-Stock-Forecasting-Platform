import pandas as pd
import numpy as np
from typing import Tuple

def calculate_log_returns(series: pd.Series) -> pd.Series:
    """Computes log returns of a price series."""
    return np.log(series / series.shift(1))

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Computes standard Relative Strength Index (RSI) using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder's exponential moving average
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Computes MACD line, Signal line, and MACD Histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Computes upper band, middle band, and lower band of Bollinger Bands."""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    return upper, middle, lower

def calculate_volume_shocks(volume: pd.Series, period: int = 20) -> pd.Series:
    """Measures current volume compared to its rolling average."""
    avg_vol = volume.rolling(window=period).mean()
    return volume / (avg_vol + 1e-9)

def calculate_market_regime(returns: pd.Series, window: int = 20) -> pd.Series:
    """Identifies high vs low volatility regimes based on rolling return standard deviation."""
    rolling_vol = returns.rolling(window=window).std()
    historical_avg_vol = rolling_vol.expanding(min_periods=window).mean()
    # 1 for high volatility, 0 for low volatility
    regime = (rolling_vol > historical_avg_vol).astype(int)
    return regime

def calculate_rolling_correlations(ticker_returns: pd.Series, etf_returns: pd.Series, window: int = 20) -> pd.Series:
    """Calculates rolling correlation between a stock and a benchmark ETF."""
    return ticker_returns.rolling(window=window).corr(etf_returns)

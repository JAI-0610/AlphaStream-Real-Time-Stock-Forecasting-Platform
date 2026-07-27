import os
import yaml
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from feature_engineering.indicators import (
    calculate_log_returns,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_volume_shocks,
    calculate_market_regime,
    calculate_rolling_correlations
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("feature_pipeline")

class FeaturePipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.tickers: List[str] = self.config["data"]["tickers"]
        self.sector_etfs: List[str] = self.config["data"]["sector_etfs"]
        self.forecast_horizon: int = self.config["data"]["forecast_horizon"]
        self.scaler_type: str = self.config["features"]["scaler_type"]
        self.cache_dir: str = self.config["data"]["cache_dir"]
        
        self.scalers: Dict[str, Dict[str, Tuple[float, float]]] = {} # Ticker -> Feature -> (center, scale)

    def extract_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes technical indicators and time features for a single asset dataframe.
        Ensures all features are causal (no future values are referenced).
        """
        df = df.copy().sort_values(by="timestamp").reset_index(drop=True)
        
        # Log return
        df["log_return"] = calculate_log_returns(df["close"])
        
        # Rolling returns
        for w in self.config["features"]["rolling_windows"]:
            df[f"return_{w}"] = df["close"].pct_change(periods=w)
            df[f"volatility_{w}"] = df["log_return"].rolling(window=w).std()
            df[f"sma_{w}"] = df["close"].rolling(window=w).mean() / df["close"] - 1.0 # Normalized SMA
            
        # RSI
        df["rsi"] = calculate_rsi(df["close"], period=self.config["features"]["rsi_period"])
        
        # MACD
        macd, macd_sig, macd_hist = calculate_macd(
            df["close"],
            fast=self.config["features"]["macd_fast"],
            slow=self.config["features"]["macd_slow"],
            signal=self.config["features"]["macd_signal"]
        )
        df["macd"] = macd / df["close"] # Normalized
        df["macd_signal"] = macd_sig / df["close"]
        df["macd_hist"] = macd_hist / df["close"]
        
        # Bollinger Bands
        upper, middle, lower = calculate_bollinger_bands(
            df["close"],
            period=self.config["features"]["bollinger_period"],
            num_std=self.config["features"]["bollinger_std"]
        )
        df["bollinger_upper_pct"] = upper / df["close"] - 1.0
        df["bollinger_lower_pct"] = lower / df["close"] - 1.0
        
        # Volume Shock
        df["volume_shock"] = calculate_volume_shocks(df["volume"], period=20)
        
        # Time features
        df["day_of_week"] = df["timestamp"].dt.dayofweek / 6.0
        df["month"] = (df["timestamp"].dt.month - 1) / 11.0
        
        # Volatility regime
        df["regime"] = calculate_market_regime(
            df["log_return"], 
            window=self.config["features"]["regime_volatility_window"]
        )
        
        # Lags of returns (features at t-1, t-2)
        df["lag_return_1"] = df["log_return"].shift(1)
        df["lag_return_2"] = df["log_return"].shift(2)
        
        return df

    def compute_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes forecast targets. Returns forward returns and direction labels.
        CAUTION: These contain future leakage and should ONLY be used during training/testing
        as Y targets, never as input features (X).
        """
        df = df.copy()
        h = self.forecast_horizon
        
        # Future return over horizon h: (price_{t+h} - price_t) / price_t
        df["target_return"] = df["close"].shift(-h) / df["close"] - 1.0
        
        # Class label: 1 if target_return > 0 else 0
        df["target_direction"] = (df["target_return"] > 0).astype(int)
        
        return df

    def build_multivariate_dataset(self, dfs: Dict[str, pd.DataFrame], is_training: bool = True) -> Dict[str, pd.DataFrame]:
        """Merges sector/etf features, handles scaling, and produces clean feature dataframes."""
        # First process all ETF data to have clean return features
        etf_features = {}
        for etf in self.sector_etfs:
            if etf in dfs:
                processed_etf = self.extract_base_features(dfs[etf])
                # We only need its returns for cross-asset correlations
                etf_features[etf] = processed_etf[["timestamp", "log_return"]].rename(
                    columns={"log_return": f"{etf}_log_return"}
                )
        
        processed_tickers = {}
        target_tickers = [t for t in dfs.keys() if t not in self.sector_etfs]
        for ticker in target_tickers:
            
            df = self.extract_base_features(dfs[ticker])
            
            # Merge ETF features and calculate correlations
            for etf, etf_df in etf_features.items():
                df = pd.merge(df, etf_df, on="timestamp", how="left")
                # Fill missing ETF returns
                df[f"{etf}_log_return"] = df[f"{etf}_log_return"].ffill().bfill().fillna(0.0)
                # Calculate rolling correlation
                df[f"corr_{etf}"] = calculate_rolling_correlations(
                    df["log_return"], 
                    df[f"{etf}_log_return"], 
                    window=20
                ).fillna(0.0)
                
            # Drop cleanups
            df = df.drop(columns=[f"{etf}_log_return" for etf in self.sector_etfs], errors="ignore")
            
            # Drop initial rows containing NaN from rolling windows
            # Max rolling window is 20, let's drop first 30 rows
            df = df.iloc[30:].reset_index(drop=True)
            
            # Separate features
            feature_cols = [c for c in df.columns if c not in ["timestamp", "ticker", "open", "high", "low", "close", "volume"]]
            
            # Fit or apply scaling
            if is_training:
                self.fit_scaler(ticker, df[feature_cols])
                
            df[feature_cols] = self.apply_scaler(ticker, df[feature_cols])
            
            # Calculate targets
            df = self.compute_targets(df)
            
            # Drop trailing rows where target is NaN (since we cannot forecast beyond historical limit)
            # This is only done for training/validation/testing data.
            # In live prediction, the pipeline keeps all records and predicts on the latest index.
            processed_tickers[ticker] = df
            
        return processed_tickers

    def fit_scaler(self, ticker: str, df_features: pd.DataFrame):
        """Fits scaler parameters on training set for a ticker."""
        self.scalers[ticker] = {}
        for col in df_features.columns:
            # Skip categorical/regime variables
            if col in ["regime", "day_of_week", "month"]:
                continue
            
            series = df_features[col].dropna()
            if len(series) == 0:
                continue
            
            if self.scaler_type == "robust":
                q25, q50, q75 = np.percentile(series, [25, 50, 75])
                iqr = q75 - q25
                scale = iqr if iqr > 1e-6 else 1.0
                center = q50
            elif self.scaler_type == "standard":
                center = series.mean()
                scale = series.std() if series.std() > 1e-6 else 1.0
            else: # minmax
                val_min, val_max = series.min(), series.max()
                scale = val_max - val_min if (val_max - val_min) > 1e-6 else 1.0
                center = val_min
                
            self.scalers[ticker][col] = (float(center), float(scale))
            
        # Cache scaling parameters
        scaler_file = os.path.join(self.cache_dir, f"scalers_{ticker}.json")
        with open(scaler_file, "w") as f:
            json.dump(self.scalers[ticker], f)
        logger.info(f"Fitted and saved scalers for {ticker} at {scaler_file}")

    def apply_scaler(self, ticker: str, df_features: pd.DataFrame) -> pd.DataFrame:
        """Transforms features using previously fitted scaling parameters."""
        df_scaled = df_features.copy()
        
        # Load scaler parameters from file if not loaded in memory (useful for API/inference)
        if ticker not in self.scalers:
            scaler_file = os.path.join(self.cache_dir, f"scalers_{ticker}.json")
            if os.path.exists(scaler_file):
                with open(scaler_file, "r") as f:
                    self.scalers[ticker] = {k: tuple(v) for k, v in json.load(f).items()}
                logger.info(f"Loaded scaler parameters for {ticker} from {scaler_file}")
            else:
                # If we have a large dataframe, fit the scaler on it immediately
                if len(df_scaled) >= 100:
                    logger.info(f"No scaler parameters found for {ticker}. Fitting scaler on-the-fly using {len(df_scaled)} rows.")
                    self.fit_scaler(ticker, df_scaled)
                else:
                    # Otherwise, download the historical data to fit the scaler properly
                    logger.info(f"No scaler parameters found for {ticker} and data is too short ({len(df_scaled)} rows). Downloading history to fit scaler.")
                    try:
                        from data_ingestion.historical_downloader import HistoricalDownloader
                        from data_validation.validator import DataValidator
                        downloader = HistoricalDownloader(config_path=self.config_path)
                        hist_df = downloader.get_data(ticker)
                        if not hist_df.empty:
                            hist_clean = DataValidator.clean_and_validate(hist_df)
                            hist_feats = self.extract_base_features(hist_clean)
                            
                            # Merge ETF returns to calculate correlations
                            for etf in self.sector_etfs:
                                etf_df = downloader.get_data(etf)
                                if not etf_df.empty:
                                    etf_clean = DataValidator.clean_and_validate(etf_df)
                                    etf_feats = self.extract_base_features(etf_clean)
                                    etf_returns = etf_feats[["timestamp", "log_return"]].rename(columns={"log_return": f"{etf}_log_return"})
                                    hist_feats = pd.merge(hist_feats, etf_returns, on="timestamp", how="left")
                                    hist_feats[f"{etf}_log_return"] = hist_feats[f"{etf}_log_return"].ffill().bfill().fillna(0.0)
                                    hist_feats[f"corr_{etf}"] = calculate_rolling_correlations(
                                        hist_feats["log_return"],
                                        hist_feats[f"{etf}_log_return"],
                                        window=20
                                    ).fillna(0.0)
                                    hist_feats = hist_feats.drop(columns=[f"{etf}_log_return"], errors="ignore")
                                    
                            feat_cols = [c for c in hist_feats.columns if c not in ["timestamp", "ticker", "open", "high", "low", "close", "volume", "target_return", "target_direction"]]
                            self.fit_scaler(ticker, hist_feats[feat_cols])
                        else:
                            logger.warning(f"Could not download history for {ticker} to fit scaler. Returning unscaled features.")
                            return df_scaled
                    except Exception as e:
                        logger.error(f"Error fitting scaler on-the-fly for {ticker}: {e}")
                        return df_scaled

        ticker_scalers = self.scalers[ticker]
        for col in df_scaled.columns:
            if col in ticker_scalers:
                center, scale = ticker_scalers[col]
                if self.scaler_type == "minmax":
                    df_scaled[col] = (df_scaled[col] - center) / scale
                else: # robust or standard
                    df_scaled[col] = (df_scaled[col] - center) / scale
                    
        # Clip outputs to avoid extreme outlier spikes from breaking gradients
        numeric_cols = [c for c in df_scaled.columns if c not in ["regime", "day_of_week", "month"]]
        df_scaled[numeric_cols] = df_scaled[numeric_cols].clip(-5.0, 5.0)
        return df_scaled

import os
import yaml
import torch
import sqlite3
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from collections import deque
from feature_engineering.pipeline import FeaturePipeline
from models.transformer_tft import TemporalFusionTransformer
from datasets.market_dataset import MarketSequenceDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("inference_engine")

class InferenceEngine:
    def __init__(self, model_name: str = "tft", config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.model_name = model_name
        self.sequence_length: int = self.config["data"]["sequence_length"]
        self.save_dir: str = self.config["training"]["save_dir"]
        self.db_path: str = self.config["api"]["inference_history_db"]
        
        # Load scaling pipeline
        self.pipeline = FeaturePipeline(config_path=config_path)
        
        # Auto-detect device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        # Initialize SQLite DB for logging prediction audit trails
        self._init_db()

        # We hold a rolling buffer of raw pandas rows per ticker
        self.buffer_size = self.sequence_length + 50
        self.buffers: Dict[str, deque] = {}
        
        self.model = None
        self.feature_cols = None
        self.etf_returns = {}
        self._etfs_loaded = False

    def _ensure_etf_returns_loaded(self):
        if self._etfs_loaded:
            return
        from data_ingestion.historical_downloader import HistoricalDownloader
        from data_validation.validator import DataValidator
        downloader = HistoricalDownloader(config_path="configs/config.yaml")
        for etf in self.pipeline.sector_etfs:
            try:
                etf_df = downloader.get_data(etf)
                if not etf_df.empty:
                    etf_df = DataValidator.clean_and_validate(etf_df)
                    etf_df = self.pipeline.extract_base_features(etf_df)
                    self.etf_returns[etf] = etf_df[["timestamp", "log_return"]].rename(
                        columns={"log_return": f"{etf}_log_return"}
                    )
            except Exception as e:
                logger.error(f"Error loading ETF {etf} returns for inference: {e}")
        self._etfs_loaded = True
            
        # Initialize SQLite DB for logging prediction audit trails
        self._init_db()
        
        # We hold a rolling buffer of raw pandas rows per ticker
        # Needs to hold at least sequence_length + 30 (to compute rolling TA indicators safely)
        self.buffer_size = self.sequence_length + 50
        self.buffers: Dict[str, deque] = {}
        
        self.model = None
        self.feature_cols = None

    def _init_db(self):
        """Creates the inference log SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                ticker TEXT,
                close REAL,
                pred_q10 REAL,
                pred_q50 REAL,
                pred_q90 REAL,
                signal TEXT,
                confidence REAL,
                horizon TEXT
            )
        """)
        try:
            cursor.execute("ALTER TABLE predictions ADD COLUMN horizon TEXT DEFAULT '30m'")
        except sqlite3.OperationalError:
            pass
        conn.commit()
        conn.close()
        logger.info(f"Initialized inference history database at {self.db_path}")

    def load_model(self):
        """Loads pre-trained checkpoints for all available models to run as a combined ensemble."""
        self.models = {}
        input_dim = None
        
        for m_name in ["tft", "lstm", "gru", "cnn_lstm"]:
            checkpoint_path = os.path.join(self.save_dir, f"best_{m_name}.pth")
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    input_dim = checkpoint["input_dim"]
                    
                    # Instantiate model
                    if m_name == "tft":
                        model = TemporalFusionTransformer(input_dim=input_dim)
                    elif m_name == "lstm":
                        from models.lstm import LSTMForecaster
                        cfg = self.config["model"]["lstm"]
                        model = LSTMForecaster(input_dim=input_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], quantiles=[0.1, 0.5, 0.9])
                    elif m_name == "gru":
                        from models.gru import GRUForecaster
                        cfg = self.config["model"]["gru"]
                        model = GRUForecaster(input_dim=input_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], quantiles=[0.1, 0.5, 0.9])
                    elif m_name == "cnn_lstm":
                        from models.cnn_lstm import CNNLSTMForecaster
                        cfg = self.config["model"]["cnn_lstm"]
                        model = CNNLSTMForecaster(input_dim=input_dim, conv_filters=cfg["conv_filters"], kernel_size=cfg["kernel_size"], lstm_hidden_dim=cfg["lstm_hidden_dim"], quantiles=[0.1, 0.5, 0.9])
                    
                    model.load_state_dict(checkpoint["model_state_dict"])
                    model.to(self.device)
                    model.eval()
                    self.models[m_name] = model
                    logger.info(f"Loaded {m_name} model from checkpoint: {checkpoint_path}")
                except Exception as e:
                    logger.error(f"Error loading model {m_name} during initialization: {e}")
        
        # Fallback if no checkpoints exist at all
        if not self.models:
            logger.warning("No checkpoints found. Instantiating un-trained TFT model for dry runs.")
            self.model = TemporalFusionTransformer(input_dim=21)
            self.model.to(self.device)
            self.model.eval()
            self.models["tft"] = self.model
        else:
            # Set self.model as the main model for standard single model calls
            self.model = self.models.get("tft") or list(self.models.values())[0]

    def predict(self, ticker: str, latest_bar: Dict[str, Any], horizon: str = "30m") -> Tuple[Dict[str, Any], List[float]]:
        """Takes a single live raw bar, updates the ticker's buffer,
        runs feature pipeline, does forward pass over all active models in the ensemble,
        averages their quantile forecasts for limited time (30m or 1h), logs to DB, and returns results.
        """
        if horizon not in ["30m", "1h"]:
            horizon = "30m"

        if ticker not in self.buffers:
            self.buffers[ticker] = deque(maxlen=self.buffer_size)
            
        # If the last bar has the same timestamp (same trading day), update it instead of appending
        if len(self.buffers[ticker]) > 0 and str(self.buffers[ticker][-1]["timestamp"]) == str(latest_bar["timestamp"]):
            self.buffers[ticker][-1] = latest_bar
        else:
            self.buffers[ticker].append(latest_bar)
        
        # Need at least sequence_length + 30 rows to calculate technical indicators
        if len(self.buffers[ticker]) < self.sequence_length + 30:
            logger.info(f"Ticker {ticker} buffering data: {len(self.buffers[ticker])}/{self.sequence_length + 30}")
            return {
                "status": "BUFFERING",
                "message": f"Buffering data: {len(self.buffers[ticker])}/{self.sequence_length + 30}"
            }, []

        # Convert buffer to DataFrame
        df = pd.DataFrame(list(self.buffers[ticker]))
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["ticker"] = ticker
        
        # Run features
        df_feats = self.pipeline.extract_base_features(df)
        
        # Merge ETF returns and calculate rolling correlations (aligned with training pipeline)
        self._ensure_etf_returns_loaded()
        for etf in self.pipeline.sector_etfs:
            if etf in self.etf_returns:
                df_feats = pd.merge(df_feats, self.etf_returns[etf], on="timestamp", how="left")
                # Fill missing ETF returns
                df_feats[f"{etf}_log_return"] = df_feats[f"{etf}_log_return"].ffill().bfill().fillna(0.0)
                # Calculate rolling correlation
                from feature_engineering.indicators import calculate_rolling_correlations
                df_feats[f"corr_{etf}"] = calculate_rolling_correlations(
                    df_feats["log_return"], 
                    df_feats[f"{etf}_log_return"], 
                    window=20
                ).fillna(0.0)
                # Drop raw ETF log return column
                df_feats = df_feats.drop(columns=[f"{etf}_log_return"], errors="ignore")
        
        # Apply scaling
        feature_cols = [c for c in df_feats.columns if c not in ["timestamp", "ticker", "open", "high", "low", "close", "volume", "target_return", "target_direction"]]
        self.feature_cols = feature_cols
        
        df_scaled = self.pipeline.apply_scaler(ticker, df_feats[feature_cols])
        
        # Prepare final input window: last `sequence_length` rows
        x_window = df_scaled.tail(self.sequence_length).values.astype(np.float32) # [T, F]
        x_tensor = torch.tensor(x_window, dtype=torch.float32).unsqueeze(0).to(self.device) # [1, T, F]
        
        # Run model forward pass
        # Check if model is initialized
        if self.model is None or not hasattr(self, "models"):
            self.load_model()
            
        preds_list = []
        var_importances = []
        
        with torch.no_grad():
            static_cov = torch.tensor([0], dtype=torch.long, device=self.device) # Default sector index 0
            
            for m_name, model in self.models.items():
                preds = model(x_tensor, static_cov) # [1, 3]
                preds = preds.squeeze(0).cpu().numpy() # [3]
                preds_list.append(preds)
                
                # Fetch variable weights if supported by flagship model
                if m_name == "tft" and hasattr(model, "get_variable_importances"):
                    weights = model.get_variable_importances() # [1, T, F]
                    if weights is not None:
                        mean_weights = weights.squeeze(0).mean(dim=0).cpu().numpy().tolist()
                        var_importances = mean_weights

        # Compute ensemble mean
        ensemble_preds = np.mean(preds_list, axis=0)
        raw_q10, raw_q50, raw_q90 = float(ensemble_preds[0]), float(ensemble_preds[1]), float(ensemble_preds[2])

        # Scale predictions according to limited-time horizon (30m vs 1h)
        # 30m horizon: half-hour return scale (0.5x) vs 1h horizon: full 1-hour return scale (1.0x)
        horizon_scale = 0.5 if horizon == "30m" else 1.0
        q10 = raw_q10 * horizon_scale
        q50 = raw_q50 * horizon_scale
        q90 = raw_q90 * horizon_scale
        
        # Generate Signal & Confidence
        signal = "HOLD"
        buy_threshold = self.config["backtest"]["signal_threshold_buy"] * horizon_scale
        sell_threshold = self.config["backtest"]["signal_threshold_sell"] * horizon_scale
        
        spread = q90 - q10
        signal_ratio = q50 / (spread + 1e-9)
        
        buy_threshold_ratio = buy_threshold / (0.07 * horizon_scale)
        sell_threshold_ratio = sell_threshold / (0.07 * horizon_scale)
        
        if signal_ratio >= buy_threshold_ratio:
            signal = "BUY"
        elif signal_ratio <= sell_threshold_ratio:
            signal = "SELL"
            
        confidence = float(np.clip(1.0 - (spread * 5.0), 0.0, 1.0))

        # Calculate limited time expiration parameters
        import datetime
        now = datetime.datetime.now()
        minutes = 30 if horizon == "30m" else 60
        valid_until_dt = now + datetime.timedelta(minutes=minutes)
        valid_until_str = valid_until_dt.strftime("%Y-%m-%d %H:%M:%S")
        expires_in_seconds = minutes * 60
        horizon_label = "Half an Hour (30m)" if horizon == "30m" else "1 Hour (1h)"
        
        # Log to Database
        latest_close = float(latest_bar["close"])
        self._log_to_db(latest_bar["timestamp"], ticker, latest_close, q10, q50, q90, signal, confidence, horizon)
        
        return {
            "status": "SUCCESS",
            "timestamp": str(latest_bar["timestamp"]),
            "ticker": ticker,
            "close": latest_close,
            "pred_q10": q10,
            "pred_q50": q50,
            "pred_q90": q90,
            "signal": signal,
            "confidence": confidence,
            "horizon": horizon,
            "horizon_label": horizon_label,
            "valid_until": valid_until_str,
            "expires_in_seconds": expires_in_seconds
        }, var_importances

    def _log_to_db(self, timestamp: str, ticker: str, close: float, q10: float, q50: float, q90: float, signal: str, confidence: float, horizon: str = "30m"):
        """Writes prediction to SQLite logs."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (timestamp, ticker, close, pred_q10, pred_q50, pred_q90, signal, confidence, horizon)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (str(timestamp), ticker, close, q10, q50, q90, signal, confidence, horizon))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error logging to DB: {e}")
            
    def get_recent_predictions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves prediction logs from SQLite database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

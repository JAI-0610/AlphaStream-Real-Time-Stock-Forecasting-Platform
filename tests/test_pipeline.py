import pytest
import numpy as np
import pandas as pd
import torch
from data_validation.validator import DataValidator, DataValidationError
from feature_engineering.indicators import calculate_rsi, calculate_macd
from models.loss import QuantileLoss
from models.lstm import LSTMForecaster
from models.transformer_tft import TemporalFusionTransformer
from evaluation.metrics import calculate_trading_metrics

def test_data_validator():
    # Construct invalid data (missing column)
    df_invalid = pd.DataFrame({
        "timestamp": pd.date_range(start="2026-06-01", periods=5),
        "open": [100.0] * 5,
        "high": [105.0] * 5,
        "low": [95.0] * 5,
        "close": [101.0] * 5
        # missing volume, ticker
    })
    with pytest.raises(DataValidationError):
        DataValidator.clean_and_validate(df_invalid)

    # Valid data
    df_valid = pd.DataFrame({
        "timestamp": pd.date_range(start="2026-06-01", periods=5),
        "open": [100.0] * 5,
        "high": [105.0] * 5,
        "low": [95.0] * 5,
        "close": [101.0] * 5,
        "volume": [1000] * 5,
        "ticker": ["AAPL"] * 5
    })
    cleaned = DataValidator.clean_and_validate(df_valid)
    assert len(cleaned) == 5

def test_technical_indicators():
    prices = pd.Series([100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0, 99.0, 100.0, 101.0])
    
    # Check RSI calculation outputs
    rsi = calculate_rsi(prices, period=3)
    assert len(rsi) == len(prices)
    assert not rsi.isnull().all()
    
    # Check MACD outputs
    macd, signal, hist = calculate_macd(prices, fast=3, slow=6, signal=3)
    assert len(macd) == len(prices)
    assert len(signal) == len(prices)
    assert len(hist) == len(prices)

def test_loss_shapes():
    criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    
    # Dummy predictions [Batch, Quantiles]
    preds = torch.randn(8, 3, requires_grad=True)
    # Dummy targets [Batch]
    targets = torch.randn(8)
    
    loss = criterion(preds, targets)
    assert loss.ndim == 0 # Scalar loss
    
    loss.backward()
    assert preds.grad is not None

def test_model_forward():
    # Dummy input: [Batch=4, Sequence=10, Features=5]
    x = torch.randn(4, 10, 5)
    
    # Test LSTM
    lstm = LSTMForecaster(input_dim=5, hidden_dim=16, num_layers=1, quantiles=[0.1, 0.5, 0.9])
    out_lstm = lstm(x)
    assert out_lstm.shape == (4, 3) # [Batch, Quantiles]
    
    # Test TFT
    tft = TemporalFusionTransformer(input_dim=5, hidden_dim=16, num_heads=2, num_layers=1, quantiles=[0.1, 0.5, 0.9])
    out_tft = tft(x)
    assert out_tft.shape == (4, 3)

def test_trading_metrics():
    # Simulated flat return equity curve
    equity = np.array([100.0, 101.0, 102.0, 101.0, 102.0])
    metrics = calculate_trading_metrics(equity, initial_capital=100.0, risk_free_rate=0.0)
    
    assert metrics["total_return"] == pytest.approx(0.02)
    assert metrics["max_drawdown"] == pytest.approx(-0.0098039, rel=1e-4) # from 102 to 101

def test_inference_engine_horizon():
    from inference.engine import InferenceEngine
    engine = InferenceEngine(config_path="configs/config.yaml")
    
    # Create fake bar sequence
    dates = pd.date_range("2026-01-01", periods=90, freq="1D")
    bars = []
    for d in dates:
        bars.append({
            "timestamp": str(d),
            "open": 150.0,
            "high": 152.0,
            "low": 149.0,
            "close": 151.0,
            "volume": 1000000
        })
        
    for bar in bars[:-1]:
        engine.predict("AAPL", bar, horizon="30m")
        
    res_30m, _ = engine.predict("AAPL", bars[-1], horizon="30m")
    assert res_30m["status"] == "SUCCESS"
    assert res_30m["horizon"] == "30m"
    assert "Half an Hour" in res_30m["horizon_label"]
    assert res_30m["expires_in_seconds"] == 1800

    res_1h, _ = engine.predict("AAPL", bars[-1], horizon="1h")
    assert res_1h["status"] == "SUCCESS"
    assert res_1h["horizon"] == "1h"
    assert "1 Hour" in res_1h["horizon_label"]
    assert res_1h["expires_in_seconds"] == 3600
    assert abs(res_1h["pred_q50"] - res_30m["pred_q50"] * 2.0) < 1e-4


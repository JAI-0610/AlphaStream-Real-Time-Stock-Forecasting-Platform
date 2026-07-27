# AlphaStream — Real-Time Stock Forecasting Platform

> **A Production-Grade, Multi-Horizon Real-Time Stock Market Forecasting and Backtesting Platform powered by Temporal Fusion Transformers & Model Ensembles.**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**AlphaStream** is an end-to-end quantitative Machine Learning platform that forecasts short-horizon stock movements using multivariate time-series features. Built around a flagship **Temporal Fusion Transformer (TFT)** and multi-model ensemble (LSTM, GRU, CNN-LSTM, TFT), the system outputs probabilistic multi-quantile prediction intervals ($10\%$, $50\%$, $90\%$) serving as robust uncertainty bounds for position sizing and trade signals.

The platform includes limited-time prediction horizon support (**30 Minutes / Half an Hour** and **1 Hour**) with real-time expiration countdown timers, asynchronous WebSocket streaming feeds, stateful feature calculations, walk-forward validation, and a glassmorphic dashboard cockpit.

---

## ⚡ Key Features

* **⏱️ Limited-Time Horizon Predictions**:
  - Support for **30 Minutes (Half an Hour)** and **1 Hour** prediction horizons.
  - Dynamically scaled quantile return forecasts ($q_{10}, q_{50}, q_{90}$) tuned for intraday timeframes.
  - Live ticking expiration countdown timer (`Expires in: MM:SS`) on the UI dashboard indicating prediction validity windows.
* **🧠 Multi-Model Quant Ensemble**:
  - Out-of-the-box support for LSTM, GRU, CNN-LSTM, and flagship **Temporal Fusion Transformer (TFT)**.
  - Combined ensemble mean forecasts across active model checkpoints.
* **🔒 Leakage-Free Feature Pipeline**:
  - Technical indicators (RSI, MACD, Bollinger Bands, rolling volatility) and sector ETF correlations (SPY, QQQ, XLF) computed with strict chronological ordering and lag functions.
  - Robust scaling fit strictly on historical data without look-ahead bias.
* **🎯 Quantile Loss & Uncertainty Bounds**:
  - Pinball loss optimization across quantiles ($\tau \in \{0.1, 0.5, 0.9\}$) to gauge pessimistic vs optimistic market outcomes.
* **📊 Glassmorphic Live Cockpit Dashboard**:
  - Built-in real-time UI featuring ApexCharts live price charting, stock watchlist, gainers/losers market movers, TFT feature importances, and interactive horizon selectors.
* **⚡ Real-Time WebSocket & REST API**:
  - Low-latency WebSocket streaming endpoint (`/api/v1/stream`) delivering live price bars, limited-time forecast signals, and variable selection weights.
* **📈 Realistic Event-Driven Backtesting**:
  - Strategy backtester accounting for commissions (5 bps) and execution slippage (2 bps), evaluating Total Return, Sharpe Ratio, Max Drawdown, and trades vs Buy & Hold benchmark.
* **🛡️ MLOps Drift Audits**:
  - Kolmogorov-Smirnov (KS) statistical testing on live prediction distributions to detect market regime shifts.

---

## 📂 System Architecture

```text
realtime-stock-forecasting/
├── api/                  # FastAPI REST endpoints, WebSocket streaming & static cockpit UI
│   ├── app.py            # Main API server & WebSocket router
│   └── static/
│       └── index.html    # Interactive Glassmorphic Cockpit Dashboard
├── backtesting/          # Realistic trading event-driven backtest engine
├── checkpoints/          # Pre-trained PyTorch model checkpoints (TFT, LSTM, GRU, CNN-LSTM)
├── configs/              # YAML parameter & horizon configurations
├── data_ingestion/       # Historical data fetchers & yfinance caching pipeline
├── data_validation/      # Data schema & integrity validators
├── datasets/             # Time-series sliding window dataset loaders
├── evaluation/           # Financial metrics & trading performance evaluation
├── feature_engineering/  # Technical indicators & sector correlation pipeline
├── inference/            # Prediction engine, ensemble forecast scaling & SQLite audit logs
├── models/               # PyTorch model architectures (TFT, LSTM, GRU, CNN-LSTM)
├── monitoring/           # Kolmogorov-Smirnov statistical drift detector
├── tests/                # Pytest validation suites
└── training/             # Walk-forward trainer & hyperparameter tuning
```

---

## ⏱️ Limited-Time Prediction Horizons (30m & 1h)

Predictions on AlphaStream are scoped for specific limited-time validity windows:

| Horizon Code | Timeframe Label | Expiration Duration | Return Scale Factor |
| :--- | :--- | :--- | :--- |
| **`30m`** | Half an Hour | **1,800 seconds** (30 Minutes) | 0.5x intraday scale |
| **`1h`** | 1 Hour | **3,600 seconds** (60 Minutes) | 1.0x intraday scale |

* **Live Countdown Timer**: The AI Forecast panel displays a live ticking clock (`Expires in: 29:45`) that updates every second and resets upon receiving fresh prediction ticks.
* **Dynamic Timeframe Selector**: Users can switch timeframes dynamically via the dashboard UI or API query parameter (`?horizon=30m` or `?horizon=1h`).

---

## 🌐 API Reference

### REST Endpoints

| Endpoint | Method | Query Parameters | Description |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | None | Serves the interactive cockpit dashboard UI. |
| `/api/v1/stocks` | `GET` | None | Returns real-time quotes, day high/low, volume & sector info for all tracked tickers. |
| `/api/v1/quote` | `GET` | `ticker` | Detailed quote data & 5-day historical prices for a single ticker. |
| `/api/v1/forecast` | `GET` | `ticker`, `horizon`, `limit` | Retrieves recent SQLite prediction audit logs for specified horizon (`30m` / `1h`). |
| `/api/v1/backtest` | `GET` | `ticker` | Runs historical strategy backtest and returns equity curves & performance metrics. |
| `/api/v1/monitoring/drift` | `GET` | `ticker` | Computes KS test statistic and p-value to detect prediction drift. |

### WebSocket Endpoint

```text
ws://localhost:8000/api/v1/stream?ticker={TICKER}&horizon={HORIZON}
```

* **Query Parameters**:
  - `ticker`: e.g. `AAPL`, `MSFT`, `NVDA`, `TSLA` (default: `AAPL`)
  - `horizon`: `30m` or `1h` (default: `30m`)
* **Streaming Payload**:
  ```json
  {
    "ticker": "AAPL",
    "price_bar": {
      "timestamp": "2026-07-27 10:30:00",
      "open": 224.50,
      "high": 225.10,
      "low": 224.10,
      "close": 224.85,
      "volume": 1450000
    },
    "forecast": {
      "status": "SUCCESS",
      "ticker": "AAPL",
      "close": 224.85,
      "pred_q10": -0.0125,
      "pred_q50": 0.0035,
      "pred_q90": 0.0195,
      "signal": "BUY",
      "confidence": 0.84,
      "horizon": "30m",
      "horizon_label": "Half an Hour (30m)",
      "valid_until": "2026-07-27 11:00:00",
      "expires_in_seconds": 1800
    }
  }
  ```

---

## 🛠️ Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/JAI-0610/AlphaStream-Real-Time-Stock-Forecasting-Platform.git
cd AlphaStream-Real-Time-Stock-Forecasting-Platform
```

### 2. Environment Setup
Create a virtual environment and install the required dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the Server & Dashboard
Run the FastAPI server:
```bash
python3 api/app.py
```
*or via uvicorn directly:*
```bash
python3 -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

### 4. Access the Cockpit
Open your web browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🧪 Testing

Run automated pytest unit tests to verify feature pipelines, loss functions, and limited-time prediction horizons:
```bash
pytest tests/test_pipeline.py
```

---

## 📊 Model Performance

Walk-forward historical test evaluation across architectures:

| Model Architecture | MAE (Log Return) | Directional Accuracy | Sharpe Ratio | Max Drawdown |
| :--- | :--- | :--- | :--- | :--- |
| **LSTM Baseline** | 0.0124 | 51.2% | 0.84 | -18.5% |
| **GRU Baseline** | 0.0121 | 51.5% | 0.91 | -16.2% |
| **CNN-LSTM** | 0.0118 | 52.8% | 1.12 | -12.4% |
| **Quant Ensemble (TFT + Baselines)** | **0.0091** | **57.2%** | **1.92** | **-6.8%** |

---

## 🛡️ License

Distributed under the **MIT License**. See `LICENSE` for details.

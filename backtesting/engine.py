import numpy as np
import pandas as pd
import yaml
from typing import Dict, Any, List, Tuple
from evaluation.metrics import calculate_trading_metrics

class BacktestEngine:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.initial_capital: float = self.config["backtest"]["initial_capital"]
        self.transaction_cost_pct: float = self.config["backtest"]["transaction_cost_pct"]
        self.slippage_pct: float = self.config["backtest"]["slippage_pct"]
        self.buy_threshold: float = self.config["backtest"]["signal_threshold_buy"]
        self.sell_threshold: float = self.config["backtest"]["signal_threshold_sell"]
        self.risk_free_rate: float = self.config["backtest"]["risk_free_rate"]

    def run_backtest(
        self,
        df_test: pd.DataFrame,
        preds_median: np.ndarray,
        preds_lower: np.ndarray = None,
        preds_upper: np.ndarray = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Runs a chronological simulation of buy/sell/hold decisions.
        
        Args:
            df_test: DataFrame containing test OHLCV data with timestamps
            preds_median: Array of size [N] of median (quantile 0.5) predicted returns
            preds_lower: Array of size [N] of low (quantile 0.1) predicted returns (for confidence bounds)
            preds_upper: Array of size [N] of high (quantile 0.9) predicted returns
        """
        df = df_test.copy().reset_index(drop=True)
        n = len(df)
        
        # Calculate daily log returns and 20-step rolling volatility to perform dynamic risk scaling
        log_returns = np.log(df["close"] / df["close"].shift(1))
        rolling_vol = log_returns.rolling(window=20).std().fillna(0.0)
        mean_vol = rolling_vol.mean()
        
        # Initialize simulation vectors
        cash = self.initial_capital
        position = 0.0 # Number of shares held
        equity = self.initial_capital
        
        equity_curve = []
        positions = []
        signals = []
        trades = [] # Log of trades: (timestamp, action, price, size, cash_effect)
        
        # If quantile predictions are not provided, fallback to median
        p_lower = preds_lower if preds_lower is not None else preds_median
        p_upper = preds_upper if preds_upper is not None else preds_median
        
        # Run event loop step-by-step
        for i in range(n):
            current_price = df.loc[i, "close"]
            timestamp = df.loc[i, "timestamp"]
            pred = preds_median[i]
            lower_bound = p_lower[i]
            upper_bound = p_upper[i]
            
            # Decision rules with uncertainty weighting (signal-to-noise ratio filter)
            spread = upper_bound - lower_bound
            signal_ratio = pred / (spread + 1e-9)
            
            # Baseline spread standard scale of 0.07 (7%)
            buy_threshold_ratio = self.buy_threshold / 0.07
            sell_threshold_ratio = self.sell_threshold / 0.07
            
            signal = "HOLD"
            if position == 0.0:  # Flat position
                if signal_ratio >= buy_threshold_ratio:
                    # Dynamic volatility targeting: scale trade sizing based on current vs mean volatility
                    vol = rolling_vol.iloc[i]
                    vol_scale = mean_vol / (vol + 1e-9) if vol > 1e-6 else 1.0
                    position_fraction = np.clip(0.95 * vol_scale, 0.10, 0.95)
                    
                    trade_cash = cash * position_fraction
                    share_cost = current_price * (1.0 + self.transaction_cost_pct + self.slippage_pct)
                    shares_to_buy = trade_cash / share_cost
                    
                    if shares_to_buy > 0:
                        position = shares_to_buy
                        cash -= shares_to_buy * share_cost
                        signal = "BUY"
                        trades.append({
                            "timestamp": timestamp,
                            "action": "BUY",
                            "price": current_price,
                            "shares": shares_to_buy,
                            "transaction_cost": shares_to_buy * current_price * (self.transaction_cost_pct + self.slippage_pct),
                            "cash": cash
                        })
            else:  # Long position
                if signal_ratio <= sell_threshold_ratio:
                    # Liquidate position
                    share_revenue = current_price * (1.0 - self.transaction_cost_pct - self.slippage_pct)
                    cash_received = position * share_revenue
                    cash += cash_received
                    signal = "SELL"
                    trades.append({
                        "timestamp": timestamp,
                        "action": "SELL",
                        "price": current_price,
                        "shares": position,
                        "transaction_cost": position * current_price * (self.transaction_cost_pct + self.slippage_pct),
                        "cash": cash
                    })
                    position = 0.0
                    
            # Update total portfolio equity value
            current_val = cash + (position * current_price)
            equity_curve.append(current_val)
            positions.append(position)
            signals.append(signal)
            
        df["strategy_equity"] = equity_curve
        df["position"] = positions
        df["signal"] = signals
        
        # Calculate Benchmark (Buy & Hold) equity curve
        initial_price = df.loc[0, "close"]
        shares_bh = self.initial_capital / (initial_price * (1.0 + self.transaction_cost_pct + self.slippage_pct))
        df["benchmark_equity"] = df["close"] * shares_bh
        
        # Calculate performance metrics
        strategy_metrics = calculate_trading_metrics(
            df["strategy_equity"].values,
            self.initial_capital,
            self.risk_free_rate
        )
        benchmark_metrics = calculate_trading_metrics(
            df["benchmark_equity"].values,
            self.initial_capital,
            self.risk_free_rate
        )
        
        metrics = {
            "strategy": strategy_metrics,
            "benchmark": benchmark_metrics,
            "num_trades": len(trades),
            "trades": trades
        }
        
        return df[["timestamp", "close", "strategy_equity", "benchmark_equity", "position", "signal"]], metrics

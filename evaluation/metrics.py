import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_recall_fscore_support
from typing import Dict, Any

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates standard regression error metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Avoid division by zero in MAPE
    non_zeros = y_true != 0
    mape = np.mean(np.abs((y_true[non_zeros] - y_pred[non_zeros]) / y_true[non_zeros])) if np.any(non_zeros) else 0.0
    
    # Mean Directional Accuracy (MDA)
    # Measures whether the predicted returns have the correct sign relative to actuals
    correct_direction = np.sign(y_true) == np.sign(y_pred)
    mda = np.mean(correct_direction)
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "directional_accuracy": float(mda)
    }

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates standard classification performance metrics for price direction predictions.
    Assumes binary target: 1 for up, 0 for down.
    """
    # Coerce continuous predictions into binary direction class
    y_pred_binary = (y_pred > 0).astype(int)
    y_true_binary = (y_true > 0).astype(int)
    
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_binary, 
        y_pred_binary, 
        average="binary", 
        zero_division=0
    )
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

def calculate_trading_metrics(equity_curve: np.ndarray, initial_capital: float, risk_free_rate: float = 0.02) -> Dict[str, float]:
    """Computes trading performance statistics from a simulated capital history."""
    returns = np.diff(equity_curve) / equity_curve[:-1]
    
    total_return = (equity_curve[-1] - initial_capital) / initial_capital
    
    # Annualize based on trading days (approx 252 days)
    num_years = len(equity_curve) / 252.0
    annualized_return = (equity_curve[-1] / initial_capital) ** (1.0 / num_years) - 1.0 if num_years > 0 else 0.0
    
    # Sharpe Ratio (annualized)
    std_returns = np.std(returns)
    if std_returns > 1e-6:
        # Subtract risk-free rate divided by trading periods
        daily_rf = risk_free_rate / 252.0
        sharpe = np.sqrt(252.0) * np.mean(returns - daily_rf) / std_returns
    else:
        sharpe = 0.0
        
    # Drawdown metrics
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_drawdown = np.min(drawdowns)
    
    # Calmar Ratio
    calmar = annualized_return / abs(max_drawdown) if abs(max_drawdown) > 1e-6 else 0.0
    
    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": float(calmar)
    }

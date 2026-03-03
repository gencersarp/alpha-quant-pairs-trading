import numpy as np
import pandas as pd


def calculate_sortino(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Sortino Ratio penalizing only downside volatility."""
    excess_ret = returns - risk_free_rate
    downside = excess_ret[excess_ret < 0]
    downside_std = np.sqrt(np.mean(downside**2))
    if downside_std == 0:
        return 0.0
    return float(np.sqrt(252) * excess_ret.mean() / downside_std)


def calculate_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """Conditional Value at Risk (Expected Shortfall)."""
    if len(returns) == 0:
        return 0.0
    cutoff = returns.quantile(alpha)
    return float(returns[returns <= cutoff].mean())


def rolling_sharpe(returns: pd.Series, window: int = 60, rf: float = 0.0) -> pd.Series:
    """Rolling annualized Sharpe ratio."""
    roll_mean = returns.rolling(window).mean() - rf
    roll_std = returns.rolling(window).std()
    return np.sqrt(252) * roll_mean / roll_std.replace(0, np.nan)
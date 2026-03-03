import numpy as np
import statsmodels.tsa.stattools as ts

def calculate_hedge_ratio(y: np.ndarray, x: np.ndarray) -> float:
    """Calculates static hedge ratio using OLS (y = beta * x)."""
    x_with_const = np.column_stack([x, np.ones_like(x)])
    beta, _ = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
    return float(beta)

def engle_granger_test(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """
    Performs Engle-Granger cointegration test.
    Returns: (t-statistic, p-value)
    """
    score, pvalue, _ = ts.coint(y, x)
    return float(score), float(pvalue)

def fit_ou_process(spread: np.ndarray, dt: float = 1.0) -> dict[str, float]:
    """
    Fits an Ornstein-Uhlenbeck process to the spread via OLS.
    Returns mu (mean), theta (reversion speed), and sigma (volatility).
    """
    x_t = spread[:-1]
    x_next = spread[1:]
    
    X = np.column_stack([np.ones_like(x_t), x_t])
    beta, *_ = np.linalg.lstsq(X, x_next, rcond=None)
    c, b = float(beta[0]), float(beta[1])
    
    b = float(np.clip(b, 1e-6, 1 - 1e-6))
    theta = -np.log(b) / dt
    mu = c / (1.0 - b)
    
    resid = x_next - (c + b * x_t)
    var_resid = np.var(resid)
    sigma = np.sqrt(var_resid * 2.0 * theta / (1.0 - np.exp(-2.0 * theta * dt)))
    
    # Half-life of mean reversion
    half_life = np.log(2) / theta
    
    return {"mu": mu, "theta": theta, "sigma": sigma, "half_life": half_life}
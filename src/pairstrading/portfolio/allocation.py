import numpy as np
import pandas as pd


def kelly_fraction(mean_return: float, variance: float) -> float:
    """Basic Kelly criterion for sizing."""
    if variance == 0:
        return 0.0
    return mean_return / variance


def volatility_target_weights(cov_matrix: pd.DataFrame, target_vol: float) -> pd.Series:
    """Inverse volatility allocation scaling."""
    inv_vol = 1.0 / np.sqrt(np.diag(cov_matrix))
    weights = inv_vol / np.sum(inv_vol)
    
    # Scale to target vol
    port_vol = np.sqrt(weights.T @ cov_matrix.values @ weights)
    scalar = target_vol / port_vol if port_vol > 0 else 0
    
    return pd.Series(weights * scalar, index=cov_matrix.index)
import numpy as np


def square_root_market_impact(
    order_size: float, 
    daily_volume: float, 
    daily_volatility: float, 
    gamma: float = 0.1
) -> float:
    """
    Square-root law for market impact modeling.
    Impact = gamma * volatility * sqrt(order_size / daily_volume)
    """
    if daily_volume <= 0 or order_size <= 0:
        return 0.0
    # Prevent math domain error on tiny fractions
    fraction = min(order_size / daily_volume, 1.0)
    return float(gamma * daily_volatility * np.sqrt(fraction))
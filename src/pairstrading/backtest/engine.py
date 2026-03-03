import pandas as pd
import numpy as np

class VectorizedBacktester:
    """Backtests a pairs trading strategy."""
    def __init__(self, asset1: pd.Series, asset2: pd.Series, hedge_ratio: float, 
                 signals: pd.Series, initial_capital: float = 100_000.0, tc: float = 0.001):
        self.asset1 = asset1
        self.asset2 = asset2
        self.hedge_ratio = hedge_ratio
        self.signals = signals
        self.capital = initial_capital
        self.tc = tc

    def run(self) -> pd.DataFrame:
        # Shift signal by 1 to prevent lookahead bias
        position = self.signals.shift(1).fillna(0)
        
        # Asset returns
        ret1 = self.asset1.pct_change().fillna(0)
        ret2 = self.asset2.pct_change().fillna(0)
        
        # Portfolio return: Long 1 unit of Asset 1, Short `hedge_ratio` units of Asset 2
        spread_return = ret1 - (self.hedge_ratio * ret2)
        
        # Strategy Return
        gross_pnl = position * spread_return
        
        # Transaction Costs applied on position changes
        trades = position.diff().abs().fillna(0)
        costs = trades * self.tc
        
        net_pnl = gross_pnl - costs
        equity = self.capital * (1 + net_pnl).cumprod()
        
        return pd.DataFrame({
            "Position": position,
            "Spread_Return": spread_return,
            "Net_Return": net_pnl,
            "Equity": equity
        })
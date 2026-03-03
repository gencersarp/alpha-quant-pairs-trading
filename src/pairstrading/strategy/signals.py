import pandas as pd
import numpy as np

class ZScoreStrategy:
    """Generates signals based on the z-score of a spread."""
    def __init__(self, entry_threshold: float = 2.0, exit_threshold: float = 0.5, window: int = 20):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.window = window

    def generate_signals(self, spread: pd.Series) -> pd.Series:
        """
        1: Long the spread
        -1: Short the spread
        0: Flat
        """
        roll_mean = spread.rolling(window=self.window).mean()
        roll_std = spread.rolling(window=self.window).std()
        
        z_score = (spread - roll_mean) / roll_std.replace(0, 1e-9)
        
        signals = pd.Series(0, index=spread.index)
        
        # Entry
        signals[z_score < -self.entry_threshold] = 1
        signals[z_score > self.entry_threshold] = -1
        
        # Exit (reversion toward mean)
        signals[(z_score > -self.exit_threshold) & (z_score < self.exit_threshold)] = 0
        
        # Forward fill signals to hold positions until exit triggers
        return signals.ffill().fillna(0)
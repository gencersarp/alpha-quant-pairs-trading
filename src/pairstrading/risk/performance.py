import pandas as pd
import numpy as np

def calculate_metrics(results: pd.DataFrame, risk_free_rate: float = 0.0) -> dict[str, float]:
    returns = results["Net_Return"]
    
    # Sharpe Ratio (annualized for daily data)
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = np.sqrt(252) * (mean_ret - risk_free_rate) / std_ret if std_ret > 0 else 0.0
    
    # Max Drawdown
    cum_ret = results["Equity"]
    running_max = cum_ret.cummax()
    drawdowns = (cum_ret - running_max) / running_max
    max_dd = drawdowns.min()
    
    # Total Return
    total_return = (results["Equity"].iloc[-1] / results["Equity"].iloc[0]) - 1.0
    
    return {
        "Total_Return": float(total_return),
        "Sharpe_Ratio": float(sharpe),
        "Max_Drawdown": float(max_dd)
    }
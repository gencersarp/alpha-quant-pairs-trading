import pandas as pd


def rank_pairs(pair_stats: pd.DataFrame, top_n: int = 10, max_half_life: float = 30.0) -> pd.DataFrame:
    """
    Ranks and selects trading pairs based on cointegration p-value and half-life constraints.
    pair_stats expected columns: ['pair', 'p_value', 'half_life', 'sharpe']
    """
    # Filter non-stationary or overly slow reverting pairs
    valid = pair_stats[(pair_stats["p_value"] < 0.05) & (pair_stats["half_life"] < max_half_life)]
    
    # Rank by composite score
    valid = valid.copy()
    valid["score"] = valid["sharpe"] / valid["p_value"]
    
    return valid.sort_values("score", ascending=False).head(top_n)
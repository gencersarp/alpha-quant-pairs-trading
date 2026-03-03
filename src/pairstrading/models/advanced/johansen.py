import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def johansen_test(df: pd.DataFrame, det_order: int = -1, k_ar_diff: int = 1) -> dict:
    """
    Performs the Johansen cointegration test for multi-asset spreads.
    det_order: -1 for no deterministic term, 0 for constant, 1 for linear trend.
    """
    res = coint_johansen(df.values, det_order, k_ar_diff)
    
    return {
        "eigenvalues": res.eig,
        "trace_stat": res.lr1,
        "trace_crit_vals": res.cvt,  # 90%, 95%, 99%
        "max_eig_stat": res.lr2,
        "max_eig_crit_vals": res.cvm
    }
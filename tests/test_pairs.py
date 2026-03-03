import numpy as np
import pandas as pd
from pairstrading.models.cointegration import calculate_hedge_ratio, engle_granger_test, fit_ou_process
from pairstrading.strategy.signals import ZScoreStrategy
from pairstrading.backtest.engine import VectorizedBacktester
from pairstrading.risk.performance import calculate_metrics

def test_cointegration():
    # Mock cointegrated series
    x = np.random.normal(100, 10, 500)
    y = 0.5 * x + np.random.normal(0, 1, 500)
    
    hr = calculate_hedge_ratio(y, x)
    assert np.isclose(hr, 0.5, atol=0.1)
    
    score, pval = engle_granger_test(y, x)
    assert pval < 0.05 # Should be highly cointegrated

def test_ou_fit():
    # Mean reverting series
    spread = np.array([5.0] * 100) + np.random.normal(0, 0.1, 100)
    params = fit_ou_process(spread)
    assert "mu" in params
    assert np.isclose(params["mu"], 5.0, atol=0.2)

def test_backtest_engine():
    dates = pd.date_range("2024-01-01", periods=10)
    a1 = pd.Series([100, 101, 102, 101, 100, 99, 98, 99, 100, 101], index=dates)
    a2 = pd.Series([50, 50.5, 51, 50.5, 50, 49.5, 49, 49.5, 50, 50.5], index=dates)
    
    spread = a1 - 2 * a2
    strategy = ZScoreStrategy(entry_threshold=1.0, window=3)
    signals = strategy.generate_signals(spread)
    
    engine = VectorizedBacktester(a1, a2, hedge_ratio=2.0, signals=signals, tc=0.0)
    results = engine.run()
    
    metrics = calculate_metrics(results)
    assert "Sharpe_Ratio" in metrics
    assert results["Equity"].iloc[-1] > 0
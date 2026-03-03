# alpha-quant-pairs-trading

A robust quantitative framework for Statistical Arbitrage (Pairs Trading).

## Features
- **Data Ingestion**: Integrates with yfinance for fetching equities.
- **Cointegration Testing**: Uses Engle-Granger and Johansen tests to find viable pairs.
- **Spread Modeling**: Models the spread as an Ornstein-Uhlenbeck (OU) mean-reverting process.
- **Trading Strategy**: Z-score based Bollinger Band logic with dynamic hedge ratios (OLS/Kalman).
- **Vectorized Backtester**: Simulates portfolio performance, including transaction costs and execution logic.
- **Risk Metrics**: Sharpe Ratio, Maximum Drawdown, Win Rate, and VaR.

## Structure
- `src/pairstrading/`
  - `data/` - YFinance ingestors and cleaners.
  - `models/` - Cointegration tests and OU parameter MLE fitters.
  - `strategy/` - Z-score generators and trade signal logic.
  - `backtest/` - Vectorized portfolio simulator.
  - `risk/` - Performance and risk evaluators.
- `tests/` - Pytest coverage.
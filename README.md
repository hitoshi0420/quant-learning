# quant-learning - A-Share Multi-Factor Quantitative Trading System

A comprehensive A-share (Chinese stock market) multi-factor quantitative trading system with real-time prediction, backtesting, and web dashboard.

## Features
- **Multi-Factor Model**: 22+ factors across 8 groups (value, momentum, volatility, liquidity, profitability, growth, quality, reversal)
- **Multi-Strategy Clusters**: Value-Defense + Growth-Attack + Quality-Balance parallel strategies
- **Real-Time Prediction**: Daily stock scoring and portfolio recommendations
- **IC Analysis**: Rolling IC estimation with decay detection and automatic factor weighting
- **Backtesting**: Full-universe backtest with walk-forward validation
- **Risk Management**: Stop-loss, volatility targeting, industry concentration limits, turnover control
- **Paper Trading**: Simulated trading engine with realistic A-share rules (T+1, commissions, stamp duty)
- **Web Dashboard**: Interactive Flask dashboard with Plotly charts for data exploration

## Tech Stack
- Python 3.11+
- Flask (web dashboard)
- Baostock (A-share data source)
- Pandas, NumPy, PyArrow (data processing)
- Plotly (interactive charts)
- Scipy (statistics & optimization)
- Loguru (logging)

## Project Structure
`
quant-learning/
  dashboard.py              # Web dashboard (Flask + Plotly)
  explore.py                # Data exploration tool
  report.py                 # Report generation
  start.bat                 # One-click launcher
  scripts/
    live_engine.py           # Real-time prediction engine (v4.0)
    factor_library.py        # Factor definitions & strategy clusters
    factor_engine.py         # Factor computation engine
    ic_estimator.py          # IC estimation pipeline
    ic_decay.py              # IC decay tracking & factor health
    portfolio_builder.py     # Portfolio construction (equal/ICIR/MaxSharpe/RiskParity)
    risk_manager.py          # Risk control module
    paper_trading_engine.py  # Simulated trading engine
    data_fetcher.py          # Baostock data fetcher
    data_orchestrator.py     # Data loading & caching layer
    multi_factor.py          # Backtesting script
    walkforward.py           # Walk-forward validation
    optimize.py              # Strategy optimization
    config.py                # Global configuration
    auth.py                  # Dashboard authentication
  data/                      # Market data storage
  logs/                      # Log files
`

## Quick Start
`ash
pip install -r requirements.txt
python dashboard.py          # Start web dashboard at http://localhost:8050
`

Or double-click start.bat.

## Running Predictions
`ash
python -m scripts.live_engine    # Run real-time stock prediction
python multi_factor.py --plot    # Run backtest with charts
python walkforward.py --plot     # Walk-forward validation
`

## Configuration
Edit scripts/config.py to adjust:
- Liquidity filters, factor weights, risk parameters
- IC estimation windows, decay thresholds
- Commission rates, slippage assumptions

## License
MIT

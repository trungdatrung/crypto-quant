# crypto-quant

Predict the **direction** (up/down) of BTC, ETH and SOL with classical machine learning, then backtest a
**fee-aware** long/flat strategy using walk-forward validation and compare it against Buy & Hold.

!!! warning "Educational project"
    This is a course project, not investment advice.

## Quick start

```bash
git clone https://github.com/the-daonm/crypto-quant.git && cd crypto-quant
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebook]"

pytest                                            # test suite
python -m cq.pipeline --symbol BTC/USDT --tf 1d   # metrics for the full, design and holdout periods
streamlit run app.py                              # dashboard at http://localhost:8501
```

## How it works

| Step | Module | What it does |
|---|---|---|
| Data | [`cq.data`](api.md#cq.data) | Downloads OHLCV bars from the Binance public API via ccxt and caches them as parquet |
| Features | [`cq.features`](api.md#cq.features) | 21 causal technical features (momentum, volatility, RSI, MACD, …) |
| Labels | [`cq.labels`](api.md#cq.labels) | Price direction over the next `horizon` bars (default 3) |
| Models | [`cq.models`](api.md#cq.models) | Always-long baseline, Logistic Regression, LightGBM |
| Validation | [`cq.validation`](api.md#cq.validation) | Expanding-window walk-forward with an embargo |
| Backtest | [`cq.backtest`](api.md#cq.backtest) | Long/flat positions with a hysteresis band and fees |
| Metrics | [`cq.metrics`](api.md#cq.metrics) | Accuracy, AUC, Sharpe, max drawdown, … |
| Pipeline | [`cq.pipeline`](api.md#cq.pipeline) | Runs everything end to end; design/holdout reporting |
| Paper trading | [`cq.signal`](api.md#cq.signal) | Logs the signal for the last closed bar; see [Paper trading](paper-trading.md) |

## Key results

On the 2025–2026 **holdout** (data never used to choose the configuration), the default LightGBM strategy reaches an
average Sharpe ratio of **+0.10** vs **−0.11** for Buy & Hold, beats Buy & Hold on 4 of 6 pairs, and cuts the average
maximum drawdown from −66% to −51%. The main lesson: **transaction costs decide whether a weak signal is usable.**

Full tables and discussion are in the
[README](https://github.com/the-daonm/crypto-quant#results-out-of-sample-01-fee-data-up-to-nov-2026).
A beginner-friendly explainer in Vietnamese is available under [Giải thích](GIAI-THICH.md).

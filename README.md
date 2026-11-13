# crypto-quant

[![CI](https://github.com/the-daonm/crypto-quant/actions/workflows/ci.yml/badge.svg)](https://github.com/the-daonm/crypto-quant/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://the-daonm.github.io/crypto-quant/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

Predict the **direction** (up/down) of BTC, ETH and SOL with classical machine learning, then backtest a
**fee-aware** long/flat strategy using walk-forward validation and compare it against Buy & Hold.

> An educational project, not investment advice.
> Documentation, including the API reference: **https://the-daonm.github.io/crypto-quant/**
> New to crypto, ML or backtesting? Read **[docs/GIAI-THICH.md](docs/GIAI-THICH.md)** first: a beginner-friendly
> explainer (in Vietnamese) covering all the background and every part of the project.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # fish: source .venv/bin/activate.fish
pip install -e ".[dev,notebook]"

pytest                                                 # 13 tests: look-ahead, backtest, walk-forward
python -m cq.pipeline --symbol BTC/USDT --tf 1d        # full-period + design/holdout metrics, CSVs in results/
python -m cq.pipeline --symbol ETH/USDT --tf 4h --horizon 1 --band 0   # reproduce the original v1 setup
streamlit run app.py                                   # dashboard at http://localhost:8501
python -m cq.signal                                    # paper trading: log today's signal to paper/log.csv
jupyter notebook notebooks/eda.ipynb                   # EDA and results analysis
```

The first run downloads data and caches it in `data/` (not committed). If Binance is blocked on your network,
set `exchange` in `config.yaml` to another ccxt exchange such as `okx`, `bybit` or `kraken`.

## Data

OHLCV bars come from the **Binance public market-data API** via [ccxt](https://github.com/ccxt/ccxt): no API key
required, from 2019-01-01 (SOL from 2020-08) to the present, on `1d` and `4h` timeframes.
The repository contains no raw data; data is used for educational purposes only. Bars that have not closed yet are dropped.

## Method

| Step | Approach |
|---|---|
| Features (`features.py`) | 21 technical features: lagged returns, momentum, volatility, RSI, MACD histogram, Bollinger %B, ATR, distance to MA20/50/200, volume z-score, … All features are **causal** (use data ≤ t only). |
| Labels (`labels.py`) | `y_t = 1` if `close_{t+h} > close_t`, with `h = 3` bars by default |
| Models (`models.py`) | Always-long baseline · Logistic Regression (standardised, C=0.1) · LightGBM (shallow, heavily regularised) |
| Validation (`validation.py`) | **Expanding-window walk-forward**: ≥ ~2 years of training, ~3-month test blocks, an embargo of `h` bars (labels overlap), model refit in every fold. All reported results are out-of-sample. |
| Signals (`backtest.positions`) | **Hysteresis**: enter when p > 0.55, exit when p < 0.45, otherwise keep the current position. The threshold can be chosen `auto` on training data |
| Backtest (`backtest.py`) | Signal at the close of bar t, position held until the close of bar t+1; 0.1% fee per position change; long/flat only |
| Design / holdout | The configuration was selected **only on data before 2025-01-01**; 2025 onwards is a holdout evaluated once |
| Metrics (`metrics.py`) | Accuracy, AUC, F1 · total return, CAGR, Sharpe, max drawdown, exposure, win rate, number of trades |

**Paper trading** (`signal.py`): after each bar closes, the model's signal is logged before the future is known,
on a schedule, with retries, a marked cache fallback and idempotent runs. See [docs/paper-trading.md](docs/paper-trading.md).

The test suite in `tests/` checks that:
- corrupting future data never changes past features;
- multi-bar labels are aligned correctly while the backtest still uses 1-bar returns;
- always-long with zero fees reproduces Buy & Hold exactly, and fees are charged once per position change;
- hysteresis holds the position inside the band and reduces trading;
- walk-forward folds never overlap and respect the embargo;
- the paper-trading log reproduces the backtest bar by bar and never logs the same bar twice.

## Results (out-of-sample, 0.1% fee, data up to Nov 2026)

**v1** = the original version (1-bar label, 0.5 threshold, no band).
**v2** = the current version (3-bar label, ±0.05 hysteresis band).

### Average over 6 pairs (3 coins × 2 timeframes)

| | Design 2021–2024 Sharpe | Design max DD | **Holdout 2025–2026 Sharpe** | Holdout max DD | Beats B&H on holdout | Trades (full period) |
|---|---:|---:|---:|---:|---:|---:|
| Buy & Hold | **0.97** | −72% | −0.11 | −66% | – | – |
| LightGBM v1 | 0.28 | −70% | −1.28 | −75% | 1/6 | ~1,760 |
| LogReg v2 | 0.68 | −63% | **0.10** | **−51%** | **4/6** | ~200 |
| **LightGBM v2** | 0.87 | **−59%** | **0.10** | **−51%** | **4/6** | ~340 |

### Holdout Jan 2025 → Nov 2026 per pair (LightGBM v2 vs Buy & Hold)

| Pair | B&H return | v2 return | B&H Sharpe | v2 Sharpe | B&H max DD | v2 max DD |
|---|---:|---:|---:|---:|---:|---:|
| BTC 1d | −20% | −11% | −0.07 | −0.00 | −53% | −54% |
| BTC 4h | −21% | −20% | −0.08 | −0.12 | −53% | −48% |
| ETH 1d | −36% | **+49%** | −0.02 | 0.66 | −68% | −44% |
| ETH 4h | −38% | **+32%** | −0.07 | 0.54 | −68% | −54% |
| SOL 1d | −55% | −47% | −0.19 | −0.34 | −76% | −65% |
| SOL 4h | −55% | −24% | −0.22 | −0.15 | −78% | −41% |

### Fee sensitivity: BTC 4h, LightGBM

| Fee per trade | 0% | 0.02% | 0.05% | 0.10% |
|---|---:|---:|---:|---:|
| v1 return (5,955 trades) | +1765% | +467% | −5% | **−95%** |
| v2 return (907 trades) | +903% | +736% | +537% | **+305%** |

### Findings

1. **The signal is real but weak**: out-of-sample accuracy is about 50–55% and AUC about 0.50–0.56.
2. **Fees decide everything.** v1 has a signal, but thousands of trades wipe out the capital. v2 trades 5–10× less
   thanks to a less noisy label and the hysteresis band, lifting the average design-period Sharpe from 0.28 to 0.87.
3. **The improvement holds on the holdout** (data never used for selection): Sharpe goes from −1.28 to +0.10,
   v2 beats Buy & Hold on 4 of 6 pairs, and the average max drawdown is −51% vs −66%.
4. **In the bull market of the design period, v2 still does not consistently beat Buy & Hold.** The model's main
   value is *reducing risk in downturns*, not higher returns.
5. Limitations: the holdout covers only ~22 months; the 3 coins are highly correlated; ~150 configurations were tried
   (on the design period) during development, so selection bias remains possible. Slippage is not modelled yet.

### Future work
- Volatility-based position sizing, short positions
- Non-price data: funding rates, open interest, on-chain metrics
- More coins and markets for robustness; hyperparameter tuning with nested walk-forward

See the [issue tracker](https://github.com/the-daonm/crypto-quant/issues) for planned work.

## Project layout

```
src/cq/        data · features · labels · models · validation · backtest · metrics · pipeline · signal
app.py         Streamlit dashboard
pages/         extra dashboard pages (paper trading)
notebooks/     EDA and results analysis
docs/          GIAI-THICH.md: beginner explainer (Vietnamese)
tests/         pytest suite
.github/       issue and pull request templates
config.yaml    pairs, timeframes, fee, horizon, band, holdout, walk-forward windows
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow (fork → branch → PR → review),
commit guidelines and methodology rules. Participation is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
Changes are tracked in [CHANGELOG.md](CHANGELOG.md).

## License

Released under the [MIT License](LICENSE).

Built on these open-source libraries (thanks to their authors):

| Library | Used for | License |
|---|---|---|
| [ccxt](https://github.com/ccxt/ccxt) | Exchange data access | MIT |
| [pandas](https://pandas.pydata.org/) · [NumPy](https://numpy.org/) | Data processing | BSD-3-Clause |
| [scikit-learn](https://scikit-learn.org/) | Logistic Regression, metrics | BSD-3-Clause |
| [LightGBM](https://github.com/microsoft/LightGBM) | Gradient boosting | MIT |
| [Streamlit](https://streamlit.io/) | Dashboard | Apache-2.0 |
| [Plotly](https://plotly.com/python/) | Charts | MIT |
| [PyArrow](https://arrow.apache.org/) | Parquet I/O | Apache-2.0 |
| [PyYAML](https://pyyaml.org/) | Configuration | MIT |
| [pytest](https://pytest.org/) · [ruff](https://github.com/astral-sh/ruff) | Testing, linting | MIT |

Market data is obtained from the Binance public API and is **not** covered by this repository's license;
the repository does not redistribute any data.

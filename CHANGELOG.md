# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Auto-merge for Dependabot patch and minor updates after CI passes; major updates still need a review.
- Paper trading (#10): `python -m cq.signal` logs the signal for the last closed bar to `paper/log.csv`, with
  retries, a marked cache fallback, idempotent runs and a failure on stale data; a scheduled workflow publishes
  the log to the `paper-log` branch; a *Paper trading* dashboard page shows the equity against Buy & Hold.

## [0.2.0] - 2026-11-12

### Added
- Multi-bar labels (`horizon`) with a matching embargo in walk-forward validation.
- Hysteresis band (`band`) on entry/exit signals to reduce trading.
- Separate design and holdout (`holdout_start`) reporting in the CLI, dashboard and notebook.
- `docs/GIAI-THICH.md`: beginner explainer (Vietnamese), including 1d vs 4h and live prediction.
- MIT License, contributing guide, Code of Conduct (Contributor Covenant 2.1) and security policy.
- Issue templates (bug report, feature request, experiment) and a pull request template.
- CI on GitHub Actions: `ruff` lint/format checks and `pytest` on Python 3.11–3.13.
- Documentation site built with MkDocs and deployed to GitHub Pages, with an API reference.
- Dependabot updates for pip and GitHub Actions; release workflow that builds and publishes sdist/wheel on tags.
- Project metadata in `pyproject.toml` (license, URLs, keywords, classifiers).

### Changed
- Default configuration is now `horizon = 3`, `band = 0.05`. On the 2025–2026 holdout, the average LightGBM
  Sharpe ratio improved from −1.28 to +0.10.
- Code, dashboard, notebook and project documentation are now in English;
  the beginner explainer `docs/GIAI-THICH.md` remains in Vietnamese.

### Known issues
- `cq.pipeline.load_config()` reads `config.yaml` from the source tree, so the CLI only works from a source checkout
  (`pip install -e .`), not from an installed wheel.

## [0.1.0] - 2026-11-12

### Added
- OHLCV download from Binance via ccxt with a parquet cache.
- 21 causal technical features and next-bar direction labels.
- Baseline, Logistic Regression and LightGBM models; walk-forward validation; `auto` threshold selection.
- Fee-aware long/flat backtest; classification and trading metrics.
- `python -m cq.pipeline` CLI, Streamlit dashboard and EDA notebook.

[Unreleased]: https://github.com/the-daonm/crypto-quant/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/the-daonm/crypto-quant/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/the-daonm/crypto-quant/releases/tag/v0.1.0

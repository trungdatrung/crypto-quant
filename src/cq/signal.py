"""Paper trading: compute the signal for the last closed bar and append it to a log.

Each run downloads the latest bars, trains on the full labelled history, predicts the last closed
bar and applies the hysteresis rule starting from the previous position found in the log. The row
also books the profit or loss of the position held since the previous logged bar.

Nothing is traded: the log is a record made *before* the future is known, to be checked later.

Usage: python -m cq.signal                      # every pair and timeframe in the configuration
       python -m cq.signal --symbol BTC/USDT --tf 1d
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

from cq import data
from cq.features import build_features
from cq.labels import make_dataset
from cq.models import MODEL_NAMES, make_model
from cq.pipeline import load_config

LOG_PATH = Path("paper") / "log.csv"
# Log published by the scheduled workflow (`main` is protected, so it lives on its own branch).
LOG_URL = "https://raw.githubusercontent.com/the-daonm/crypto-quant/paper-log/log.csv"
LOG_COLUMNS = [
    "run_time",  # when the script ran (UTC)
    "bar_close",  # close time of the bar the signal is computed on (UTC)
    "symbol",
    "timeframe",
    "model",
    "proba",  # predicted probability that the price is higher `horizon` bars later
    "position",  # 1 = long, 0 = flat, held from this bar's close
    "close",
    "pnl",  # log return booked by this row: previous position × price change, minus the fee
    "equity",  # paper-trading equity, starting at 1
    "bh_equity",  # Buy & Hold equity over the same bars
    "commit",
    "source",  # "live" = fresh download, "cache" = download failed and cached bars were used
]
MAX_BARS_LATE = 2


class StaleDataError(RuntimeError):
    """The newest closed bar is older than expected, so no signal is logged."""


@dataclass
class Signal:
    bar_close: pd.Timestamp
    proba: float
    close: float


def bar_length(timeframe: str) -> pd.Timedelta:
    """Duration of one bar, e.g. "4h" -> 4 hours (uses ccxt's timeframe parser)."""
    return pd.Timedelta(seconds=ccxt.Exchange.parse_timeframe(timeframe))


def next_position(proba: float, previous: float, threshold: float = 0.5, band: float = 0.0) -> float:
    """One step of `cq.backtest.positions`: enter above the band, exit below it, otherwise hold."""
    if proba > threshold + band:
        return 1.0
    if proba < threshold - band:
        return 0.0
    if band == 0:
        return 0.0  # proba == threshold: the plain rule `proba > threshold` is false
    return float(previous)


def fetch_latest(
    symbol: str,
    timeframe: str,
    since: str,
    exchange: str,
    attempts: int = 3,
    base_delay: float = 2.0,
    loader: Callable[..., pd.DataFrame] = data.load_ohlcv,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[pd.DataFrame, str]:
    """Download fresh bars, retrying network errors with exponential backoff.

    Returns `(ohlcv, source)`. If every attempt fails, the cached bars are returned with
    `source = "cache"` so that the log shows the data may be stale; without a cache the last
    network error is raised.
    """
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            return loader(symbol, timeframe, since, exchange, refresh=True), "live"
        except ccxt.NetworkError as e:  # includes RequestTimeout and ExchangeNotAvailable
            error = e
            if attempt < attempts - 1:
                sleep(base_delay * 2**attempt)
    cache = data._cache_path(exchange, symbol, timeframe)
    if not cache.exists():
        raise error
    return pd.read_parquet(cache), "cache"


def latest_signal(ohlcv: pd.DataFrame, timeframe: str, model: str, horizon: int) -> Signal:
    """Train on every labelled bar and predict the last closed bar.

    The last `horizon` bars have no label yet, so `make_dataset` leaves them out of training.
    """
    features = build_features(ohlcv)
    X, y, _ = make_dataset(features, ohlcv["close"], horizon=horizon)
    last = features.iloc[[-1]]
    if last.isna().any(axis=None):
        raise ValueError("Not enough history to compute the features of the last bar.")
    proba = float(make_model(model).fit(X, y).predict_proba(last)[0, 1])
    bar_close = ohlcv.index[-1] + bar_length(timeframe)
    return Signal(bar_close, proba, float(ohlcv["close"].iloc[-1]))


def bars_late(bar_close: pd.Timestamp, now: pd.Timestamp, timeframe: str) -> float:
    """How many bars have closed since `bar_close` (0 = `bar_close` is the newest closed bar)."""
    return (now - bar_close) / bar_length(timeframe)


def read_log(path: Path = LOG_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(path, parse_dates=["run_time", "bar_close"])


def record(
    log: pd.DataFrame,
    signal: Signal,
    *,
    symbol: str,
    timeframe: str,
    model: str,
    run_time: pd.Timestamp,
    threshold: float = 0.5,
    band: float = 0.0,
    fee: float = 0.001,
    commit: str = "unknown",
    source: str = "live",
) -> tuple[pd.DataFrame, dict | None]:
    """Append the signal to the log and return `(log, row)`; `row` is None if nothing was added.

    Idempotent: a bar that is already in the log for this pair and timeframe (or is older than the
    last logged bar) is ignored. The fee is booked on the row where the position changes, so the
    cumulative result equals `cq.backtest.backtest` on the same probabilities and prices.
    """
    history = log[(log["symbol"] == symbol) & (log["timeframe"] == timeframe)]
    if len(history):
        prev = history.iloc[-1]
        if signal.bar_close <= prev["bar_close"]:
            return log, None
        prev_position, prev_equity, prev_bh = float(prev["position"]), float(prev["equity"]), float(prev["bh_equity"])
        bar_return = float(np.log(signal.close / prev["close"]))
    else:
        prev_position, prev_equity, prev_bh, bar_return = 0.0, 1.0, 1.0, 0.0

    position = next_position(signal.proba, prev_position, threshold, band)
    pnl = prev_position * bar_return + float(np.log1p(-fee * abs(position - prev_position)))
    row = {
        "run_time": run_time,
        "bar_close": signal.bar_close,
        "symbol": symbol,
        "timeframe": timeframe,
        "model": model,
        "proba": signal.proba,
        "position": position,
        "close": signal.close,
        "pnl": pnl,
        "equity": prev_equity * float(np.exp(pnl)),
        "bh_equity": prev_bh * float(np.exp(bar_return)),
        "commit": commit,
        "source": source,
    }
    new = pd.DataFrame([row], columns=LOG_COLUMNS)
    return (new if log.empty else pd.concat([log, new], ignore_index=True)), row


def current_commit() -> str:
    """Short hash of the code that produced the signal (`GITHUB_SHA` on CI, else `git rev-parse`)."""
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha[:7]
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(
    symbols: list[str],
    timeframes: list[str],
    cfg: dict,
    model: str = "lightgbm",
    log_path: Path = LOG_PATH,
    now: pd.Timestamp | None = None,
    fetch: Callable[..., tuple[pd.DataFrame, str]] = fetch_latest,
) -> list[str]:
    """Log one signal per pair and timeframe. Returns the list of problems (empty = success).

    A stale or failed pair never stops the other pairs; the log is written before returning.
    """
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    log, commit, problems = read_log(log_path), current_commit(), []
    for symbol in symbols:
        for timeframe in timeframes:
            tag = f"{symbol} {timeframe}"
            try:
                ohlcv, source = fetch(symbol, timeframe, cfg["since"], cfg["exchange"])
                signal = latest_signal(ohlcv, timeframe, model, cfg.get("horizon", 1))
                late = bars_late(signal.bar_close, now, timeframe)
                if late > MAX_BARS_LATE:
                    raise StaleDataError(
                        f"newest bar closed at {signal.bar_close} ({late:.1f} bars ago, source: {source})"
                    )
            except (ccxt.BaseError, StaleDataError, ValueError) as e:
                problems.append(f"{tag}: {type(e).__name__}: {e}")
                continue
            log, row = record(
                log,
                signal,
                symbol=symbol,
                timeframe=timeframe,
                model=model,
                run_time=now,
                threshold=cfg["threshold"],
                band=cfg.get("band", 0.0),
                fee=cfg["fee"],
                commit=commit,
                source=source,
            )
            if row is None:
                print(f"{tag}: bar closing {signal.bar_close} is already logged, nothing to do")
            else:
                action = "LONG" if row["position"] else "FLAT"
                print(
                    f"{tag}: bar closing {signal.bar_close} | p = {signal.proba:.3f} | {action} | "
                    f"equity {row['equity']:.4f} vs Buy & Hold {row['bh_equity']:.4f} | {source}"
                )
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(log_path, index=False)
    return problems


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", action="append", help="pair to run; repeat for several (default: all in the config)")
    ap.add_argument("--tf", action="append", help="timeframe to run; repeat for several (default: all in the config)")
    ap.add_argument("--model", default="lightgbm", choices=MODEL_NAMES)
    ap.add_argument("--exchange", default=None, help="override the exchange from the configuration")
    ap.add_argument("--log", type=Path, default=LOG_PATH, help=f"log file (default: {LOG_PATH.as_posix()})")
    args = ap.parse_args(argv)
    if args.exchange:
        cfg = {**cfg, "exchange": args.exchange}

    problems = run(args.symbol or cfg["symbols"], args.tf or cfg["timeframes"], cfg, args.model, args.log)
    for p in problems:
        print(f"ERROR {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

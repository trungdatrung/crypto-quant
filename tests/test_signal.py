import ccxt
import numpy as np
import pandas as pd
import pytest

from cq import data, signal
from cq.backtest import backtest, positions
from cq.signal import Signal, bars_late, fetch_latest, latest_signal, next_position, read_log, record

CFG = {"since": "2020-01-01", "exchange": "binance", "horizon": 3, "threshold": 0.5, "band": 0.05, "fee": 0.001}


def _synthetic_ohlcv(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = close * np.exp(rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.uniform(1e3, 1e4, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def _record_all(proba: pd.Series, close: pd.Series, **kwargs) -> pd.DataFrame:
    log = read_log("does-not-exist.csv")
    for t in proba.index:
        log, _ = record(
            log,
            Signal(t, float(proba[t]), float(close[t])),
            symbol="BTC/USDT",
            timeframe="1d",
            model="lightgbm",
            run_time=t + pd.Timedelta(minutes=5),
            **kwargs,
        )
    return log


@pytest.mark.parametrize("band", [0.0, 0.05])
def test_next_position_matches_backtest_positions(band):
    rng = np.random.default_rng(3)
    proba = pd.Series(np.round(0.5 + rng.normal(0, 0.06, 300), 2))  # rounded, so ties with the threshold occur
    expected = positions(proba, 0.5, band)
    previous, stepwise = 0.0, []
    for p in proba:
        previous = next_position(p, previous, 0.5, band)
        stepwise.append(previous)
    assert stepwise == expected.tolist()


def test_log_equity_matches_backtest():
    """Logging bar by bar must give the same equity as the vectorized backtest on the same inputs."""
    rng = np.random.default_rng(4)
    idx = pd.date_range("2026-01-01", periods=120, freq="D", tz="UTC")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, 120))), index=idx)
    proba = pd.Series(0.5 + rng.normal(0, 0.06, 120), index=idx)

    log = _record_all(proba, close, threshold=0.5, band=0.05, fee=0.001)

    fwd = np.log(close).diff().shift(-1).fillna(0.0)  # the last position has not earned anything yet
    bt = backtest(proba, fwd, threshold=0.5, fee=0.001, band=0.05)
    assert log["position"].tolist() == bt["position"].tolist()
    assert log["equity"].iloc[-1] == pytest.approx(bt["equity"].iloc[-1])
    assert log["bh_equity"].iloc[-1] == pytest.approx(close.iloc[-1] / close.iloc[0])


def test_first_row_pays_the_entry_fee_and_next_row_books_the_previous_position():
    idx = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    close = pd.Series([100.0, 110.0, 99.0], index=idx)
    proba = pd.Series([0.9, 0.1, 0.1], index=idx)  # long, exit, stay flat
    log = _record_all(proba, close, threshold=0.5, band=0.05, fee=0.01)

    assert log["position"].tolist() == [1.0, 0.0, 0.0]
    assert log["equity"].tolist() == pytest.approx([0.99, 0.99 * 1.1 * 0.99, 0.99 * 1.1 * 0.99])
    assert log["bh_equity"].tolist() == pytest.approx([1.0, 1.1, 0.99])


def test_recording_the_same_bar_twice_is_a_no_op():
    idx = pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC")
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=idx)
    proba = pd.Series(0.9, index=idx)
    log = _record_all(proba, close)

    again, row = record(
        log, Signal(idx[-1], 0.1, 50.0), symbol="BTC/USDT", timeframe="1d", model="lightgbm", run_time=idx[-1]
    )
    assert row is None and again is log
    older, row = record(
        log, Signal(idx[1], 0.1, 50.0), symbol="BTC/USDT", timeframe="1d", model="lightgbm", run_time=idx[-1]
    )
    assert row is None and older is log


def test_pairs_and_timeframes_are_logged_independently():
    t = pd.Timestamp("2026-01-01", tz="UTC")
    log = read_log("does-not-exist.csv")
    for symbol, timeframe in [("BTC/USDT", "1d"), ("ETH/USDT", "1d"), ("BTC/USDT", "4h"), ("BTC/USDT", "1d")]:
        log, _ = record(log, Signal(t, 0.9, 100.0), symbol=symbol, timeframe=timeframe, model="logreg", run_time=t)
    assert len(log) == 3  # the repeated BTC/USDT 1d bar is ignored
    assert not log.duplicated(["symbol", "timeframe", "bar_close"]).any()


def test_bars_late():
    close = pd.Timestamp("2026-01-02 00:00", tz="UTC")
    assert bars_late(close, pd.Timestamp("2026-01-02 00:05", tz="UTC"), "1d") < 1
    assert bars_late(close, pd.Timestamp("2026-01-05 00:05", tz="UTC"), "1d") > 3
    assert bars_late(close, pd.Timestamp("2026-01-02 12:00", tz="UTC"), "4h") == 3


def test_latest_signal_predicts_the_last_closed_bar():
    ohlcv = _synthetic_ohlcv()
    s = latest_signal(ohlcv, "1d", "logreg", horizon=3)
    assert s.bar_close == ohlcv.index[-1] + pd.Timedelta(days=1)
    assert s.close == ohlcv["close"].iloc[-1]
    assert 0.0 <= s.proba <= 1.0


def test_latest_signal_needs_enough_history():
    with pytest.raises(ValueError, match="Not enough history"):
        latest_signal(_synthetic_ohlcv(100), "1d", "logreg", horizon=3)  # the 200-bar moving average is still NaN


def test_fetch_retries_with_exponential_backoff():
    calls, delays = [], []

    def flaky_loader(symbol, timeframe, since, exchange, refresh):
        calls.append(refresh)
        if len(calls) < 3:
            raise ccxt.RequestTimeout("timed out")
        return _synthetic_ohlcv(10)

    df, source = fetch_latest("BTC/USDT", "1d", "2020-01-01", "binance", loader=flaky_loader, sleep=delays.append)
    assert source == "live" and len(df) == 10
    assert calls == [True, True, True]  # always a fresh download, never the cache
    assert delays == [2.0, 4.0]


def test_fetch_falls_back_to_the_cache_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    cached = _synthetic_ohlcv(10)
    cached.to_parquet(data._cache_path("binance", "BTC/USDT", "1d"))

    def broken_loader(*args, **kwargs):
        raise ccxt.ExchangeNotAvailable("HTTP 451")

    df, source = fetch_latest("BTC/USDT", "1d", "2020-01-01", "binance", loader=broken_loader, sleep=lambda s: None)
    assert source == "cache"
    pd.testing.assert_frame_equal(df, cached, check_freq=False)


def test_fetch_without_a_cache_raises_the_network_error(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)

    def broken_loader(*args, **kwargs):
        raise ccxt.RequestTimeout("timed out")

    with pytest.raises(ccxt.RequestTimeout):
        fetch_latest("BTC/USDT", "1d", "2020-01-01", "binance", loader=broken_loader, sleep=lambda s: None)


def test_run_is_idempotent_and_round_trips_through_the_csv(tmp_path):
    ohlcv = _synthetic_ohlcv()
    log_path = tmp_path / "paper" / "log.csv"
    now = ohlcv.index[-1] + pd.Timedelta(days=1, minutes=5)

    def fetch(symbol, timeframe, since, exchange):
        return ohlcv, "live"

    assert signal.run(["BTC/USDT"], ["1d"], CFG, "logreg", log_path, now=now, fetch=fetch) == []
    first = log_path.read_text()
    assert signal.run(["BTC/USDT"], ["1d"], CFG, "logreg", log_path, now=now, fetch=fetch) == []
    assert log_path.read_text() == first  # the second run for the same bar changes nothing

    log = read_log(log_path)
    assert list(log.columns) == signal.LOG_COLUMNS
    assert len(log) == 1 and log["source"].iloc[0] == "live"
    assert log["bar_close"].iloc[0] == ohlcv.index[-1] + pd.Timedelta(days=1)


def test_run_marks_cached_data_and_books_the_next_bar(tmp_path):
    ohlcv = _synthetic_ohlcv()
    log_path = tmp_path / "log.csv"
    sources = iter(["live", "cache"])

    def fetch_until(n):
        return lambda symbol, timeframe, since, exchange: (ohlcv.iloc[:n], next(sources))

    for n in (399, 400):
        now = ohlcv.index[n - 1] + pd.Timedelta(days=1, minutes=5)
        assert signal.run(["BTC/USDT"], ["1d"], CFG, "logreg", log_path, now=now, fetch=fetch_until(n)) == []

    log = read_log(log_path)
    assert log["source"].tolist() == ["live", "cache"]
    assert log["bh_equity"].iloc[1] == pytest.approx(ohlcv["close"].iloc[399] / ohlcv["close"].iloc[398])


def test_run_reports_stale_data_and_does_not_log_it(tmp_path):
    ohlcv = _synthetic_ohlcv()
    log_path = tmp_path / "log.csv"
    now = ohlcv.index[-1] + pd.Timedelta(days=5)

    problems = signal.run(["BTC/USDT"], ["1d"], CFG, "logreg", log_path, now=now, fetch=lambda *a: (ohlcv, "cache"))
    assert len(problems) == 1 and "StaleDataError" in problems[0]
    assert read_log(log_path).empty


def test_run_continues_after_a_failed_pair(tmp_path):
    ohlcv = _synthetic_ohlcv()
    now = ohlcv.index[-1] + pd.Timedelta(days=1, minutes=5)

    def fetch(symbol, timeframe, since, exchange):
        if symbol == "BTC/USDT":
            raise ccxt.RequestTimeout("timed out")
        return ohlcv, "live"

    problems = signal.run(["BTC/USDT", "ETH/USDT"], ["1d"], CFG, "logreg", tmp_path / "log.csv", now=now, fetch=fetch)
    assert len(problems) == 1 and problems[0].startswith("BTC/USDT 1d")
    assert read_log(tmp_path / "log.csv")["symbol"].tolist() == ["ETH/USDT"]

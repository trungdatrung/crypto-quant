# Paper trading

A backtest replays the past. **Paper trading** records the model's decision right after each bar closes, before
the future exists, and checks it later. Nothing is traded with real money. It is the most honest test in the
project: there is no way to leak future data or to choose a configuration after seeing the result.

## Run it

```bash
python -m cq.signal                               # every pair and timeframe in config.yaml
python -m cq.signal --symbol BTC/USDT --tf 1d     # a single pair
```

Each run, for every pair and timeframe:

1. downloads the latest bars through ccxt (the bar that has not closed yet is dropped);
2. trains the model (LightGBM by default) on the **full** labelled history;
3. predicts the last closed bar and applies the hysteresis rule, starting from the previous position in the log;
4. appends one row to `paper/log.csv` and books the profit or loss of the position held since the previous row.

## The log

| Column | Meaning |
|---|---|
| `run_time` | when the script ran (UTC) |
| `bar_close` | close time of the bar the signal is computed on (UTC) |
| `symbol`, `timeframe`, `model` | what was predicted, and with which model |
| `proba` | predicted probability that the price is higher `horizon` bars later |
| `position` | 1 = long, 0 = flat, held from this bar's close |
| `close` | close price of the bar |
| `pnl` | log return booked by this row: previous position × price change, minus the fee if the position changed |
| `equity`, `bh_equity` | paper-trading and Buy & Hold equity, both starting at 1 |
| `commit` | version of the code that produced the row |
| `source` | `live` = fresh download; `cache` = the download failed and cached bars were used |

The fee is booked on the row where the position changes, so the equity equals what `cq.backtest.backtest` gives for
the same probabilities and prices (covered by `test_log_equity_matches_backtest`).

## Reliability

- **Retries.** Network errors (timeouts, exchange unavailable) are retried 3 times with exponential backoff (2 s, 4 s).
- **Cache fallback, never silent.** If the download still fails, the cached bars are used and the row is marked `cache`.
- **Idempotent.** A bar that is already logged for a pair and timeframe is skipped, so running twice changes nothing.
  This also lets one schedule serve both timeframes: the `1d` signal is only logged on the first run after 00:00 UTC.
- **Stale data fails loudly.** If the newest bar closed more than 2 bars ago, nothing is logged for that pair and the
  command exits with status 1. Other pairs are still processed.

## Schedule (GitHub Actions)

`.github/workflows/paper-trading.yml` runs at minute 5 of every fourth hour (UTC), a few minutes after each `4h` bar
closes, and can be started manually (*Actions → Paper trading → Run workflow*).

`main` is protected, so the workflow commits the log to a separate **`paper-log`** branch (created on the first run)
instead. The *Paper trading* page of the dashboard reads a local `paper/log.csv` if there is one, and otherwise the
file on that branch. When a pair fails or its data is stale, the job fails and GitHub notifies the team; rows that
succeeded are still committed.

### Maintainer checklist (first run)

- Start the workflow manually once and check that the `paper-log` branch appears.
- GitHub-hosted runners are in the US, where **Binance may refuse requests (HTTP 451)**. This could not be tested from
  a fork. If the first run fails for that reason, change the workflow command to an exchange that is reachable, e.g.
  `python -m cq.signal --log paper/log.csv --exchange kraken --symbol BTC/USD --symbol ETH/USD --symbol SOL/USD`,
  and note the change here. Keep one exchange for the whole log: prices differ slightly between exchanges.
- Scheduled workflows are paused by GitHub after 60 days without repository activity.

Alternatively, run it from a local cron job (`5 */4 * * *`, UTC) and commit `log.csv` to the `paper-log` branch.

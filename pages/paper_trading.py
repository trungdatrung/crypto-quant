"""Streamlit page: paper-trading log (equity vs Buy & Hold). Opens from the sidebar of `app.py`."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cq.signal import LOG_PATH, LOG_URL, read_log

st.set_page_config(page_title="Paper trading · Crypto Quant", layout="wide")


@st.cache_data(ttl=600, show_spinner="Loading the paper-trading log…")
def get_log() -> tuple[pd.DataFrame, str]:
    """Prefer a local log (developer machine); otherwise read the one published by the scheduled workflow."""
    if LOG_PATH.exists():
        return read_log(LOG_PATH), f"local file `{LOG_PATH.as_posix()}`"
    return pd.read_csv(LOG_URL, parse_dates=["run_time", "bar_close"]), "the `paper-log` branch on GitHub"


st.title("Paper trading")
st.caption(
    "Signals recorded right after each bar closed, before the future was known. Nothing is traded with real money. "
    "An educational project, not investment advice."
)

try:
    log, origin = get_log()
except Exception as e:  # no local log and the published one is unreachable or does not exist yet
    st.info(
        "No paper-trading log is available yet. Create one with `python -m cq.signal`, "
        "or wait for the scheduled workflow to publish it."
    )
    st.caption(f"Details: {type(e).__name__}: {e}")
    st.stop()

if log.empty:
    st.info("The paper-trading log is empty. Run `python -m cq.signal` to record the first signals.")
    st.stop()

pairs = log[["symbol", "timeframe"]].drop_duplicates()
symbol = st.sidebar.selectbox("Pair", pairs["symbol"].unique())
tf = st.sidebar.selectbox("Timeframe", pairs.loc[pairs["symbol"] == symbol, "timeframe"].unique())
rows = log[(log["symbol"] == symbol) & (log["timeframe"] == tf)].set_index("bar_close")
last = rows.iloc[-1]

st.subheader(f"{symbol} · {tf} · {last['model']}")
st.caption(f"{len(rows)} bars logged from {rows.index[0]:%Y-%m-%d %H:%M} to {rows.index[-1]:%Y-%m-%d %H:%M} UTC")

c = st.columns(5)
c[0].metric("Paper return", f"{last['equity'] - 1:.2%}", f"B&H {last['bh_equity'] - 1:.2%}", delta_color="off")
c[1].metric("Current position", "Long" if last["position"] else "Flat")
c[2].metric("Last P(up)", f"{last['proba']:.3f}")
c[3].metric("Position changes", f"{int(rows['position'].diff().abs().fillna(rows['position'].iloc[0]).sum())}")
c[4].metric("Rows from cached data", f"{int((rows['source'] != 'live').sum())}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=rows.index, y=rows["equity"], name="Paper trading", line=dict(color="#2563eb")))
fig.add_trace(go.Scatter(x=rows.index, y=rows["bh_equity"], name="Buy & Hold", line=dict(color="#9ca3af")))
fig.update_layout(
    height=420,
    margin=dict(t=30, b=10),
    legend=dict(orientation="h", y=1.08),
    yaxis_title="Equity (starting capital = 1)",
)
st.plotly_chart(fig, width="stretch")

st.subheader("Latest signal per pair")
latest = log.sort_values("bar_close").groupby(["symbol", "timeframe"]).tail(1).set_index(["symbol", "timeframe"])
st.dataframe(
    latest[["bar_close", "proba", "position", "close", "equity", "bh_equity", "source"]].style.format(
        {"proba": "{:.3f}", "position": "{:.0f}", "equity": "{:.4f}", "bh_equity": "{:.4f}"}
    ),
    width="stretch",
)

with st.expander(f"Full log for {symbol} {tf}"):
    st.dataframe(rows.iloc[::-1], width="stretch")
st.caption(f"Source: {origin}.")

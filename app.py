import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title="PriceAction AI", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px;}
.metric-card {background: #111827; border: 1px solid #263244; padding: 14px; border-radius: 12px;}
.small-muted {color: #9ca3af; font-size: .88rem;}
div[data-testid="stMetric"] {background: #111827; border: 1px solid #263244; padding: 12px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.title("📈 PriceAction AI")
st.caption("Rule-based price action analysis • Educational tool • Not financial advice")

with st.sidebar:
    st.header("⚙️ Analysis settings")
    pivot_len = st.slider("Swing sensitivity", 2, 8, 3, help="Number of candles on each side used to confirm a swing.")
    risk_pct = st.slider("Risk per trade (%)", 0.1, 2.0, 1.0, 0.1)
    rr_target = st.selectbox("Take-profit risk/reward", [1.5, 2.0, 2.5, 3.0], index=1)
    st.divider()
    st.write("**Supported CSV columns**")
    st.caption("Required: Open, High, Low, Close. Optional: Date/Datetime, Volume. Column names are case-insensitive.")

def make_demo_data(n=260, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().floor("h"), periods=n, freq="h")
    close = [1.0800]
    regime = 0.000035
    for i in range(1, n):
        if i % 65 == 0:
            regime = rng.choice([-0.000045, 0.000045])
        move = regime + rng.normal(0, 0.00022)
        close.append(max(0.5, close[-1] + move))
    close = np.array(close)
    op = np.r_[close[0], close[:-1]]
    spread = rng.uniform(0.00005, 0.00028, n)
    high = np.maximum(op, close) + spread
    low = np.minimum(op, close) - spread
    vol = rng.integers(80, 600, n)
    return pd.DataFrame({"Date": dates, "Open": op, "High": high, "Low": low, "Close": close, "Volume": vol})

def normalize_columns(df):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    aliases = {
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c", "adj close", "adj_close"],
        "volume": ["volume", "vol", "tick volume", "tick_volume"],
        "date": ["date", "datetime", "time", "timestamp"]
    }
    rename = {}
    for target, names in aliases.items():
        for name in names:
            if name in lookup:
                rename[lookup[name]] = target.title()
                break
    out = df.rename(columns=rename).copy()
    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"]).sort_values("Date")
    else:
        out["Date"] = np.arange(len(out))
    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=required).reset_index(drop=True)
    if len(out) < 40:
        raise ValueError("Please provide at least 40 valid candles for meaningful analysis.")
    return out

def find_pivots(df, length):
    highs, lows = df["High"].to_numpy(), df["Low"].to_numpy()
    ph = [np.nan] * len(df)
    pl = [np.nan] * len(df)
    for i in range(length, len(df)-length):
        if highs[i] == np.max(highs[i-length:i+length+1]):
            ph[i] = highs[i]
        if lows[i] == np.min(lows[i-length:i+length+1]):
            pl[i] = lows[i]
    return pd.Series(ph), pd.Series(pl)

def analyse(df, length):
    d = df.copy()
    d["PivotHigh"], d["PivotLow"] = find_pivots(d, length)
    pivot_highs = [(i, x) for i, x in enumerate(d["PivotHigh"]) if pd.notna(x)]
    pivot_lows = [(i, x) for i, x in enumerate(d["PivotLow"]) if pd.notna(x)]
    bias = "Ranging / unclear"
    structure = "Not enough confirmed swings"
    if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
        h1, h2 = pivot_highs[-2][1], pivot_highs[-1][1]
        l1, l2 = pivot_lows[-2][1], pivot_lows[-1][1]
        if h2 > h1 and l2 > l1:
            bias, structure = "Bullish", "Higher highs and higher lows"
        elif h2 < h1 and l2 < l1:
            bias, structure = "Bearish", "Lower highs and lower lows"
        else:
            structure = "Mixed swing structure"
    # Use only swings confirmed before the latest candle.
    prior_highs = [(i, x) for i, x in pivot_highs if i < len(d)-length]
    prior_lows = [(i, x) for i, x in pivot_lows if i < len(d)-length]
    last = d.iloc[-1]
    prev = d.iloc[-2]
    last_high = prior_highs[-1][1] if prior_highs else np.nan
    last_low = prior_lows[-1][1] if prior_lows else np.nan
    sweep_low = bool(pd.notna(last_low) and last["Low"] < last_low and last["Close"] > last_low)
    sweep_high = bool(pd.notna(last_high) and last["High"] > last_high and last["Close"] < last_high)
    bull_bos = bool(pd.notna(last_high) and last["Close"] > last_high and prev["Close"] <= last_high)
    bear_bos = bool(pd.notna(last_low) and last["Close"] < last_low and prev["Close"] >= last_low)
    # Simplified structure-shift proxy; label explicitly as a proxy.
    recent_closes = d["Close"].tail(12)
    momentum_up = recent_closes.iloc[-1] > recent_closes.iloc[0]
    momentum_down = recent_closes.iloc[-1] < recent_closes.iloc[0]
    if sweep_low and momentum_up:
        signal = "Potential bullish reversal"
        rationale = "Price traded below a prior confirmed swing low and closed back above it."
    elif sweep_high and momentum_down:
        signal = "Potential bearish reversal"
        rationale = "Price traded above a prior confirmed swing high and closed back below it."
    elif bull_bos and bias == "Bullish":
        signal = "Potential bullish continuation"
        rationale = "Latest candle closed above a prior confirmed swing high in bullish structure."
    elif bear_bos and bias == "Bearish":
        signal = "Potential bearish continuation"
        rationale = "Latest candle closed below a prior confirmed swing low in bearish structure."
    else:
        signal = "No confirmed setup"
        rationale = "No complete sweep/structure trigger was detected on the latest candle."
    d["MA20"] = d["Close"].rolling(20).mean()
    d["MA50"] = d["Close"].rolling(50).mean()
    return d, bias, structure, signal, rationale, last_high, last_low, sweep_low, sweep_high, bull_bos, bear_bos

def fmt(x):
    return "—" if pd.isna(x) else f"{x:.5f}"

# Data source
st.subheader("1. Load market candles")
source = st.radio("Choose data source", ["Use demo data", "Upload CSV"], horizontal=True)
if source == "Upload CSV":
    uploaded = st.file_uploader("Upload OHLC CSV", type=["csv"])
    if uploaded:
        try:
            raw = normalize_columns(pd.read_csv(uploaded))
        except Exception as e:
            st.error(str(e))
            st.stop()
    else:
        st.info("Upload a CSV containing Open, High, Low and Close columns.")
        st.stop()
else:
    raw = make_demo_data()

d, bias, structure, signal, rationale, last_high, last_low, sweep_low, sweep_high, bull_bos, bear_bos = analyse(raw, pivot_len)

st.caption(f"Loaded {len(d)} candles • Latest candle: {d['Date'].iloc[-1]}")

# Summary
st.subheader("2. Market overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Market bias", bias)
c2.metric("Structure", structure)
c3.metric("Latest close", fmt(d["Close"].iloc[-1]))
c4.metric("Detected setup", signal)
if signal == "No confirmed setup":
    st.info(rationale)
elif "bullish" in signal.lower():
    st.success(rationale)
else:
    st.warning(rationale)

# Chart
st.subheader("3. Price chart and swing levels")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=d["Date"], open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Price"))
fig.add_trace(go.Scatter(x=d["Date"], y=d["MA20"], name="MA20", line=dict(width=1)))
fig.add_trace(go.Scatter(x=d["Date"], y=d["MA50"], name="MA50", line=dict(width=1)))
ph = d.dropna(subset=["PivotHigh"])
pl = d.dropna(subset=["PivotLow"])
fig.add_trace(go.Scatter(x=ph["Date"], y=ph["PivotHigh"], mode="markers", name="Swing highs", marker=dict(symbol="triangle-down", size=8)))
fig.add_trace(go.Scatter(x=pl["Date"], y=pl["PivotLow"], mode="markers", name="Swing lows", marker=dict(symbol="triangle-up", size=8)))
fig.update_layout(height=480, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=25,b=10), template="plotly_dark", legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# Structure diagnostics
st.subheader("4. Price action diagnostics")
diag = pd.DataFrame([
    ["Liquidity sweep below prior low", "Detected" if sweep_low else "Not detected"],
    ["Liquidity sweep above prior high", "Detected" if sweep_high else "Not detected"],
    ["Bullish BOS proxy", "Detected" if bull_bos else "Not detected"],
    ["Bearish BOS proxy", "Detected" if bear_bos else "Not detected"],
    ["Previous confirmed swing high", fmt(last_high)],
    ["Previous confirmed swing low", fmt(last_low)],
], columns=["Condition", "Result"])
st.dataframe(diag, use_container_width=True, hide_index=True)
st.caption("Swing points are confirmed only after the selected number of candles have formed on both sides. This introduces confirmation delay and avoids using unconfirmed pivots.")

# Trade planner
st.subheader("5. Trade planner")
st.caption("Enter a hypothetical trade. The calculator does not place orders or recommend that you trade.")
col_a, col_b = st.columns(2)
with col_a:
    direction = st.selectbox("Trade direction", ["BUY", "SELL"])
    entry = st.number_input("Entry price", min_value=0.000001, value=float(d["Close"].iloc[-1]), format="%.6f")
    stop = st.number_input("Stop-loss price", min_value=0.000001, value=float(d["Close"].iloc[-1] * (0.998 if direction=="BUY" else 1.002)), format="%.6f")
with col_b:
    balance = st.number_input("Account balance ($)", min_value=1.0, value=100.0, step=10.0)
    risk = st.number_input("Risk (%)", min_value=0.1, max_value=2.0, value=float(risk_pct), step=0.1)
    target_rr = st.selectbox("Target R multiple", [1.0, 1.5, 2.0, 2.5, 3.0], index=2)
valid = (stop < entry) if direction == "BUY" else (stop > entry)
if not valid:
    st.error("Invalid stop: BUY stop must be below entry; SELL stop must be above entry.")
else:
    distance = abs(entry-stop)
    target = entry + distance*target_rr if direction=="BUY" else entry-distance*target_rr
    risk_cash = balance*risk/100
    st.markdown("**Trade plan**")
    p1,p2,p3,p4 = st.columns(4)
    p1.metric("Stop distance", f"{distance:.6f}")
    p2.metric("Risk amount", f"${risk_cash:.2f}")
    p3.metric("Target price", f"{target:.6f}")
    p4.metric("Planned R:R", f"1:{target_rr:g}")
    st.caption("Exact lot/contract size requires the instrument's contract specification, quote currency and account currency. This MVP deliberately does not guess those values.")

# Backtesting helper
st.subheader("6. Simple historical review")
st.write("Use the table below to inspect the most recent candles. This first MVP detects conditions on the latest candle; it does not yet claim to run a fully validated automated backtest.")
show_cols = ["Date","Open","High","Low","Close"]
if "Volume" in d.columns:
    show_cols.append("Volume")
st.dataframe(d[show_cols].tail(20), use_container_width=True, hide_index=True)

csv_out = d[show_cols].to_csv(index=False).encode("utf-8")
st.download_button("Download normalized candle data", data=csv_out, file_name="priceaction_candles.csv", mime="text/csv")

with st.expander("How to use this MVP"):
    st.markdown("""
    1. Start with demo data to explore the dashboard.
    2. Export historical OHLC candles from your charting or broker platform as CSV.
    3. Upload the CSV and check that dates and prices are correct.
    4. Read the market bias and diagnostics; do not treat a detected condition as an automatic trade.
    5. Use the risk planner to calculate a hypothetical stop and target.
    6. Record and manually review at least 100 historical setups before considering live use.

    **Limitations:** No live prices, broker connection, automatic order execution, news filter, or validated multi-trade backtest in this version. CHOCH and BOS are simplified proxies and require visual confirmation.
    """)

st.divider()
st.caption("PriceAction AI MVP • Educational use only • Trading involves substantial risk.")

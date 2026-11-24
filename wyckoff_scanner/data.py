"""Data loaders.  All return a DataFrame indexed by UTC timestamp with columns
open, high, low, close[, volume], sorted ascending."""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd

# ---------------------------------------------------------------- CSV ------
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    tcol = next(cols[c] for c in ("time", "timestamp", "date", "datetime") if c in cols)
    ts = df[tcol]
    if pd.api.types.is_numeric_dtype(ts):
        idx = pd.to_datetime(ts, unit="s" if ts.max() < 1e11 else "ms", utc=True)
    else:
        idx = pd.to_datetime(ts, utc=True)
    out = pd.DataFrame({k: df[cols[k]].astype(float) for k in ("open", "high", "low", "close")
                        if k in cols})
    if "volume" in cols:
        out["volume"] = df[cols["volume"]].astype(float)
    out.index = idx
    return out.sort_index()


# ------------------------------------------------------------ Coinbase -----
# Public Advanced Trade endpoints (no API key needed).
CB = "https://api.coinbase.com/api/v3/brokerage/market"
CB_GRAN = {"1m": ("ONE_MINUTE", 60), "5m": ("FIVE_MINUTE", 300), "15m": ("FIFTEEN_MINUTE", 900),
           "30m": ("THIRTY_MINUTE", 1800), "1h": ("ONE_HOUR", 3600), "2h": ("TWO_HOUR", 7200),
           "6h": ("SIX_HOUR", 21600), "1D": ("ONE_DAY", 86400)}
# 4h and 1W aren't native: fetch this and resample
CB_SOURCE_FOR = {"4h": "1h", "1W": "1D"}


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "wyckoff-scanner"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def coinbase_products(product_type: str = "SPOT", expiry: str | None = None) -> list[dict]:
    """product_type: SPOT or FUTURE.  expiry (FUTURE only): PERPETUAL or EXPIRING."""
    url = f"{CB}/products?product_type={product_type}&limit=5000"
    if expiry:
        url += f"&contract_expiry_type={expiry}"
    prods = _get(url).get("products", [])
    return [p for p in prods if not p.get("trading_disabled") and not p.get("is_disabled")]


def coinbase_candles(product_id: str, tf: str, bars: int = 500) -> pd.DataFrame:
    from .timeframes import resample
    if tf in CB_SOURCE_FOR:
        src = CB_SOURCE_FOR[tf]
        mult = {"4h": 4, "1W": 7}[tf]
        return resample(coinbase_candles(product_id, src, bars * mult), tf)
    gran, secs = CB_GRAN[tf]
    end = int(time.time())
    rows = []
    while len(rows) < bars:
        start = end - 350 * secs
        data = _get(f"{CB}/products/{product_id}/candles?start={start}&end={end}&granularity={gran}")
        candles = data.get("candles", [])
        if not candles:
            break
        rows += candles
        end = start
        time.sleep(0.12)   # stay under public rate limit
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows).astype(float)
    df.index = pd.to_datetime(df["start"], unit="s", utc=True)
    df = df[["open", "high", "low", "close", "volume"]]
    df = df[~df.index.duplicated()].sort_index()
    return df.iloc[-bars:]


# stablecoins and fiat-like tokens: no Wyckoff structure worth scanning
STABLES = {"USDT", "USDC", "DAI", "PYUSD", "EURC", "EUROC", "GUSD", "PAX", "USDP", "TUSD", "BUSD",
           "FDUSD", "USDS", "GYEN", "RLUSD", "USD1", "LUSD", "FRAX", "USDE", "SUSD", "EUR", "GBP"}


def quote_volume(p: dict) -> float:
    """Approximate 24h volume in the quote currency (USD for -USD pairs)."""
    try:
        v = float(p.get("approximate_quote_24h_volume") or 0)
        if v:
            return v
        return float(p.get("volume_24h") or 0) * float(p.get("price") or 0)
    except (TypeError, ValueError):
        return 0.0


def coinbase_top_products(n: int | None = None, quote: str = "USD",
                          exclude_stables: bool = True, products: list[dict] | None = None) -> list[dict]:
    """USD spot pairs ranked by 24h dollar volume (top n, or all if n is None)."""
    prods = products if products is not None else coinbase_products("SPOT")
    prods = [p for p in prods if p.get("quote_currency_id", p["product_id"].split("-")[-1]) == quote
             and not p.get("view_only") and not p.get("cancel_only")]
    if exclude_stables:
        prods = [p for p in prods
                 if (p.get("base_currency_id") or p["product_id"].split("-")[0]) not in STABLES]
    prods.sort(key=quote_volume, reverse=True)
    return prods[:n] if n else prods


# which native Coinbase candles each timeframe is built from, and the multiplier
CB_BUILD = {"5m": ("5m", 1), "15m": ("15m", 1), "30m": ("30m", 1), "1h": ("1h", 1),
            "4h": ("1h", 4), "1D": ("1D", 1), "1W": ("1D", 7)}


def coinbase_frames(product_id: str, tfs: list[str], intraday_bars: int = 1000,
                    other_bars: int = 400, fetch=None) -> dict[str, pd.DataFrame]:
    """All requested timeframes for one product, downloading each native
    granularity once (4h is built from 1h, 1W from 1D)."""
    from .timeframes import resample
    fetch = fetch or _coinbase_native
    want = {tf: (intraday_bars if tf in ("5m", "15m", "30m") else other_bars) for tf in tfs}
    need: dict[str, int] = {}
    for tf, bars in want.items():
        src, mult = CB_BUILD[tf]
        need[src] = max(need.get(src, 0), bars * mult)
    native = {src: fetch(product_id, src, bars) for src, bars in need.items()}
    out = {}
    for tf, bars in want.items():
        src, mult = CB_BUILD[tf]
        df = native[src]
        if df.empty:
            continue
        out[tf] = (df if mult == 1 else resample(df, tf)).iloc[-bars:]
    return out


def _coinbase_native(product_id: str, tf: str, bars: int) -> pd.DataFrame:
    return coinbase_candles(product_id, tf, bars)


def coinbase_tv_symbol(product: dict) -> str:
    """Best-effort TradingView symbol.  Spot maps cleanly (COINBASE:BTCUSD).
    Futures/perps naming on TradingView should be verified; falls back to spot."""
    base = product.get("base_currency_id") or product["product_id"].split("-")[0]
    quote_ = product.get("quote_currency_id") or "USD"
    return f"COINBASE:{base}{quote_}"


# Stocks: see alpaca.py

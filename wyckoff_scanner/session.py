"""US equity session handling.

TradingView shows stocks in regular trading hours (09:30-16:00 New York) by
default, with intraday bars anchored at 09:30:
  1h -> 09:30, 10:30, ... 15:30 (last bar is 30 min)
  4h -> 09:30-13:30, 13:30-16:00
Alpaca's intraday bars include pre/post-market trades and are clock-aligned, so
we keep only RTH bars and rebuild higher intraday timeframes from them.
"""
from __future__ import annotations

import pandas as pd

NY = "America/New_York"
OPEN_MIN, SESSION_MIN = 9 * 60 + 30, 390

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}


def rth_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ny = df.index.tz_convert(NY)
    mins = ny.hour * 60 + ny.minute - OPEN_MIN
    return df[(mins >= 0) & (mins < SESSION_MIN)]


def session_resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate intraday bars into `minutes` buckets anchored at the 09:30 open.
    Input bars must be no larger than `minutes` and evenly divide into it."""
    df = rth_only(df)
    if df.empty:
        return df
    ny = df.index.tz_convert(NY)
    mins = ny.hour * 60 + ny.minute - OPEN_MIN
    bucket = (mins // minutes) * minutes
    key = ny.normalize() + pd.to_timedelta(OPEN_MIN + bucket, unit="min")
    agg = dict(AGG, **({"volume": "sum"} if "volume" in df else {}))
    out = df.groupby(key.tz_convert("UTC")).agg(agg)
    out.index.name = None
    return out

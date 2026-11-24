"""Multi-timeframe scanning and highest-valid-timeframe assignment.

A range is "valid" on a timeframe when the full SC -> AR -> EQ sequence is
detected there with two-candle legs each way.  When the same range (same climax
and reaction prices) is found on several timeframes, it is classified on the
highest one, and its deviation limits are judged on that timeframe's closes.
"""
from __future__ import annotations

import pandas as pd

from .ranges import Range, detect_ranges
from .structure import find_swings

# ordered lowest -> highest
TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "1D", "1W"]
PANDAS_RULE = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "2h": "2h",
               "4h": "4h", "6h": "6h", "1D": "1D", "1W": "W-MON", "1M": "MS"}


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = PANDAS_RULE[tf]
    kw = {"label": "left", "closed": "left"} if tf == "1W" else {}
    out = df.resample(rule, **kw).agg({"open": "first", "high": "max", "low": "min",
                                       "close": "last", **({"volume": "sum"} if "volume" in df else {})})
    return out.dropna(subset=["open"])


def build_frames(base: pd.DataFrame, tfs: list[str]) -> dict[str, pd.DataFrame]:
    return {tf: resample(base, tf) for tf in tfs}


def scan_frames(frames: dict[str, pd.DataFrame], require_color: bool = False,
                only_active: bool = False, bos_from_lower_tfs: bool = False,
                require_bos: bool = False, allow_nested: bool = True,
                tap2_return: str = "either", trend_lookback: int = 50,
                opposite_max: int = 1) -> list[Range]:
    """frames: {tf: ohlc dataframe}.  The AR's break of structure is judged on
    the range's own timeframe unless bos_from_lower_tfs=True.  Keys must be ordered lowest -> highest TF
    (or be a subset of TIMEFRAMES, in which case they are sorted)."""
    from . import levels, ranges as _r
    levels.clear_cache()
    _r._MIRRORS.clear(); _r._NS.clear(); _r._OWN.clear()
    order = sorted(frames, key=lambda t: TIMEFRAMES.index(t) if t in TIMEFRAMES else 99)
    ltf_highs, ltf_lows = [], []
    by_key: dict = {}
    done: list[str] = []            # lower timeframes already scanned (for Model 2 levels)
    for tf in order:
        df = frames[tf]
        if len(df) < 10:
            continue
        lh, ll = (ltf_highs, ltf_lows) if bos_from_lower_tfs else ([], [])
        for r in detect_ranges(df, tf, lh, ll, require_color=require_color,
                               require_bos=require_bos, allow_nested=allow_nested,
                               tap2_return=tap2_return, trend_lookback=trend_lookback,
                               opposite_max=opposite_max,
                               ltf_frames=[(t, frames[t]) for t in reversed(done)]):
            by_key[r.key] = r          # higher TF overwrites lower TF
        sw = find_swings(df, require_color)
        ltf_highs += [(df.index[s.idx], s.price) for s in sw if s.kind == "high"]
        ltf_lows += [(df.index[s.idx], s.price) for s in sw if s.kind == "low"]
        done.append(tf)
    out = list(by_key.values())
    if only_active:
        out = [r for r in out if r.status not in ("invalidated", "completed", "superseded")]
    return out

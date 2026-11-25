"""Bojan lows / highs: a wick formation that allows an aggressive entry.

Read from the guide (accumulation; distribution is the mirror, upper wicks):

  * 3 or 4 consecutive candles.
  * Every candle has a real lower wick: at least `min_wick` of its range sits
    below its body.
  * A later candle's wick sweeps below an earlier candle's low (a new low made
    by a wick), and the candle with the lowest low has a long wick: at least
    half of its range below its body.
  * No candle closes below the first candle's low: the sweeps are rejected and
    the bodies hold above.

Where it counts (any reasonable timeframe = the model's own timeframe and the
two below it):

  * tap 3 (spring / UTAD): the formation's lowest low is the tap 3 extreme, or
    within `tol` of it, and lands in or next to the tap 3 extreme candle.
  * LPS / LPSY: after tap 3 is confirmed, with its lowest low above the tap 3
    extreme.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .timeframes import TIMEFRAMES

BIG = np.iinfo(np.int64).max


def find_bojans(o, h, l, c, min_wick: float = 0.3, sizes=(4, 3)) -> list[tuple[int, int, int]]:
    """(first, last, lowest) candle indexes of each Bojan low.  Overlapping
    windows are merged into the longest one."""
    n = len(c)
    rng = h - l
    body_lo = np.minimum(o, c)
    wick = np.where(rng > 0, (body_lo - l) / np.where(rng > 0, rng, 1), 0.0)
    good = wick >= min_wick
    out, used_to = [], -1
    for i in range(n):
        for k in sizes:
            j = i + k - 1
            if j >= n or i <= used_to and j <= used_to:
                continue
            if not good[i:j + 1].all():
                continue
            m = i + int(np.argmin(l[i:j + 1]))
            if m == i or l[m] >= l[i]:
                continue
            if wick[m] < 0.5:
                continue
            if (c[i:j + 1] < l[i]).any():
                continue
            out.append((i, j, m))
            used_to = max(used_to, j)
            break
    return out


def _arrays(df, mirror):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    if mirror:
        o, h, l, c = -o, -l, -h, -c
    return o, h, l, c, pd.DatetimeIndex(df.index).as_unit("ns").asi8


def bojans_for_tap3(frames: dict, tf: str, side: str, ext_start, ext_end, extreme: float,
                    tol: float, lps_after=None, min_wick: float = 0.3, depth: int = 2) -> list[dict]:
    """Bojans at tap 3 and at the LPS/LPSY after it, on `tf` and up to `depth`
    timeframes below it."""
    mirror = side == "distribution"
    x = -extreme if mirror else extreme
    s_ns, e_ns = int(pd.Timestamp(ext_start).value), int(pd.Timestamp(ext_end).value)
    lps_ns = None if lps_after is None else max(int(pd.Timestamp(lps_after).value), e_ns)
    span = max(e_ns - s_ns, 1)
    pos = TIMEFRAMES.index(tf) if tf in TIMEFRAMES else len(TIMEFRAMES)
    tfs = [t for t in TIMEFRAMES[max(0, pos - depth):pos + 1][::-1] if t in frames]
    found = []
    for t in tfs:
        df = frames[t]
        if df is None or len(df) < 3:
            continue
        o, h, l, c, ts = _arrays(df, mirror)
        tend = np.append(ts[1:], BIG)
        at3, lps_hit = None, None
        for i, j, m in find_bojans(o, h, l, c, min_wick):
            low = l[m]
            # tap 3: lowest candle in or next to the tap 3 extreme candle, low at the extreme
            if at3 is None and s_ns - span <= ts[m] < e_ns + span and abs(low - x) <= tol:
                at3 = (i, j, m)
            # LPS / LPSY: after tap 3 was confirmed, holds above the tap 3 extreme
            elif lps_ns is not None and ts[i] >= lps_ns and low > x + tol:
                lps_hit = (i, j, m)            # keep the latest one
        for where, hit in (("tap3", at3), ("lps", lps_hit)):
            if hit is None:
                continue
            i, j, m = hit
            found.append({"where": where, "tf": t,
                          "start": pd.Timestamp(ts[i], tz="UTC"), "end": pd.Timestamp(tend[j] if tend[j] != BIG else ts[j], tz="UTC"),
                          "low_time": pd.Timestamp(ts[m], tz="UTC"),
                          "price": float(-l[m] if mirror else l[m]), "candles": j - i + 1})
    return found

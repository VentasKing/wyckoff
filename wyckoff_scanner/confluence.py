"""Confluence checks shown on a setup's card.

zone_at_tap3: does a Model 1 tap 3 (spring / UTAD) land in a fresh order block?

  * Order block: the last opposite-colour candle before a move that created a fair
    value gap (demand: last bearish candle before a bullish FVG; supply mirrored).
    Zone = that candle's high to low.
  * Fresh: no candle traded into the zone after the FVG completed and before the
    leg down into tap 3 began (the high between tap 2 and tap 3).
  * Hit: the tap 3 extreme is inside the zone, or within `tol` of it.
  * Any timeframe the symbol is scanned on counts; the highest timeframe wins,
    then the zone closest to the extreme.
Written for demand; supply passes mirrored candles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .timeframes import TIMEFRAMES

BIG = np.iinfo(np.int64).max


def _arrays(df: pd.DataFrame, mirror: bool):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    if mirror:
        o, h, l, c = -o, -l, -h, -c
    t = pd.DatetimeIndex(df.index).as_unit("ns").asi8
    return o, h, l, c, t


def _zones(o, h, l, c, t, leg_start_ns: int):
    """Fresh demand order blocks whose FVG completed before the leg into tap 3."""
    tend = np.append(t[1:], BIG)
    cut = int(np.searchsorted(tend, leg_start_ns, side="right"))   # candles ended by the leg start
    if cut < 3:
        return []
    # suffix minimum of lows over [k, cut)
    suf = np.minimum.accumulate(l[:cut][::-1])[::-1]
    out, seen, last_bear = [], set(), None
    for j in range(2, cut):
        if c[j - 2] < o[j - 2]:
            last_bear = j - 2
        if l[j] > h[j - 2] and last_bear is not None and last_bear not in seen:
            seen.add(last_bear)
            top, bot = float(h[last_bear]), float(l[last_bear])
            if j + 1 >= cut or suf[j + 1] > top:
                out.append((bot, top, last_bear, j))
    return out


def zone_at_tap3(frames: dict, side: str, leg_start, extreme: float, tol: float) -> dict | None:
    mirror = side == "distribution"
    x = -extreme if mirror else extreme
    ls = int(pd.Timestamp(leg_start).value)
    order = sorted(frames, key=lambda f: TIMEFRAMES.index(f) if f in TIMEFRAMES else -1, reverse=True)
    for tf in order:
        df = frames[tf]
        if df is None or len(df) < 3:
            continue
        o, h, l, c, t = _arrays(df, mirror)
        best = None
        for bot, top, ob, j in _zones(o, h, l, c, t, ls):
            if not (bot - tol <= x <= top + tol):
                continue
            dist = 0.0 if bot <= x <= top else min(abs(x - bot), abs(x - top))
            key = (dist, -ob)
            if best is None or key < best[0]:
                best = (key, bot, top, ob, j)
        if best is not None:
            _, bot, top, ob, j = best
            lo, hi = (-top, -bot) if mirror else (bot, top)
            return {"tf": tf, "lo": lo, "hi": hi, "distance": best[0][0],
                    "inside": best[0][0] == 0.0,
                    "ob": pd.Timestamp(t[ob], tz="UTC"), "fvg": pd.Timestamp(t[j], tz="UTC")}
    return None

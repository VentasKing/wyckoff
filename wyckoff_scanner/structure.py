"""Market structure using the two-candle rule.

Valid structure = two candles moving in one direction, where the second candle
trades beyond the first:
  * two candles down: candle 2 makes a lower low than candle 1
  * two candles up:   candle 2 makes a higher high than candle 1
An inside candle (e.g. a doji fully covered by the previous candle) does not
trade lower/higher, so that pair is not valid structure.  With
``require_color=True`` both candles must also close bearish/bullish.

Swings alternate high/low like a zigzag.  A swing high is confirmed once two
candles down print after the highest high since the last swing low (and vice
versa).  If price runs past the last confirmed swing before the next one
confirms, that swing is extended to the new extreme.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Swing:
    kind: str          # "high" or "low"
    idx: int           # bar index of the extreme
    price: float
    confirm_idx: int   # bar index on which the swing was confirmed / last extended
    time: object = None


def _arrays(df):
    return tuple(df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))


def pair_down(arrs, i, require_color=False) -> bool:
    """Candles i-1, i form valid two-candle down structure."""
    o, h, l, c = arrs
    if not l[i] < l[i - 1]:
        return False
    return (c[i - 1] < o[i - 1] and c[i] < o[i]) if require_color else True


def pair_up(arrs, i, require_color=False) -> bool:
    o, h, l, c = arrs
    if not h[i] > h[i - 1]:
        return False
    return (c[i - 1] > o[i - 1] and c[i] > o[i]) if require_color else True


def has_two_candle_leg(df_or_arrs, start, end, direction, require_color=False) -> bool:
    """True if two candles strictly after `start` and at/before `end` form a
    valid pair in `direction`."""
    arrs = _arrays(df_or_arrs) if isinstance(df_or_arrs, pd.DataFrame) else df_or_arrs
    f = pair_up if direction == "up" else pair_down
    return any(f(arrs, i, require_color) for i in range(start + 2, end + 1))


def find_swings(df: pd.DataFrame, require_color: bool = False) -> list[Swing]:
    arrs = _arrays(df)
    o, h, l, c = arrs
    n = len(df)
    swings: list[Swing] = []
    if n < 3:
        return swings

    state = None          # None | "seek_high" | "seek_low"
    hi = lo = 0           # candidate extremes
    pending = None        # bar that broke the last swing right after a candidate

    for i in range(1, n):
        if state in (None, "seek_high") and h[i] > h[hi]:
            hi = i
        if state in (None, "seek_low") and l[i] < l[lo]:
            lo = i

        # confirmations take priority
        if state in (None, "seek_high") and i - 1 > hi and pair_down(arrs, i, require_color):
            swings.append(Swing("high", hi, h[hi], i))
            pending = None
            state = "seek_low"
            lo = min(range(hi + 1, i + 1), key=lambda k: l[k])
            continue
        if state in (None, "seek_low") and i - 1 > lo and pair_up(arrs, i, require_color):
            swings.append(Swing("low", lo, l[lo], i))
            pending = None
            state = "seek_high"
            hi = max(range(lo + 1, i + 1), key=lambda k: h[k])
            continue

        # otherwise extend the last swing if price ran through it.  Exception: if
        # this is the FIRST candle after a candidate swing (which already has its
        # two candles in), wait one candle — the next candle may complete the
        # two-candle structure and confirm that swing instead.
        if swings:
            last = swings[-1]
            if pending is not None:
                k = pending
                pending = None
                if last.kind == "low" and min(l[k], l[i]) < last.price:
                    j = k if l[k] <= l[i] else i
                    last.idx, last.price, last.confirm_idx = j, l[j], i
                    hi = max(range(j, i + 1), key=lambda q: h[q]) if j < i else j
                    continue
                if last.kind == "high" and max(h[k], h[i]) > last.price:
                    j = k if h[k] >= h[i] else i
                    last.idx, last.price, last.confirm_idx = j, h[j], i
                    lo = min(range(j, i + 1), key=lambda q: l[q]) if j < i else j
                    continue
            if last.kind == "low" and l[i] < last.price:
                if state == "seek_high" and i == hi + 1 and hi > last.idx:
                    pending = i
                else:
                    last.idx, last.price, last.confirm_idx = i, l[i], i
                    hi = i
            elif last.kind == "high" and h[i] > last.price:
                if state == "seek_low" and i == lo + 1 and lo > last.idx:
                    pending = i
                else:
                    last.idx, last.price, last.confirm_idx = i, h[i], i
                    lo = i

    if swings:
        ts = df.index[[sw.idx for sw in swings]]
        for sw, t in zip(swings, ts):
            sw.time = t
    return swings

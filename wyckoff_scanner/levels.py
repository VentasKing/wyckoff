"""Model 2 levels across timeframes.

Extreme liquidity: the first two-candle structure low (two bearish candles
trading lower, then two candles trading higher) that forms after the tap-2
extreme and before the high between tap 2 and tap 3, taken from the HIGHEST
timeframe (the range's own, then lower ones) that has such structure.

Extreme demand: the first order block (last bearish candle since the tap-2
extreme) whose move created a bullish fair value gap, taken from the highest
timeframe that has one.

A level touched before tap 2 completes (price back in the range) is mitigated and
the next one on that timeframe is used.  Everything is written for accumulation;
distribution passes mirrored candles.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
import pandas as pd

BIG = np.iinfo(np.int64).max
REQUIRE_ORIGIN = False   # optional: only accept an order block at the origin of the move
ORIGIN_OB = 2            #   (tap-2 extreme candle or one of the next 2 candles)
FVG_WITHIN = 3           #   whose FVG prints within 3 candles


@dataclass
class Frame:
    tf: str
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    t: np.ndarray          # bar start times, int64 ns


_CACHE: dict = {}      # (id(df), tf) -> Frame, cleared at the start of each scan


def clear_cache() -> None:
    _CACHE.clear()


def make_frame(tf: str, df: pd.DataFrame) -> Frame:
    key = (id(df), tf)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] is df:
        return hit[1]
    F = Frame(tf, *(df[k].to_numpy(float) for k in ("open", "high", "low", "close")),
              pd.DatetimeIndex(df.index).as_unit("ns").asi8)
    _CACHE[key] = (df, F)
    return F


def structure_break(F: Frame, hit_ns: int, now_end_ns: int) -> tuple[bool, float | None]:
    """After a Model 2 tap: has price broken structure on this frame?
    Takes the lowest low since the hit (the tap-3 extreme), finds the last
    two-candle swing high before it, and checks for a close above it.
    Returns (confirmed, extreme)."""
    a = int(np.searchsorted(F.t, hit_ns, side="left"))
    b = int(np.searchsorted(F.t, now_end_ns, side="left"))
    if b <= a:
        return False, None
    k = a + int(np.argmin(F.l[a:b]))
    idxs, prices = _swing_highs(F)
    pos = bisect.bisect_left(idxs, k)
    if pos == 0:
        return False, float(F.l[k])
    level = prices[pos - 1]
    return bool((F.c[k + 1:b] > level).any()), float(F.l[k])


def _swing_highs(F: Frame) -> list[tuple[int, float]]:
    cache = getattr(F, "_highs", None)
    if cache is None:
        from .structure import find_swings
        df = pd.DataFrame({"open": F.o, "high": F.h, "low": F.l, "close": F.c},
                          index=pd.to_datetime(F.t, utc=True))
        hs = [(s.idx, s.price) for s in find_swings(df) if s.kind == "high"]
        cache = ([i for i, _ in hs], [p for _, p in hs])
        object.__setattr__(F, "_highs", cache)
    return cache


@dataclass
class Level:
    kind: str          # "liquidity" | "demand"
    tf: str
    price: float       # liquidity level, or the zone's top for demand
    lo: float          # zone bottom (demand); same as price for liquidity
    formed_ns: int     # start time of the bar that confirmed the level (FVG's 3rd candle)
    order: int         # bar index of the level's own candle (earlier = first)
    src_ns: int = 0    # start time of the source candle (order block / liquidity low)


class _Scan:
    """Walks one timeframe forward from the tap-2 extreme, only as far as asked,
    collecting unmitigated liquidity lows and order-block demand zones."""

    def __init__(self, F: Frame, start: int, armed_ns: int):
        self.F, self.start = F, start
        self.pos = start + 2
        self.armed_end = int(np.searchsorted(F.t, armed_ns, side="left"))
        self.pend = None
        self.last_bear = None
        self.seen_ob: set = set()
        self.liqs: list[Level] = []
        self.dzs: list[Level] = []

    def _mitigated(self, after: int, hit) -> bool:
        seg = self.F.l[after + 1: self.armed_end]
        return bool(seg.size and hit(seg).any())

    def advance(self, until_ns: int) -> None:
        F, start = self.F, self.start
        o, h, l, c, t = F.o, F.h, F.l, F.c, F.t
        n = len(o)
        while self.pos < n and t[self.pos] < until_ns:
            j = self.pos
            # --- liquidity: two bearish candles trading lower, then two trading higher
            if self.pend is None:
                if j - 1 > start and c[j - 1] < o[j - 1] and c[j] < o[j] and l[j] < l[j - 1]:
                    self.pend = [float(l[j]), j]
            elif l[j] < self.pend[0]:
                self.pend = [float(l[j]), j]
            elif j - 1 > self.pend[1] and h[j] > h[j - 1]:
                p = self.pend[0]
                if not self._mitigated(j, lambda x, p=p: x < p):
                    self.liqs.append(Level("liquidity", F.tf, p, p, int(t[j]), self.pend[1],
                                           int(t[self.pend[1]])))
                self.pend = None
            # --- extreme demand: order block (last bearish candle since the tap-2
            # extreme) whose move created a fair value gap
            if c[j - 2] < o[j - 2]:
                self.last_bear = j - 2
            if l[j] > h[j - 2]:
                ob = self.last_bear if self.last_bear is not None else start
                at_origin = ob - start <= ORIGIN_OB and j - ob <= FVG_WITHIN
                if (at_origin or not REQUIRE_ORIGIN) and ob not in self.seen_ob:
                    self.seen_ob.add(ob)
                    top = float(h[ob])
                    if not self._mitigated(j, lambda x, p=top: x <= p):
                        self.dzs.append(Level("demand", F.tf, top, float(l[ob]), int(t[j]), ob,
                                              int(t[ob])))
            self.pos += 1


def _start_index(F: Frame, t2_start: int, t2_end: int) -> int | None:
    """Index of the tap-2 extreme on this frame (lowest low inside the range-TF
    tap-2 candle).  None if this frame's data doesn't cover tap 2."""
    if len(F.t) == 0 or F.t[0] > t2_start:
        return None
    lo = int(np.searchsorted(F.t, t2_start, side="left"))
    hi = int(np.searchsorted(F.t, t2_end, side="left"))
    if hi <= lo:
        return None
    return lo + int(np.argmin(F.l[lo:hi]))


class Model2Levels:
    def __init__(self, frames: list[Frame], own_start: int, t2_start: int, t2_end: int,
                 armed_ns: int):
        """frames: highest timeframe first; frames[0] is the range's own.
        armed_ns: end of the candle that completed tap 2 (touches before it mitigate)."""
        self.scans: list[_Scan] = []
        for k, F in enumerate(frames):
            start = own_start if k == 0 else _start_index(F, t2_start, t2_end)
            if start is not None:
                self.scans.append(_Scan(F, start, armed_ns))

    def frame(self, tf: str) -> Frame | None:
        return next((sc.F for sc in self.scans if sc.F.tf == tf), None)

    def pick(self, kind: str, high_end_ns: int) -> Level | None:
        """The extreme level formed before the high between tap 2 and tap 3, on
        the highest timeframe that has one."""
        for sc in self.scans:
            sc.advance(high_end_ns)
            items = sc.liqs if kind == "liquidity" else sc.dzs
            if items:
                return min(items, key=lambda x: x.order)
        return None

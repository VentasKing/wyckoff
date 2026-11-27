"""Supply / demand zone scanner (the Zones tab).

Zones (demand; supply is the mirror):

  * Structure demand: a bearish market-structure leg (each candle trading below
    the one before) that runs straight into a bullish candle leaving a fair
    value gap.  The zone is every candle of that leg, wicks included.
  * Order-block demand: when the candle just before the displacement candle
    did not extend a bearish leg (the structure had already turned), that
    candle, whatever its colour, wicks included.
  * Fresh: untouched from the fair value gap until price first trades into it.
    Only that first touch counts.  A close through the far side breaks it.

Bojans: formed (3-4 candles, see bojan.py) or forming (2 candles with the
shape so far, the newest still open) with a wick in the zone, on the zone's own
timeframe or any lower timeframe in the list, after the zone formed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .bojan import find_bojans

# ordered low -> high
ZONE_TFS = ["4h", "5h", "6h", "7h", "8h", "12h", "18h", "1D", "2D", "3D", "4D", "5D", "6D", "1W", "2W", "3W"]
CRYPTO_ZONE_TFS = ZONE_TFS[:]
STOCK_ZONE_TFS = ["8h", "1D", "1W", "2W"]

TV_INTERVAL = {"4h": "240", "5h": "300", "6h": "360", "7h": "420", "8h": "480", "12h": "720", "18h": "1080",
               "1D": "D", "2D": "2D", "3D": "3D", "4D": "4D", "5D": "5D", "6D": "6D",
               "1W": "W", "2W": "2W", "3W": "3W"}

MONDAY = pd.Timestamp("1970-01-05", tz="UTC")


def tf_rank(tf: str) -> int:
    return ZONE_TFS.index(tf) if tf in ZONE_TFS else -1


def _agg(df: pd.DataFrame, key) -> pd.DataFrame:
    g = df.groupby(key, sort=True)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last()})
    if "volume" in df:
        out["volume"] = g["volume"].sum()
    out.index = pd.DatetimeIndex(out.index).tz_convert("UTC") if out.index.tz is not None else \
        pd.DatetimeIndex(out.index).tz_localize("UTC")
    return out.dropna(subset=["open"])


def resample_clock(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """24/7 candles: hour multiples restart at 00:00 UTC each day (TradingView's
    session alignment), day multiples count from the Unix epoch, week multiples
    from a Monday."""
    n, unit = int(tf[:-1]), tf[-1]
    idx = pd.DatetimeIndex(df.index)
    if unit == "h":
        day = idx.floor("D")
        hrs = (idx.hour // n) * n
        key = day + pd.to_timedelta(hrs, unit="h")
    elif unit == "D":
        days = (idx.floor("D") - pd.Timestamp("1970-01-01", tz="UTC")).days
        key = pd.Timestamp("1970-01-01", tz="UTC") + pd.to_timedelta((days // n) * n, unit="D")
    elif unit == "W":
        days = (idx.floor("D") - MONDAY).days
        key = MONDAY + pd.to_timedelta((days // (7 * n)) * 7 * n, unit="D")
    else:
        raise ValueError(tf)
    return _agg(df, key)


# --------------------------------------------------------------- zones -----

@dataclass
class Zone:
    side: str                 # demand | supply
    kind: str                 # structure | ob
    tf: str
    lo: float
    hi: float
    first: int                # first candle of the zone (leg start / OB)
    last: int                 # last candle of the zone
    fvg: int                  # candle that completed the fair value gap
    touch: int | None = None  # first candle that traded into it
    broken: int | None = None # first close through the far side (after the touch)
    left: int | None = None   # first candle entirely outside the zone after the touch
    bojans: list = field(default_factory=list)


def _ohlc(df, mirror):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    return (-o, -l, -h, -c) if mirror else (o, h, l, c)


def find_zones(df: pd.DataFrame, tf: str, side: str = "demand") -> list[Zone]:
    mirror = side == "supply"
    o, h, l, c = _ohlc(df, mirror)
    n = len(c)
    out, seen = [], []
    for j in range(2, n):
        if not l[j] > h[j - 2]:
            continue                       # no bullish fair value gap
        d = j - 1                          # displacement candle
        if not c[d] > o[d]:
            continue
        if j >= 3 and l[j - 1] > h[j - 3]:
            continue                       # a second gap in the same move: the zone is before the first
        p = d - 1
        k = p
        while k - 1 >= 0 and l[k] < l[k - 1]:
            k -= 1
        k += int(np.argmax(h[k:p + 1]))   # the leg starts at its swing high
        if p > k and l[j] > h[k:p + 1].max():   # >= 2 candles of bearish structure, gap above all of it
            first, last, kind = k, p, "structure"
        else:
            # structure had already turned: the candle right before the big one is the order block
            first = last = p
            kind = "ob"
        if any(a <= first and last <= b for a, b in seen):
            continue                       # same candles as a zone already found (continuation gap)
        seen.append((first, last))
        lo, hi = float(l[first:last + 1].min()), float(h[first:last + 1].max())
        z = Zone(side, kind, tf, lo, hi, first, last, j)
        # first touch after the gap completed, then a close through the far side
        for x in range(j + 1, n):
            if z.touch is None:
                if l[x] <= hi:
                    z.touch = x
                    if c[x] < lo:
                        z.broken = x
                        break
            elif c[x] < lo:
                z.broken = x
                break
            elif z.left is None and l[x] > hi:
                z.left = x
        if mirror:
            z.lo, z.hi = -hi, -lo
        out.append(z)
    return out


# -------------------------------------------------------------- bojans -----

def _forming(o, h, l, c, min_wick, isolation=5):
    """The newest two candles already have a Bojan's shape (second wick lower
    and longer), the newest still open."""
    n = len(c)
    if n < 3:
        return None
    a, b = n - 2, n - 1
    rng = h - l
    w = np.minimum(o, c) - l
    share = lambda i: w[i] / rng[i] if rng[i] > 0 else 0.0
    if not (share(a) >= min_wick and share(b) >= min_wick and l[b] < l[a] and w[b] > w[a] and c[b] >= l[a]):
        return None
    near = l[max(0, a - isolation):a]
    if len(near) and (np.abs(near - l[b]) < 0.5 * w[b]).any():
        return None
    return a, b, b


def zone_bojans(z: Zone, frames: dict, tfs: list[str], min_wick: float = 0.4) -> list[dict]:
    """Bojans with a wick in the zone, on the zone's timeframe and every lower
    one in `tfs`, formed after the zone's fair value gap."""
    zdf = frames[z.tf]
    t = pd.DatetimeIndex(zdf.index)
    after = t[z.fvg + 1] if z.fvg + 1 < len(t) else None
    if after is None:
        return []
    mirror = z.side == "supply"
    lo, hi = (-z.hi, -z.lo) if mirror else (z.lo, z.hi)
    out = []
    for tf in [x for x in tfs if tf_rank(x) <= tf_rank(z.tf) and x in frames]:
        df = frames[tf]
        sub = df[df.index >= after]
        if len(sub) < 2:
            continue
        o, h, l, c = _ohlc(sub, mirror)
        ts = pd.DatetimeIndex(sub.index)
        hits = []
        for i, j, m in find_bojans(o, h, l, c, min_wick):
            if (l[i:j + 1] <= hi).any() and not (c[i:j + 1] < lo).any():
                hits.append(("formed", i, j, m))
        f = _forming(o, h, l, c, min_wick)
        if f and (l[f[0]:f[1] + 1] <= hi).any() and not (c[f[0]:f[1] + 1] < lo).any() \
                and not any(i <= f[0] <= j for _, i, j, _ in hits):
            hits.append(("forming", *f))
        for state, i, j, m in hits[-2:]:
            out.append({"tf": tf, "state": state, "start": ts[i], "end": ts[j], "time": ts[m],
                        "price": float(-l[m] if mirror else l[m]), "candles": j - i + 1})
    return out


# ------------------------------------------------------------- scanning -----

def scan_zones(frames: dict, tfs: list[str], keep: int = 6, min_wick: float = 0.4) -> list[Zone]:
    """Fresh zones whose first touch is recent (within `keep` candles of the
    zone's timeframe), or that price has stayed in since the first touch."""
    found = []
    for tf in tfs:
        df = frames.get(tf)
        if df is None or len(df) < 5:
            continue
        n = len(df)
        for side in ("demand", "supply"):
            for z in find_zones(df, tf, side):
                if z.touch is None or z.broken is not None:
                    continue
                # listed while the first touch is recent, or while price hasn't left the zone since
                if z.touch < n - keep and z.left is not None:
                    continue
                z.bojans = zone_bojans(z, frames, tfs, min_wick)
                found.append(z)
    return found


def group_zones(zones: list[Zone], min_overlap: float = 0.5) -> list[list[Zone]]:
    """Group zones on the same side across timeframes.  The highest-timeframe
    zone anchors a group; a lower one joins it when at least `min_overlap` of
    the smaller zone's height overlaps the anchor."""
    groups: list[list[Zone]] = []
    for z in sorted(zones, key=lambda z: (-tf_rank(z.tf), z.lo)):
        for g in groups:
            a = g[0]
            if a.side != z.side:
                continue
            ov = min(a.hi, z.hi) - max(a.lo, z.lo)
            small = min(a.hi - a.lo, z.hi - z.lo)
            if ov > 0 and (small <= 0 or ov >= min_overlap * small):
                g.append(z)
                break
        else:
            groups.append([z])
    return groups

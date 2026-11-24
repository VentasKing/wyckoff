"""Wyckoff range detection and the tap state machine.

Everything is written for ACCUMULATION.  Distribution is handled by mirroring
the candles (price -> -price, high<->low), running the same logic, and mapping
the results back, so every rule is applied symmetrically.

Accumulation lifecycle
----------------------
1. Tap 1 / Selling Climax (SC): a swing low that makes a new low and has a valid
   two-candle leg down into it.
2. Automatic Rally (AR): the highest high after the SC before price retraces to
   the EQ (50% of SC->AR).  The AR must break structure: it must trade above the
   most recent swing high before the SC (on this timeframe or any lower one).
   Two-candle legs are required up into the AR and down out of it.
3. Range confirmed: the first candle that trades into the EQ.
4. Tap 2 (deviation 1): price trades below the range low.  Invalid if any candle
   closes beyond the deviation limit (range low - k * range size, k = 0.25, or
   0.30 on weekly and above).  Completes when a candle closes back inside the
   range.
5. Tap 3:
   * Model 1           - price trades below the tap 2 low; invalid if a candle
                         closes beyond tap2 low - k * range size.
   * Model 2 liquidity - price sweeps the first unmitigated two-candle bearish
                         structure formed after the tap 2 low.
   * Model 2 demand    - price returns into the first unmitigated order block
                         that created a bullish fair value gap after tap 2.
Red flags: opposite end (range high) traded through before tap 3.
Invalidation: close beyond the opposite end's deviation limit before tap 3.
Completed: after a tap 3, a close above the range high.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import bisect

import numpy as np
import pandas as pd

from .structure import Swing, find_swings, has_two_candle_leg
from .levels import BIG, Model2Levels, make_frame, structure_break

HTF_FOR_030 = {"1W", "1M"}


def deviation_k(tf: str) -> float:
    return 0.30 if tf in HTF_FOR_030 else 0.25


@dataclass
class TapEvent:
    kind: str            # tap2_start | tap2_complete | tap3 | invalidated | completed | red_flag
    model: str | None    # model1 | model2_liquidity | model2_demand
    idx: int
    time: object
    price: float
    note: str = ""
    extreme: float | None = None      # tap 3: furthest price reached (the stop level)
    extreme_idx: int | None = None
    confirmed: bool = False           # tap 3: a later candle closed back past its level
    failed: bool = False              # tap 3: extreme broken after confirmation / superseded
    level: float | None = None        # tap 3: close needed to confirm / level that was hit
    level_tf: str | None = None       # Model 2: timeframe of the level (structure checked there)


@dataclass
class Range:
    side: str                    # accumulation | distribution
    tf: str
    climax_idx: int
    climax_price: float          # SC low / BC high
    reaction_idx: int
    reaction_price: float        # AR high / AR low
    confirm_idx: int
    eq: float
    k: float
    dev_limit: float             # tap-side deviation limit
    opposite_limit: float        # opposite-side deviation limit
    status: str = "awaiting_tap2"
    tap2_idx: int | None = None
    tap2_price: float | None = None
    tap2_closed_outside: bool = False
    tap2_return: str = "either"      # how far price must come back to complete tap 2
    liquidity_tf: str | None = None  # timeframe the extreme liquidity came from
    demand_tf: str | None = None     # timeframe the extreme demand/supply came from
    demand_src: dict | None = None   # {"ob": ns, "fvg": ns} candles the zone came from
    liquidity_src: dict | None = None  # {"low": ns, "confirmed": ns}
    model1_limit: float | None = None
    liquidity_level: float | None = None
    demand_zone: tuple | None = None
    red_flags: list = field(default_factory=list)
    events: list = field(default_factory=list)
    invalid_reason: str | None = None
    times: dict = field(default_factory=dict)
    end_idx: int | None = None   # bar where the range was invalidated/completed
    n_bars: int = 0              # bars in the frame it was detected on
    _liq_pending: list | None = field(default=None, repr=False)

    @property
    def key(self):
        return (self.side, round(self.climax_price, 10), round(self.reaction_price, 10))

    def tap3_events(self):
        return [e for e in self.events if e.kind == "tap3"]

    def to_dict(self):
        d = asdict(self)
        d["events"] = [asdict(e) for e in self.events]
        return d


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def detect_ranges(df: pd.DataFrame, tf: str, ltf_swing_highs=None, ltf_swing_lows=None,
                  side: str = "both", require_color: bool = False, require_bos: bool = False,
                  allow_nested: bool = True, tap2_return: str = "either",
                  ltf_frames=None, trend_lookback: int = 50) -> list[Range]:
    """Find every confirmed range on `df` and run its tap state machine to the
    last bar.  `ltf_swing_*` are lists of (time, price) from lower timeframes,
    used for the break-of-structure check.

    require_bos=False:  an AR that doesn't break prior structure is kept, with the
                        red flag "ar_no_break_of_structure".
    allow_nested=True:  a range whose climax forms inside another range's life is
                        kept, with the red flag "inside_larger_range"."""
    out: list[Range] = []
    ltf_frames = ltf_frames or []        # [(tf, df)], highest timeframe first
    if side in ("both", "accumulation"):
        ltf = [make_frame(t, d) for t, d in ltf_frames]
        out += _detect_accumulation(df, tf, ltf_swing_highs or [], require_color, "accumulation",
                                    require_bos, tap2_return, ltf, trend_lookback)
    if side in ("both", "distribution"):
        m = mirror(df)
        lows_m = [(t, -p) for t, p in (ltf_swing_lows or [])]   # LTF lows become highs
        ltf_m = [make_frame(t, _mirror_cached(d)) for t, d in ltf_frames]
        rs = _detect_accumulation(m, tf, lows_m, require_color, "distribution", require_bos,
                                  tap2_return, ltf_m, trend_lookback)
        out += [_unmirror(r) for r in rs]
    return _handle_nested(out, allow_nested)


def _handle_nested(ranges: list[Range], allow: bool) -> list[Range]:
    """A range whose climax falls inside an earlier range's life (either side) is
    nested.  Dropped when allow=False, otherwise flagged."""
    kept: list[Range] = []
    for r in sorted(ranges, key=lambda x: (x.climax_idx, x.confirm_idx)):
        outer = [k for k in kept if k.status != "superseded"
                 and k.climax_idx < r.climax_idx <= k.end_idx]
        if outer:
            if not allow:
                continue
            r.red_flags.append("inside_larger_range")
        kept.append(r)
    return kept


_MIRRORS: dict = {}
_NS: dict = {}
_OWN: dict = {}


def _ns(times):
    hit = _NS.get(id(times))
    if hit is not None and hit[0] is times:
        return hit[1]
    v = pd.DatetimeIndex(times).as_unit("ns").asi8
    _NS[id(times)] = (times, v)
    return v


def _mirror_cached(df: pd.DataFrame) -> pd.DataFrame:
    hit = _MIRRORS.get(id(df))
    if hit is not None and hit[0] is df:
        return hit[1]
    m = mirror(df)
    _MIRRORS[id(df)] = (df, m)
    return m


def mirror(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"open": -df["open"], "high": -df["low"],
                         "low": -df["high"], "close": -df["close"]}, index=df.index)


def _unmirror(r: Range) -> Range:
    neg = lambda x: None if x is None else -x
    r.climax_price, r.reaction_price = -r.climax_price, -r.reaction_price
    r.eq, r.dev_limit, r.opposite_limit = -r.eq, -r.dev_limit, -r.opposite_limit
    r.tap2_price, r.model1_limit = neg(r.tap2_price), neg(r.model1_limit)
    r.liquidity_level = neg(r.liquidity_level)
    if r.demand_zone:
        r.demand_zone = tuple(sorted((-r.demand_zone[0], -r.demand_zone[1])))
    for e in r.events:
        e.price = -e.price
        e.extreme = neg(e.extreme)
        e.level = neg(e.level)
        e.model = {"model2_demand": "model2_supply"}.get(e.model, e.model)
    r.red_flags = [f.replace("range_high", "range_low") for f in r.red_flags]
    return r


# --------------------------------------------------------------------------
# accumulation detection
# --------------------------------------------------------------------------
MIN_TREND_HISTORY = 10


def _detect_accumulation(df, tf, ltf_highs, require_color, side_label, require_bos=False,
                         tap2_return="either", ltf=None, trend_lookback=50):
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    arrs = (o, h, l, c)
    n = len(df)
    times = df.index
    swings = find_swings(df, require_color)
    struct_highs = [(times[s.idx], s.price) for s in swings if s.kind == "high"] + list(ltf_highs)
    struct_highs.sort(key=lambda x: x[0])
    sh_times = list(pd.DatetimeIndex([t for t, _ in struct_highs], tz="UTC").as_unit("ns").asi8) \
        if struct_highs else []
    sh_prices = [p for _, p in struct_highs]
    tvals = pd.DatetimeIndex(times).as_unit("ns").asi8

    ranges: list[Range] = []
    prev_low = prev_high = None
    for s in swings:
        if s.kind == "high":
            prev_high = s
            continue
        sc = s
        is_new_low = prev_low is None or sc.price < prev_low.price
        prev_low = s
        if not is_new_low:
            continue
        # tap 1 ends the trend on this timeframe, judged on CLOSES: the selling
        # climax's close (the lower close of the climax candle and the one before it,
        # since a climax often wicks and closes off its low) must be below every close
        # of the previous `trend_lookback` candles.  In an uptrend a pullback never
        # is, so ranges there are drawn from the top down, and vice versa.
        if trend_lookback:
            k = sc.idx
            end = max(0, k - 1)
            climax_close = c[end:k + 1].min()
            prior = c[max(0, end - trend_lookback): end]
            if len(prior) < min(trend_lookback, MIN_TREND_HISTORY) or prior.min() <= climax_close:
                continue
        # valid two-candle leg down into the SC
        leg_start = prev_high.idx if prev_high else 0
        if not has_two_candle_leg(arrs, leg_start, sc.idx, "down", require_color):
            continue
        # structure the AR must break: latest swing high before the SC
        pos = bisect.bisect_left(sh_times, int(tvals[sc.idx]))
        bos_level = sh_prices[pos - 1] if pos else None
        if require_bos and bos_level is None:
            continue

        run_hi = None
        for i in range(sc.idx + 1, n):
            if l[i] < sc.price:
                break
            if run_hi is None or h[i] > h[run_hi]:
                run_hi = i
                continue
            eq = (sc.price + h[run_hi]) / 2
            if l[i] <= eq:
                broke = bos_level is not None and h[run_hi] > bos_level
                ok = ((broke or not require_bos)
                      and has_two_candle_leg(arrs, sc.idx, run_hi, "up", require_color)
                      and has_two_candle_leg(arrs, run_hi, i, "down", require_color))
                if ok:
                    k = deviation_k(tf)
                    size = h[run_hi] - sc.price
                    r = Range(side=side_label, tf=tf, tap2_return=tap2_return,
                              climax_idx=sc.idx, climax_price=sc.price,
                              reaction_idx=run_hi, reaction_price=h[run_hi], confirm_idx=i,
                              eq=eq, k=k, dev_limit=sc.price - k * size,
                              opposite_limit=h[run_hi] + k * size)
                    r.times = {"climax": times[sc.idx], "reaction": times[run_hi],
                               "confirmed": times[i]}
                    if not broke:
                        r.red_flags.append("ar_no_break_of_structure")
                    _track(r, arrs, times, require_color, ltf=ltf)
                    ranges.append(r)
                break
    _apply_supersede(ranges, arrs, times, require_color, ltf)
    return ranges


# --------------------------------------------------------------------------
# tap state machine
# --------------------------------------------------------------------------
def _track(r: Range, arrs, times, require_color, stop_at: int | None = None, ltf=None):
    n = len(arrs[0])
    r.n_bars = n
    if stop_at is not None:                       # replay only up to stop_at
        arrs = tuple(a[: stop_at + 1] for a in arrs)
    try:
        _track_loop(r, arrs, times, require_color, ltf or [])
    finally:
        if r.status == "invalidated":
            # nothing from an invalidated range is a valid tap 3
            for e in r.events:
                if e.kind == "tap3" and not e.failed:
                    e.failed = True
                    e.note += f" | range invalidated: {r.invalid_reason}"
        done = r.status in ("invalidated", "completed")
        r.end_idx = r.events[-1].idx if done and r.events else len(arrs[0]) - 1
        if stop_at is not None and not done:
            r.status = "superseded"
            r.end_idx = stop_at
            r.events.append(TapEvent("superseded", None, stop_at, times[stop_at],
                                     float(r.climax_price), "a newer range formed before tap 2"))


TRACK_FLAGS = ("range_high_not_fresh",)


def _retrack(r: Range, arrs, times, require_color, stop_at, ltf=None):
    """Reset everything the tap state machine produced and replay up to stop_at."""
    r.status, r.invalid_reason = "awaiting_tap2", None
    r.tap2_idx = r.tap2_price = r.model1_limit = r.liquidity_level = r.demand_zone = None
    r.tap2_closed_outside, r._liq_pending = False, None
    r.liquidity_tf = r.demand_tf = None
    r.liquidity_src = r.demand_src = None
    r.events = []
    r.red_flags = [f for f in r.red_flags if f not in TRACK_FLAGS]
    _track(r, arrs, times, require_color, stop_at=stop_at, ltf=ltf)


def _apply_supersede(ranges: list[Range], arrs, times, require_color, ltf=None) -> None:
    """A newer range on the same side replaces an older one that hasn't printed
    its tap 2 yet: the older range is retired ("superseded") on the bar the newer
    one confirms.  Older ranges already past tap 2 are left alone, so a range
    that forms from a tap-2 extreme (range within a range) still works."""
    for r in sorted(ranges, key=lambda x: x.confirm_idx):
        if r.status == "superseded":
            continue
        for o in ranges:
            if o is r or o.status == "superseded":
                continue
            if not (o.climax_idx < r.climax_idx and o.confirm_idx < r.confirm_idx):
                continue
            if o.end_idx < r.confirm_idx:           # already finished by then
                continue
            t2 = next((e.idx for e in o.events if e.kind == "tap2_start"), None)
            if t2 is not None and t2 <= r.confirm_idx:
                continue
            _retrack(o, arrs, times, require_color, stop_at=r.confirm_idx, ltf=ltf)


def _track_loop(r: Range, arrs, times, require_color, ltf):
    o, h, l, c = arrs
    n = len(o)
    tns = _ns(times)
    m2: Model2Levels | None = None
    sc, ar = r.climax_price, r.reaction_price
    size = ar - sc

    def ev(kind, model, i, price, note=""):
        e = TapEvent(kind, model, i, times[i], float(price), note)
        r.events.append(e)
        return e

    taps: list[TapEvent] = []   # tap 3 events in play
    liq_done = dz_done = False
    tapped = set()
    end_of = lambda k: int(tns[k + 1]) if k + 1 < len(tns) else BIG

    for i in range(r.confirm_idx + 1, n):
        if r.status in ("invalidated", "completed"):
            break
        pre_tap3 = not any(not e.failed for e in taps)

        # ---- opposite end ---------------------------------------------------
        if pre_tap3:
            if c[i] > r.opposite_limit:
                r.status, r.invalid_reason = "invalidated", "closed beyond opposite deviation limit before tap 3"
                ev("invalidated", None, i, c[i], r.invalid_reason)
                break
            if h[i] > ar and "range_high_not_fresh" not in r.red_flags:
                r.red_flags.append("range_high_not_fresh")
                ev("red_flag", None, i, h[i], "opposite end traded through before tap 3")
        elif c[i] > ar:
            r.status = "completed"
            ev("completed", None, i, c[i], "closed beyond opposite end after tap 3")
            break

        # ---- tap 2 ----------------------------------------------------------
        if r.status == "awaiting_tap2" and l[i] < sc:
            r.status, r.tap2_idx, r.tap2_price = "tap2_active", i, l[i]
            ev("tap2_start", None, i, l[i])
        if r.status == "tap2_active":
            if l[i] < r.tap2_price:
                r.tap2_idx, r.tap2_price = i, l[i]
            if c[i] < r.dev_limit:
                r.status, r.invalid_reason = "invalidated", "tap 2 closed beyond deviation limit"
                ev("invalidated", None, i, c[i], r.invalid_reason)
                break
            if c[i] < sc:
                r.tap2_closed_outside = True
            # displacement: the close back inside must come from a candle AFTER
            # the one that made the tap 2 extreme
            if _tap2_returned(r, c, i, sc, size):
                r.status = "awaiting_tap3"
                r.model1_limit = r.tap2_price - r.k * size
                ev("tap2_complete", None, i, c[i],
                   "wick only" if not r.tap2_closed_outside else "closed outside, within limit")
                from .levels import Frame
                own = _OWN.get(id(times))
                if own is None or own[0] is not times or len(own[1].o) != n:
                    own = (times, Frame(r.tf, o, h, l, c, tns))
                    _OWN[id(times)] = own
                frames = [own[1]] + list(ltf)
                m2 = Model2Levels(frames, r.tap2_idx, int(tns[r.tap2_idx]), end_of(r.tap2_idx),
                                  end_of(i))
            continue

        # ---- tap 3 ----------------------------------------------------------
        if r.status in ("awaiting_tap3", "tap3"):
            new_low_m1 = l[i] < r.tap2_price and "model1" not in tapped

            # update the taps already in play (before looking for new ones)
            for e in taps:
                if e.failed or e.idx == i:
                    continue
                if e.model != "model1":
                    # Model 2: potential until structure breaks on the level's timeframe
                    if not e.confirmed and m2 is not None:
                        F = m2.frame(e.level_tf)
                        if F is not None:
                            ok, ext = structure_break(F, int(tns[e.idx]), end_of(i))
                            if ext is not None:
                                e.extreme = ext
                            if ok:
                                e.confirmed = True
                    continue
                if not e.confirmed:
                    if l[i] < e.extreme:
                        e.extreme, e.extreme_idx = float(l[i]), i
                    if c[i] > e.level and i > e.extreme_idx:
                        e.confirmed = True
                elif l[i] < e.extreme:
                    e.failed = True
                    e.note += " | stop hit: traded through the confirmed tap 3 extreme"
                    if e.model == "model1":
                        r.status = "invalidated"
                        r.invalid_reason = "traded through the confirmed tap 3 extreme"
                        ev("invalidated", None, i, l[i], r.invalid_reason)
            if r.status == "invalidated":
                break

            # model 1: price deviates the tap 2 extreme
            if l[i] < r.tap2_price:
                m1 = next((e for e in taps if e.model == "model1"), None)
                if c[i] < r.model1_limit and (m1 is None or not m1.confirmed):
                    r.status, r.invalid_reason = "invalidated", "tap 3 closed beyond model 1 deviation limit"
                    ev("invalidated", None, i, c[i], r.invalid_reason)
                    break
                if new_low_m1:
                    tapped.add("model1")
                    e = ev("tap3", "model1", i, l[i], "deviated the tap 2 extreme")
                    e.extreme, e.extreme_idx, e.level = float(l[i]), i, sc
                    taps.append(e)
                    # model 2 taps that ran into a model 1 are superseded by it
                    for e2 in taps:
                        if e2.model != "model1" and not e2.failed:
                            e2.failed = True
                            e2.note += " | superseded: price went on to deviate tap 2 (model 1)"
                liq_done = dz_done = True    # model 2 levels are retired once a model 1 starts
            elif "model1" not in tapped and m2 is not None and i - 1 > r.tap2_idx:
                # the high between tap 2 and tap 3 so far (candles before this one)
                hi = r.tap2_idx + 1 + int(np.argmax(h[r.tap2_idx + 1: i]))
                high_end = end_of(hi)
                def m2_hit(model, level, tf, note):
                    """First Model 2 level hit = the (potential) tap 3; later level
                    hits are noted on that same tap, not new taps."""
                    tapped.add(model)
                    live = next((x for x in taps if x.model != "model1" and not x.failed), None)
                    if live is not None:
                        live.note += f" | also {note}"
                        return
                    e = ev("tap3", model, i, l[i], note)
                    e.level, e.level_tf = level, tf
                    e.extreme, e.extreme_idx = float(l[i]), i
                    taps.append(e)

                if not liq_done:
                    L = m2.pick("liquidity", high_end)
                    if L is not None:
                        r.liquidity_level, r.liquidity_tf = L.price, L.tf
                        r.liquidity_src = {"low": L.src_ns, "confirmed": L.formed_ns}
                        if l[i] < L.price:
                            liq_done = True
                            m2_hit("model2_liquidity", L.price, L.tf,
                                   f"swept extreme liquidity {L.price:.6g} ({L.tf})")
                if not dz_done:
                    D = m2.pick("demand", high_end)
                    if D is not None:
                        r.demand_zone, r.demand_tf = (D.lo, D.price), D.tf
                        r.demand_src = {"ob": D.src_ns, "fvg": D.formed_ns}
                        if l[i] <= D.price:
                            dz_done = True
                            m2_hit("model2_demand", D.price, D.tf,
                                   f"tapped extreme demand {D.lo:.6g}-{D.price:.6g} ({D.tf})")

            live_taps = [e for e in taps if not e.failed]
            r.status = "tap3" if live_taps else "awaiting_tap3"


TAP2_RETURN_MODES = {
    "either": "quarter OR two_closes, whichever happens first",
    "quarter": "a later candle closes above the midpoint of range low and EQ (25% into the range)",
    "two_closes": "two consecutive later candles close back inside the range",
    "close": "a later candle closes back inside the range",
}


def _tap2_returned(r: Range, c, i: int, sc: float, size: float) -> bool:
    """Has price displaced back into the range enough to complete tap 2?
    Only candles after the one that made the tap-2 extreme count."""
    if i <= r.tap2_idx:
        return False
    quarter = c[i] > sc + 0.25 * size
    two = i - 1 > r.tap2_idx and c[i] > sc and c[i - 1] > sc
    if r.tap2_return == "either":
        return quarter or two
    if r.tap2_return == "quarter":
        return quarter
    if r.tap2_return == "two_closes":
        return two
    return c[i] > sc

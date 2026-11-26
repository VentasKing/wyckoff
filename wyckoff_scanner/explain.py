"""Explain what the scanner sees on one symbol: every candidate tap 1 per
timeframe with the rule that kept or rejected it, and what each range did."""
from __future__ import annotations

import pandas as pd

from . import ranges as R
from .timeframes import TIMEFRAMES, scan_frames


def _t(ts, tf) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d" if tf in ("1D", "1W", "1M") else "%Y-%m-%d %H:%M")


def explain(symbol: str, frames: dict, rules: dict, last: int = 12, out=print) -> list:
    R.TRACE = []
    try:
        ranges = scan_frames(frames, require_bos=rules.get("require_ar_break_of_structure", False),
                             allow_nested=rules.get("allow_nested_ranges", True),
                             tap2_return=rules.get("tap2_return", "either"),
                             trend_lookback=int(rules.get("trend_lookback", 50)),
                             opposite_max=int(rules.get("max_opposite_deviations", 1)),
                             tap3_spacing=float(rules.get("min_tap3_spacing", 0.25)))
        trace = R.TRACE
    finally:
        R.TRACE = None

    out(f"=== {symbol} ===  rules: {rules}")
    order = sorted(frames, key=lambda t: TIMEFRAMES.index(t) if t in TIMEFRAMES else 99)
    for tf in order:
        df = frames[tf]
        out(f"\n--- {tf}: {len(df)} candles, {_t(df.index[0], tf)} → {_t(df.index[-1], tf)} ---")
        for side, name in (("accumulation", "selling climax (tap 1 low)"), ("distribution", "buying climax (tap 1 high)")):
            cands = [x for x in trace if x["tf"] == tf and x["side"] == side]
            out(f"  {side}: {len(cands)} candidate {name}s, latest {min(last, len(cands))}:")
            for x in cands[-last:]:
                out(f"    {'✓' if x['ok'] else '✗'} {_t(x['time'], tf)}  {x['price']:<10.6g} {x['reason']}")
        rs = [r for r in ranges if r.tf == tf]
        out(f"  ranges kept on {tf}: {len(rs)} (a range found on several timeframes is kept on the highest)")
        for r in sorted(rs, key=lambda r: r.climax_idx)[-last:]:
            taps = [f"{e.model}:{'failed' if e.failed else 'confirmed' if e.confirmed else 'open'}@{_t(e.time, tf)}"
                    for e in r.tap3_events()]
            tap2 = f"tap2 {r.tap2_price:.6g} {_t(df.index[r.tap2_idx], tf)}" if r.tap2_idx is not None else "no tap 2"
            out(f"    • {r.side:12} tap1 {r.climax_price:<9.6g} {_t(r.times['climax'], tf)}  AR {r.reaction_price:<9.6g} "
                f"{_t(r.times['reaction'], tf)}  {tap2}  status={r.status}"
                + (f" ({r.invalid_reason})" if r.invalid_reason else "")
                + (f"  flags={r.red_flags}" if r.red_flags else "")
                + (f"  tap3={taps}" if taps else ""))
            _bojans(r, df, frames, rules, out)
    return ranges


def _bojans(r, df, frames, rules, out):
    from .bojan import bojans_for_tap3
    idx = df.index
    for e in r.tap3_events():
        ei = min(e.extreme_idx if e.extreme_idx is not None else e.idx, len(idx) - 1)
        end = idx[ei + 1] if ei + 1 < len(idx) else idx[ei] + (idx[-1] - idx[-2])
        lps = idx[min(e.confirmed_idx, len(idx) - 1)] if e.confirmed and not e.failed and e.confirmed_idx is not None else None
        tol = rules.get("tap3_zone_tolerance", 0.05) * abs(r.reaction_price - r.climax_price)
        for b in bojans_for_tap3(frames, r.tf, r.side, idx[ei], end, e.extreme if e.extreme is not None else e.price,
                                 tol, lps_after=lps, min_wick=rules.get("bojan_min_wick", 0.4),
                                 depth=int(rules.get("bojan_timeframes_below", 2))):
            out(f"        ◉ Bojan at {b['where']} on {b['tf']} ({e.model}): {b['candles']} candles")
            f = frames[b["tf"]]
            for t, row in f[(f.index >= b["start"]) & (f.index < b["end"])].iterrows():
                out(f"            {_t(t, b['tf'])}  O {row.open:<10.6g} H {row.high:<10.6g} L {row.low:<10.6g} C {row.close:<10.6g}")

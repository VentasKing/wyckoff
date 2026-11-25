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
    return ranges

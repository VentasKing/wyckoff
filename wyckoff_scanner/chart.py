"""Draw a detected range and its taps on a candlestick chart (matplotlib)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .ranges import Range

UP, DOWN = "#26a69a", "#ef5350"
COL = {"range": "#1f4e79", "eq": "#7f7f7f", "limit": "#c0392b", "tap": "#d35400",
       "liq": "#8e44ad", "zone": "#2e86c1"}
MODEL_LABEL = {"model1": "Model 1", "model2_liquidity": "M2 liquidity",
               "model2_demand": "M2 demand", "model2_supply": "M2 supply"}


def plot_range(df: pd.DataFrame, r: Range, title: str = "", pad_before: int = 25,
               pad_after: int = 15, ax=None, tz: str | None = None):
    """df: the OHLC frame the range was detected on (same timeframe)."""
    start = max(0, r.climax_idx - pad_before)
    end = min(len(df) - 1, (r.end_idx if r.end_idx is not None else len(df) - 1) + pad_after)
    d = df.iloc[start:end + 1]
    x0 = start
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 6.5))
    else:
        fig = ax.figure

    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    w = 0.6
    for i in range(len(d)):
        col = UP if c[i] >= o[i] else DOWN
        ax.vlines(i, l[i], h[i], color=col, linewidth=1)
        ax.add_patch(Rectangle((i - w / 2, min(o[i], c[i])), w, max(abs(c[i] - o[i]), 1e-12),
                               facecolor=col, edgecolor=col))

    acc = r.side == "accumulation"
    rel = lambda idx: idx - x0
    left, right = rel(r.climax_idx), len(d) - 1
    ystep = (d["high"].max() - d["low"].min()) * 0.035

    def hline(y, color, style, label, x_from=left):
        ax.hlines(y, x_from, right, colors=color, linestyles=style, linewidth=1.2)
        ax.text(right + 0.5, y, f" {label} {y:.4g}", color=color, va="center", fontsize=8)

    hline(r.climax_price, COL["range"], "-", "range low" if acc else "range high")
    hline(r.reaction_price, COL["range"], "-", "range high" if acc else "range low")
    hline(r.eq, COL["eq"], ":", "EQ", rel(r.reaction_idx))
    if getattr(r, "tap2_return", "") in ("quarter", "either"):
        q = r.climax_price + 0.25 * (r.reaction_price - r.climax_price)
        hline(q, COL["eq"], (0, (1, 4)), "tap 2 return (25%)", rel(r.confirm_idx))
    hline(r.dev_limit, COL["limit"], "--", "dev limit", rel(r.confirm_idx))
    if r.model1_limit is not None and r.tap2_idx is not None:
        hline(r.model1_limit, COL["limit"], "-.", "M1 limit", rel(r.tap2_idx))
    if r.liquidity_level is not None and r.tap2_idx is not None:
        hline(r.liquidity_level, COL["liq"], "--",
              f"ext. liquidity{f' ({r.liquidity_tf})' if r.liquidity_tf else ''}", rel(r.tap2_idx))
    if r.demand_zone is not None and r.tap2_idx is not None:
        lo, hi = r.demand_zone
        ax.add_patch(Rectangle((rel(r.tap2_idx), lo), right - rel(r.tap2_idx), hi - lo,
                               facecolor=COL["zone"], alpha=0.15, edgecolor=COL["zone"]))
        zl = ("ext. demand" if acc else "ext. supply") + (f" ({r.demand_tf})" if r.demand_tf else "")
        ax.text(rel(r.tap2_idx) + 0.5, hi if acc else lo, zl,
                color=COL["zone"], va="bottom" if acc else "top", fontsize=8)

    def mark(idx, price, text, below, color=COL["tap"]):
        x = rel(idx)
        y = price - ystep if below else price + ystep
        ax.annotate(text, (x, price), (x, y), ha="center", va="top" if below else "bottom",
                    fontsize=9, fontweight="bold", color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=1))

    mark(r.climax_idx, r.climax_price, "1\nSC" if acc else "BC\n1", acc)
    mark(r.reaction_idx, r.reaction_price, "AR", not acc, COL["range"])
    ax.axvline(rel(r.confirm_idx), color=COL["eq"], linewidth=0.8, alpha=0.6)
    ax.text(rel(r.confirm_idx), ax.get_ylim()[1], "confirmed", fontsize=7, color=COL["eq"],
            ha="center", va="bottom")
    if r.tap2_idx is not None:
        mark(r.tap2_idx, r.tap2_price, "2" if acc else "2", acc)
    by_bar: dict[int, list] = {}
    for e in r.tap3_events():
        x = e.extreme_idx if e.extreme_idx is not None else e.idx
        by_bar.setdefault(x, []).append(e)
    for idx, evs in by_bar.items():
        def tag(e):
            state = " ✗" if e.failed else (" ✓" if e.confirmed else
                                            (" (potential)" if e.model != "model1" else " (forming)"))
            return MODEL_LABEL.get(e.model, e.model) + state
        names = "\n".join(tag(e) for e in evs)
        ext = [e.extreme if e.extreme is not None else e.price for e in evs]
        price = min(ext) if acc else max(ext)
        color = "#999999" if all(e.failed for e in evs) else COL["tap"]
        mark(idx, price, f"3\n{names}" if acc else f"{names}\n3", acc, color)
    for e in r.tap3_events():
        if e.confirmed and not e.failed and e.extreme is not None:
            hline(e.extreme, COL["limit"], ":", "tap 3 stop", rel(e.extreme_idx))
    for e in r.events:
        if e.kind in ("invalidated", "completed"):
            ax.axvline(rel(e.idx), color=COL["limit"] if e.kind == "invalidated" else UP,
                       linewidth=1, linestyle="--")

    # x axis as dates
    n = len(d)
    ticks = np.linspace(0, n - 1, min(10, n)).astype(int)
    fmt = "%Y-%m-%d" if r.tf in ("1D", "1W", "1M") else "%m-%d %H:%M"  # 5m-4h show time
    ax.set_xticks(ticks)
    idx = d.index if tz is None else d.index.tz_convert(tz)
    ax.set_xticklabels([idx[t].strftime(fmt) for t in ticks], rotation=0, fontsize=8)
    if r.tf not in ("1D", "1W", "1M"):
        ax.set_xlabel(f"time ({tz or 'UTC'}" + (f", {idx[-1].strftime('%Z')})" if tz else ")"), fontsize=8)
    ax.set_xlim(-1, right + 8)
    ax.margins(y=0.08)
    flags = f"   ⚠ {', '.join(r.red_flags)}" if r.red_flags else ""
    why = f" ({r.invalid_reason})" if r.status == "invalidated" and r.invalid_reason else ""
    ax.set_title(f"{title}  {r.tf}  {r.side}  — status: {r.status}{why}{flags}", fontsize=11, loc="left")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig

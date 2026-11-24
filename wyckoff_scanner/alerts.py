"""Alerts and TradingView links."""
from __future__ import annotations

import json
import os
from urllib.parse import quote

from .ranges import Range

TV_INTERVAL = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "2h": "120",
               "4h": "240", "6h": "360", "1D": "D", "1W": "W", "1M": "M"}


def tradingview_url(tv_symbol: str, tf: str) -> str:
    """tv_symbol like 'NASDAQ:AAPL' or 'COINBASE:BTCUSD'."""
    return (f"https://www.tradingview.com/chart/?symbol={quote(tv_symbol, safe='')}"
            f"&interval={TV_INTERVAL[tf]}")


def tap3_alerts(symbol: str, tv_symbol: str, ranges: list[Range], since=None,
                max_bars_ago: int | None = None, live_only: bool = False) -> list[dict]:
    """One alert per tap 3 event.
    max_bars_ago: only taps that printed within the last N bars of their timeframe.
    live_only:    skip ranges that already completed (broke out) or were invalidated."""
    alerts = []
    for r in ranges:
        if live_only and r.status in ("completed", "invalidated", "superseded"):
            continue
        for e in r.tap3_events():
            if live_only and e.failed:
                continue
            if since is not None and e.time < since:
                continue
            bars_ago = (r.n_bars - 1 - e.idx) if r.n_bars else None
            if max_bars_ago is not None and bars_ago is not None and bars_ago > max_bars_ago:
                continue
            alerts.append({
                "symbol": symbol,
                "timeframe": r.tf,
                "side": r.side,
                "model": e.model,
                "time": str(e.time),
                "price": e.price,
                "note": e.note,
                "range": {"climax": r.climax_price, "reaction": r.reaction_price,
                          "eq": r.eq, "tap2": r.tap2_price},
                "red_flags": list(r.red_flags),
                "status": r.status,
                "bars_ago": bars_ago,
                "tap3_state": ("failed" if e.failed else
                               ("confirmed" if e.confirmed else
                                ("potential" if e.model != "model1" else "forming"))),
                "level_tf": e.level_tf,
                "stop": e.extreme if e.model == "model1" else None,
                "level": e.level if e.model != "model1" else None,
                "chart": tradingview_url(tv_symbol, r.tf),
                "range_obj": r,
                "id": f"{symbol}|{r.tf}|{r.side}|{r.climax_price}|{r.reaction_price}|{e.model}",
            })
    return alerts


def format_alert(a: dict) -> str:
    flags = f"  ⚠ {', '.join(a['red_flags'])}" if a["red_flags"] else ""
    return (f"[{a['timeframe']}] {a['symbol']} {a['side']} TAP 3 ({a['model']}) "
            f"@ {a['price']:.6g} on {a['time']}{flags}\n    {a['chart']}")


class SeenStore:
    """Remembers alert ids so re-running a scan doesn't re-alert."""

    def __init__(self, path: str):
        self.path = path
        self.ids = set(json.load(open(path))) if os.path.exists(path) else set()

    def new(self, alerts):
        fresh = [a for a in alerts if a["id"] not in self.ids]
        self.ids.update(a["id"] for a in fresh)
        return fresh

    def save(self):
        json.dump(sorted(self.ids), open(self.path, "w"))

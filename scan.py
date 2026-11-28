"""Scheduled scan: run by GitHub Actions every 30 minutes.

  * reads config.json
  * scans US stocks (Alpaca free tier) during US market hours, crypto (Coinbase) always
  * writes docs/data/signals.json for the dashboard
  * sends phone alerts through ntfy for new tap-3 events

Environment (GitHub secrets): APCA_API_KEY_ID, APCA_API_SECRET_KEY, NTFY_TOPIC.
Optional: SCAN_MARKETS=stocks,crypto to force markets; DASHBOARD_URL.

Run locally:  python scan.py            (uses the same environment variables)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from wyckoff_scanner.alerts import tradingview_url
from wyckoff_scanner.bojan import bojans_for_tap3
from wyckoff_scanner.confluence import zones_at_tap3
from wyckoff_scanner.timeframes import scan_frames

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "data" / "signals.json"
PUSHED = ROOT / "state" / "pushed.json"
NY = ZoneInfo("America/New_York")
MODEL_NAME = {"model1": "Model 1", "model2_liquidity": "Model 2 · extreme liquidity",
              "model2_demand": "Model 2 · extreme demand", "model2_supply": "Model 2 · extreme supply"}


# ----------------------------------------------------------------- helpers --
def iso(t) -> str | None:
    if t is None:
        return None
    ts = pd.Timestamp(t, tz="UTC") if isinstance(t, (int, float)) and not isinstance(t, bool) else pd.Timestamp(t)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def rnd(x, sig=8):
    if x is None:
        return None
    return float(f"{x:.{sig}g}")


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


def stock_market_open(now: datetime) -> bool:
    """Weekdays 9:00-17:00 New York: covers the session plus the daily close."""
    ny = now.astimezone(NY)
    return ny.weekday() < 5 and 9 <= ny.hour < 17


# -------------------------------------------------------------- signals ----
def signal_records(market: str, symbol: str, tv_symbol: str, frames: dict, ranges, cfg) -> list[dict]:
    hist_cut = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=cfg["display"].get("history_days", 14))
    max_live = cfg["display"].get("max_live_bars", 30)
    out = []
    for r in ranges:
        taps = r.tap3_events()
        if not taps:
            continue
        df = frames[r.tf]
        idx = df.index
        for e in taps:
            t3_time = pd.Timestamp(e.time)
            state = ("failed" if e.failed else "confirmed" if e.confirmed else
                     "forming" if e.model == "model1" else "potential")
            bars_ago = int(r.n_bars - 1 - e.idx)
            live = r.status not in ("completed", "invalidated", "superseded") and not e.failed
            stale = live and bars_ago > max_live        # played out long ago: treat as history
            live = live and not stale
            if not live and t3_time < hist_cut:      # the history window applies to finished setups only
                continue
            sid = slug(f"{symbol}-{r.tf}-{r.side}-{iso(r.times.get('climax'))}-{e.model}")
            start = max(0, r.climax_idx - 30)
            win = df.iloc[start:][-500:]
            candles = [[int(t.value // 10**9), rnd(o), rnd(h), rnd(l), rnd(c)]
                       for t, o, h, l, c in zip(win.index, win["open"], win["high"], win["low"], win["close"])]
            acc = r.side == "accumulation"
            ext_i = e.extreme_idx if e.extreme_idx is not None else e.idx
            markers = [
                {"time": iso(idx[r.climax_idx]), "price": rnd(r.climax_price), "label": "1 SC" if acc else "1 BC", "kind": "tap"},
                {"time": iso(idx[r.reaction_idx]), "price": rnd(r.reaction_price), "label": "AR", "kind": "ar"},
            ]
            if r.tap2_idx is not None:
                markers.append({"time": iso(idx[r.tap2_idx]), "price": rnd(r.tap2_price), "label": "2", "kind": "tap"})
            markers.append({"time": iso(idx[min(ext_i, len(idx) - 1)]), "price": rnd(e.extreme if e.extreme is not None else e.price),
                            "label": "3 " + {"model1": "M1"}.get(e.model, "M2"), "kind": "tap3"})
            q = r.climax_price + 0.25 * (r.reaction_price - r.climax_price)
            zone = None
            if r.demand_zone is not None:
                zone = {"lo": rnd(r.demand_zone[0]), "hi": rnd(r.demand_zone[1]), "tf": r.demand_tf,
                        "ob": iso(r.demand_src["ob"]) if r.demand_src else None,
                        "fvg": iso(r.demand_src["fvg"]) if r.demand_src else None}
            tap3_zones = []
            if e.model == "model1" and r.tap2_idx is not None:
                acc_ = r.side == "accumulation"
                a, b = r.tap2_idx + 1, max(ext_i, r.tap2_idx + 1)
                seg = df["high"].to_numpy()[a:b] if acc_ else -df["low"].to_numpy()[a:b]
                hi_i = a + int(np.argmax(seg)) if len(seg) else a
                tol = cfg["rules"].get("tap3_zone_tolerance", 0.05) * abs(r.reaction_price - r.climax_price)
                tap3_zones = [{"tf": z["tf"], "lo": rnd(z["lo"]), "hi": rnd(z["hi"]), "inside": z["inside"],
                               "distance": rnd(z["distance"]), "ob": iso(z["ob"]), "fvg": iso(z["fvg"])}
                              for z in zones_at_tap3(frames, r.side, idx[min(hi_i, len(idx) - 1)],
                                                     e.extreme if e.extreme is not None else e.price, tol)]
            ei = min(ext_i, len(idx) - 1)
            ext_end = idx[ei + 1] if ei + 1 < len(idx) else idx[ei] + (idx[-1] - idx[-2] if len(idx) > 1 else pd.Timedelta(0))
            rules_ = cfg["rules"]
            btol = rules_.get("tap3_zone_tolerance", 0.05) * abs(r.reaction_price - r.climax_price)
            bojans = [{"where": b["where"], "tf": b["tf"], "start": iso(b["start"]), "end": iso(b["end"]),
                       "time": iso(b["low_time"]), "price": rnd(b["price"]), "candles": b["candles"]}
                      for b in bojans_for_tap3(frames, r.tf, r.side, idx[ei], ext_end,
                                               e.extreme if e.extreme is not None else e.price, btol,
                                               lps_after=(idx[min(e.confirmed_idx, len(idx) - 1)]
                                                          if e.confirmed and not e.failed and e.confirmed_idx is not None else None),
                                               min_wick=rules_.get("bojan_min_wick", 0.4),
                                               depth=int(rules_.get("bojan_timeframes_below", 2)))]
            for b in bojans:
                if b["tf"] == r.tf:
                    markers.append({"time": b["time"], "price": b["price"], "kind": "bojan",
                                    "label": "Bojan" if b["where"] == "tap3" else ("Bojan LPS" if acc else "Bojan LPSY")})
            liq = None
            if r.liquidity_level is not None:
                liq = {"level": rnd(r.liquidity_level), "tf": r.liquidity_tf,
                       "low": iso(r.liquidity_src["low"]) if r.liquidity_src else None}
            timeline = [
                {"time": iso(r.times.get("climax")), "title": f"Tap 1 · {'selling' if acc else 'buying'} climax",
                 "detail": f"Range {'low' if acc else 'high'} {rnd(r.climax_price, 6)}"},
                {"time": iso(r.times.get("reaction")), "title": "Automatic " + ("rally" if acc else "reaction"),
                 "detail": f"Range {'high' if acc else 'low'} {rnd(r.reaction_price, 6)}"},
                {"time": iso(r.times.get("confirmed")), "title": "Range confirmed", "detail": f"Traded into the EQ {rnd(r.eq, 6)}"},
            ]
            for ev in r.events:
                if ev.kind == "tap2_start":
                    timeline.append({"time": iso(ev.time), "title": "Tap 2 · deviation", "detail": "Traded past the range"})
                elif ev.kind == "tap2_extended":
                    timeline.append({"time": iso(ev.time), "title": "Tap 2 extended", "detail": ev.note})
                elif ev.kind == "tap2_complete":
                    timeline.append({"time": iso(ev.time), "title": "Tap 2 complete", "detail": f"Back in the range ({ev.note})"})
                elif ev.kind == "tap3":
                    timeline.append({"time": iso(ev.time), "title": f"Tap 3 · {MODEL_NAME.get(ev.model, ev.model)}",
                                     "detail": ev.note.split(" | ")[0]})
                elif ev.kind in ("invalidated", "completed", "superseded"):
                    timeline.append({"time": iso(ev.time), "title": ev.kind.capitalize(), "detail": ev.note})
                elif ev.kind == "red_flag":
                    timeline.append({"time": iso(ev.time), "title": "Red flag", "detail": ev.note})
            out.append({
                "id": sid, "market": market, "symbol": symbol, "tf": r.tf, "side": r.side,
                "model": e.model, "model_name": MODEL_NAME.get(e.model, e.model),
                "state": state, "live": live, "status": r.status, "invalid_reason": r.invalid_reason,
                "time": iso(e.time), "price": rnd(e.price), "bars_ago": bars_ago, "stale": stale,
                "level": rnd(e.level) if e.model != "model1" else None, "level_tf": e.level_tf,
                "stop": rnd(e.extreme) if e.model == "model1" else None,
                "extreme": rnd(e.extreme), "note": e.note,
                "flags": list(r.red_flags),
                "tv": tradingview_url(tv_symbol, r.tf),
                "range": {"climax": rnd(r.climax_price), "reaction": rnd(r.reaction_price), "eq": rnd(r.eq),
                          "dev_limit": rnd(r.dev_limit), "opposite_limit": rnd(r.opposite_limit),
                          "model1_limit": rnd(r.model1_limit), "tap2": rnd(r.tap2_price),
                          "tap2_return": rnd(q) if r.tap2_return in ("either", "quarter") else None,
                          "tap2_closed_outside": r.tap2_closed_outside},
                "zone": zone, "tap3_zones": tap3_zones, "bojans": bojans, "liquidity": liq, "markers": markers, "timeline": timeline,
                "candles": candles,
            })
    return out


# ----------------------------------------------------------------- zones ---


def zone_cfg(cfg) -> dict:
    z = dict(cfg.get("zones") or {})
    from wyckoff_scanner.zones import CRYPTO_ZONE_TFS, STOCK_ZONE_TFS
    z.setdefault("enabled", True)
    z.setdefault("crypto_timeframes", CRYPTO_ZONE_TFS)
    z.setdefault("stock_timeframes", STOCK_ZONE_TFS)
    z.setdefault("keep_candles", 6)
    z.setdefault("touch_tolerance", 0.05)
    z.setdefault("min_imbalance_ratio", 2.0)
    w = dict(z.get("watch") or {})
    w.setdefault("enabled", True)
    w.setdefault("crypto_timeframes", z["crypto_timeframes"])
    w.setdefault("stock_timeframes", z["stock_timeframes"])
    w.setdefault("min_imbalance_ratio", 3.0)
    w.setdefault("max_distance_pct", 20)
    w.setdefault("structure", True)
    w.setdefault("order_blocks", True)
    z["watch"] = w
    return z


def zone_on(zc) -> bool:
    return bool(zc["enabled"] or zc["watch"]["enabled"])


def zone_tfs(zc, key) -> list[str]:
    from wyckoff_scanner.zones import tf_rank
    return sorted(set(zc[key]) | set(zc["watch"][key]), key=tf_rank)


def tv_link(tv_symbol: str, tf: str) -> str:
    from wyckoff_scanner.zones import TV_INTERVAL
    return f"https://www.tradingview.com/chart/?symbol={urllib.parse.quote(tv_symbol)}&interval={TV_INTERVAL.get(tf, 'D')}"


def zone_records(market: str, symbol: str, tv_symbol: str, frames: dict, cfg) -> tuple[list[dict], dict]:
    """Zone groups for one symbol (touched list and watch list), plus the candles
    each group's chart needs."""
    from wyckoff_scanner.zones import group_zones, scan_watch, scan_zones, tf_rank
    zc = zone_cfg(cfg)
    wc = zc["watch"]
    pick = lambda c: sorted([t for t in (c["crypto_timeframes"] if market == "crypto" else c["stock_timeframes"])
                             if t in frames], key=tf_rank)
    tfs, wtfs = pick(zc), pick(wc)
    tol = float(zc["touch_tolerance"])
    touched = scan_zones(frames, tfs, keep=int(zc["keep_candles"]), min_wick=cfg["rules"].get("bojan_min_wick", 0.4),
                         touch_tol=tol, min_ratio=float(zc["min_imbalance_ratio"])) if zc["enabled"] else []
    kinds = tuple(k for k, on in (("structure", wc["structure"]), ("ob", wc["order_blocks"])) if on)
    watch = scan_watch(frames, wtfs, touch_tol=tol, min_ratio=float(wc["min_imbalance_ratio"]),
                       max_distance=float(wc["max_distance_pct"]), kinds=kinds) if wc["enabled"] else []
    recs, candles = [], {}
    low = sorted(frames, key=tf_rank)
    last = frames[low[0]].iloc[-1] if low else None
    price = float(last["close"]) if last is not None else None

    def record(g, lst):
        top = g[0]
        tix = pd.DatetimeIndex(frames[top.tf].index)
        gid = re.sub(r"[^A-Za-z0-9_-]", "-", ("watch-" if lst == "watch" else "") +
                     f"{symbol}-{top.side}-{top.tf}-{iso(tix[top.first])}")
        lo, hi = min(z.lo for z in g), max(z.hi for z in g)
        zs, bojans = [], []
        for z in g:
            t = pd.DatetimeIndex(frames[z.tf].index)
            zs.append({"tf": z.tf, "kind": z.kind, "lo": rnd(z.lo), "hi": rnd(z.hi), "ratio": round(z.ratio, 1),
                       "start": iso(t[z.first]), "end": iso(t[z.last]), "fvg": iso(t[z.fvg]),
                       "touch": iso(t[z.touch]) if z.touch is not None else None,
                       "touch_bars_ago": int(len(t) - 1 - z.touch) if z.touch is not None else None,
                       "tv": tv_link(tv_symbol, z.tf)})
            for b in z.bojans:
                bi = frames[b["tf"]].index
                bojans.append({"zone_tf": z.tf, "tf": b["tf"], "state": b["state"], "start": iso(b["start"]),
                               "bars_ago": int(len(bi) - 1 - bi.get_loc(b["end"])),
                               "end": iso(b["end"]), "time": iso(b["time"]), "price": rnd(b["price"]),
                               "candles": b["candles"], "tv": tv_link(tv_symbol, b["tf"])})
        seen, uniq = set(), []            # one Bojan per (tf, start): keep it under its highest zone
        for b in bojans:
            k = (b["tf"], b["start"])
            if k not in seen:
                seen.add(k)
                uniq.append(b)
        rec = {"id": gid, "list": lst, "market": market, "symbol": symbol, "side": top.side,
               "tfs": list(dict.fromkeys(z.tf for z in g)), "lo": rnd(lo), "hi": rnd(hi),
               "price": rnd(price) if price is not None else None, "zones": zs, "bojans": uniq,
               "tv": tv_link(tv_symbol, top.tf), "chart": symbol_file(symbol)}
        if lst == "watch":
            dist = ((price - hi) if top.side == "demand" else (lo - price)) / price * 100 if price else None
            rec.update(state="watching", distance=round(dist, 2) if dist is not None else None,
                       touch=None, formed=max(z["fvg"] for z in zs))
        else:
            in_zone = last is not None and float(last["low"]) <= hi and float(last["high"]) >= lo
            rec.update(state="in zone" if in_zone else "touched", touch=max(z["touch"] for z in zs))
        recs.append(rec)
        for tf in {z.tf for z in g} | {b["tf"] for b in uniq}:
            candles.setdefault(symbol_file(symbol), {})[tf] = None

    for g in group_zones(touched):
        record(g, "touched")
    for g in group_zones(watch):
        record(g, "watch")
    for fname, tfmap in candles.items():
        for tf in tfmap:
            tfmap[tf] = [[int(ts.value // 10**9), rnd(o), rnd(h), rnd(l), rnd(c)]
                         for ts, o, h, l, c in frames[tf][["open", "high", "low", "close"]].iloc[-120:].itertuples()]
    return recs, candles


def symbol_file(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", symbol)


def crypto_zone_frames(natives: dict, tfs: list[str]) -> dict:
    from wyckoff_scanner.zones import resample_clock
    out = {}
    for tf in tfs:
        src = natives.get("1h") if tf.endswith("h") else natives.get("1D")
        if src is None or src.empty:
            continue
        out[tf] = src if tf == "1D" else resample_clock(src, tf)
    return out


def stock_zone_frames(cache, symbol: str, tfs: list[str]) -> dict:
    from wyckoff_scanner.session import session_resample
    from wyckoff_scanner.zones import resample_clock
    out = {}
    d = cache.load(symbol, "1Day")
    m30 = cache.load(symbol, "30Min")
    for tf in tfs:
        if tf == "1D" and d is not None:
            out[tf] = d
        elif tf.endswith("W") and d is not None:
            out[tf] = resample_clock(d, tf)
        elif tf.endswith("h") and m30 is not None:
            out[tf] = session_resample(m30, int(tf[:-1]) * 60)
        elif tf.endswith("D") and d is not None:
            out[tf] = resample_clock(d, tf)
    return out


# --------------------------------------------------------------- markets ---
def scan_stocks(cfg) -> tuple[list[dict], dict]:
    from wyckoff_scanner.alpaca import TF_SOURCE, AlpacaClient, BarCache, stock_frames, update_cache
    sc, rules = cfg["stocks"], cfg["rules"]
    client = AlpacaClient(feed=sc.get("feed", "sip"))
    if not client.key:
        raise RuntimeError("APCA_API_KEY_ID / APCA_API_SECRET_KEY secrets are not set")
    universe = client.assets()
    uni = sc.get("universe", "my list")
    if uni.startswith("most active"):
        want = client.most_actives(100)
    elif uni == "all US stocks":
        want = None
    else:
        want = [s.upper() for s in sc["tickers"]]
    if want is not None:
        byname = {u["symbol"]: u for u in universe}
        missing = [s for s in want if s not in byname]
        universe = [byname[s] for s in want if s in byname]
        if missing:
            print("stocks not found:", " ".join(missing))
    else:
        missing = []
    cache = BarCache(str(ROOT / ".bar_cache"))
    syms = [u["symbol"] for u in universe]
    zc = zone_cfg(cfg)
    srcs = {TF_SOURCE[t] for t in sc["timeframes"]}
    zon, ztfs = zone_on(zc), zone_tfs(zc, "stock_timeframes")
    if zon:
        srcs |= {"1Day"} | ({"30Min"} if any(t.endswith("h") for t in ztfs) else set())
    for src in sorted(srcs):
        print(f"  downloading {src} for {len(syms)} stocks")
        update_cache(client, cache, syms, src)
    sigs, zrecs, zcandles = [], [], {}
    for u in universe:
        if zon:
            try:
                r_, c_ = zone_records("stocks", u["symbol"], u["tv"],
                                      stock_zone_frames(cache, u["symbol"], ztfs), cfg)
                zrecs += r_
                zcandles.update(c_)
            except Exception as e:  # noqa: BLE001
                print(f"  zones {u['symbol']}: {e}")
        try:
            frames = stock_frames(cache, u["symbol"], sc["timeframes"])
            if not frames:
                continue
            ranges = scan_frames(frames, require_bos=rules["require_ar_break_of_structure"],
                                 allow_nested=rules["allow_nested_ranges"], tap2_return=rules["tap2_return"],
                                 trend_lookback=int(rules.get("trend_lookback", 50)),
                                 opposite_max=int(rules.get("max_opposite_deviations", 1)),
                                 tap3_spacing=float(rules.get("min_tap3_spacing", 0.25)))
            sigs += signal_records("stocks", u["symbol"], u["tv"], frames, ranges, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"  {u['symbol']}: {e}")
    return sigs, {"symbols": len(syms), "missing": missing, "zones": zrecs, "zone_candles": zcandles}


def scan_crypto(cfg) -> tuple[list[dict], dict]:
    from wyckoff_scanner.data import coinbase_frames, coinbase_products, coinbase_top_products, coinbase_tv_symbol
    cc, rules = cfg["crypto"], cfg["rules"]
    allp = coinbase_products("SPOT")
    uni = cc.get("universe", "my list")
    if uni == "my list":
        byid = {p["product_id"]: p for p in allp}
        want = [x.upper() for x in cc["products"]]
        missing = [w for w in want if w not in byid]
        prods = [byid[w] for w in want if w in byid]
        if missing:
            print("crypto not found on Coinbase spot:", " ".join(missing))
    else:
        missing = []
        n = None if uni == "all USD pairs" else int(uni.split()[1])
        prods = coinbase_top_products(n, products=allp)
    from wyckoff_scanner import data as D
    zc = zone_cfg(cfg)
    zon, ztfs = zone_on(zc), zone_tfs(zc, "crypto_timeframes")
    zh = any(t.endswith("h") for t in ztfs)
    sigs, zrecs, zcandles = [], [], {}
    for p in prods:
        pid = p["product_id"]
        natives: dict = {}

        def fetch(product_id, src, bars, _n=natives):
            have = _n.get(src)
            if have is None or len(have) < bars:
                _n[src] = D._coinbase_native(product_id, src, bars)
            return _n[src].iloc[-bars:]
        if zon:
            try:
                if zh:
                    fetch(pid, "1h", int(zc.get("crypto_hourly_bars", 3000)))
                fetch(pid, "1D", int(zc.get("crypto_daily_bars", 3200)))
                r_, c_ = zone_records("crypto", pid, coinbase_tv_symbol(p),
                                      crypto_zone_frames(natives, ztfs), cfg)
                zrecs += r_
                zcandles.update(c_)
            except Exception as e:  # noqa: BLE001
                print(f"  zones {pid}: {e}")
        try:
            frames = coinbase_frames(p["product_id"], cc["timeframes"], fetch=fetch)
            if not frames:
                continue
            ranges = scan_frames(frames, require_bos=rules["require_ar_break_of_structure"],
                                 allow_nested=rules["allow_nested_ranges"], tap2_return=rules["tap2_return"],
                                 trend_lookback=int(rules.get("trend_lookback", 50)),
                                 opposite_max=int(rules.get("max_opposite_deviations", 1)),
                                 tap3_spacing=float(rules.get("min_tap3_spacing", 0.25)))
            sigs += signal_records("crypto", p["product_id"], coinbase_tv_symbol(p), frames, ranges, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"  {p['product_id']}: {e}")
    return sigs, {"symbols": len(prods), "missing": missing, "zones": zrecs, "zone_candles": zcandles}


# ---------------------------------------------------------------- alerts ---
def dashboard_url() -> str:
    if os.environ.get("DASHBOARD_URL"):
        return os.environ["DASHBOARD_URL"].rstrip("/") + "/"
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if "/" in repo:
        owner, name = repo.split("/", 1)
        return f"https://{owner.lower()}.github.io/{name}/"
    return ""


def alert_events(signals: list[dict], cfg) -> list[tuple[str, dict, str]]:
    a = cfg["alerts"]
    out = []
    for s in signals:
        if not s["live"]:
            continue
        fresh = s["bars_ago"] <= a.get("max_bars_ago", 5)
        if s["state"] == "potential" and a.get("tap3_potential") and fresh:
            out.append((f"{s['id']}|potential", s, "Tap 3 potential"))
        if s["state"] == "forming" and a.get("model1_forming") and fresh:
            out.append((f"{s['id']}|forming", s, "Model 1 forming"))
        if s["state"] == "confirmed" and a.get("tap3_confirmed"):
            out.append((f"{s['id']}|confirmed", s, "Tap 3 confirmed"))
    return out


def send_ntfy(topic: str, s: dict, what: str, tz: str, opener=None) -> None:
    side = "Accumulation" if s["side"] == "accumulation" else "Distribution"
    when = pd.Timestamp(s["time"]).tz_convert(tz).strftime("%b %d %H:%M %Z")
    lines = [f"{side} · {s['model_name']}", f"@ {s['price']:g} · {when}"]
    if s["level"] is not None:
        lines.append(f"Level {s['level']:g}" + (f" ({s['level_tf']})" if s["level_tf"] else ""))
    if s["stop"] is not None:
        lines.append(f"Tap 3 extreme / stop {s['stop']:g}")
    if s.get("tap3_zones"):
        kind = "demand" if s["side"] == "accumulation" else "supply"
        lines.append(f"+ Tap 3 at fresh {kind}: " + ", ".join(
            f"{z['tf']} {z['lo']:g}-{z['hi']:g}" + ("" if z["inside"] else " (near)") for z in s["tap3_zones"]))
    if s.get("bojans"):
        lps = "LPS" if s["side"] == "accumulation" else "LPSY"
        lines.append("+ Bojan: " + ", ".join(f"{'tap 3' if b['where'] == 'tap3' else lps} {b['tf']}" for b in s["bojans"]))
    if s["flags"]:
        lines.append("Flags: " + ", ".join(s["flags"]))
    title = f"{s['symbol']} {s['tf']} - {what}"
    dash = dashboard_url()
    headers = {"Title": title.encode("latin-1", "replace").decode("latin-1"),
               "Tags": "chart_with_upwards_trend" if s["side"] == "accumulation" else "chart_with_downwards_trend",
               "Priority": "high",
               "Click": (dash + "#" + s["id"]) if dash else s["tv"],
               "Actions": f"view, TradingView, {s['tv']}"}
    req = urllib.request.Request(f"https://ntfy.sh/{topic}", data="\n".join(lines).encode(), method="POST",
                                 headers=headers)
    (opener or urllib.request.urlopen)(req, timeout=20)


def clean_topic(topic: str) -> str:
    """Accept 'wyckoff', 'ntfy.sh/wyckoff' or 'https://ntfy.sh/wyckoff'."""
    t = (topic or "").strip().rstrip("/")
    t = re.sub(r"^(https?://)?ntfy\.sh/", "", t)
    return t


def push_alerts(signals, cfg, topic, opener=None) -> int:
    topic = clean_topic(topic)
    events = alert_events(signals, cfg)
    first_run = not PUSHED.exists()
    seen = set(json.loads(PUSHED.read_text())) if PUSHED.exists() else set()
    sent = 0
    for key, s, what in events:
        if key in seen:
            continue
        if not first_run and topic:
            try:
                send_ntfy(topic, s, what, cfg["display"]["timezone"], opener=opener)
                sent += 1
            except Exception as e:  # noqa: BLE001
                print(f"  push failed for {s['symbol']}: {e}")
                continue
        seen.add(key)
    PUSHED.parent.mkdir(parents=True, exist_ok=True)
    PUSHED.write_text(json.dumps(sorted(seen), indent=0))
    if first_run:
        print(f"first run: recorded {len(seen)} existing alerts without sending (no flood)")
    return sent


def zone_alert_events(zones: list[dict], cfg) -> list[tuple[str, dict, str]]:
    """Only what just happened: a first touch on the newest or previous candle
    of the zone's timeframe, a Bojan that completed on the newest or previous
    candle of its timeframe."""
    a = cfg["alerts"]
    out = []
    for g in zones:
        if g.get("list") == "watch":
            continue
        for z in g["zones"]:
            if a.get("zone_hit", True) and z.get("touch_bars_ago", 0) <= 1:
                out.append((f"zone|{g['symbol']}|{g['side']}|{z['tf']}|{z['start']}|hit", g, f"{z['tf']} {g['side']} hit"))
        for b in g["bojans"]:
            if b["state"] == "formed" and a.get("zone_bojan", True) and b.get("bars_ago", 0) <= 1:
                out.append((f"zone|{g['symbol']}|{g['side']}|bojan|{b['tf']}|{b['start']}", g,
                            f"Bojan {'low' if g['side'] == 'demand' else 'high'} on the {b['tf']}"))
    return out


def send_zone_ntfy(topic: str, g: dict, what: str, opener=None) -> None:
    kind = "Demand" if g["side"] == "demand" else "Supply"
    lines = [f"{kind} {' · '.join(g['tfs'])}  {g['lo']:g} - {g['hi']:g}", f"Price {g['price']:g} ({g['state']})"]
    for z in g["zones"]:
        lines.append(f"{z['tf']} {'structure' if z['kind'] == 'structure' else 'order block'} {z['lo']:g}-{z['hi']:g}")
    for b in g["bojans"]:
        lines.append(f"Bojan {b['state']} on {b['tf']} at {b['price']:g}")
    dash = dashboard_url()
    headers = {"Title": f"{g['symbol']} - {what}".encode("latin-1", "replace").decode("latin-1"),
               "Tags": "green_circle" if g["side"] == "demand" else "red_circle",
               "Priority": "default",
               "Click": (dash + "#zone=" + g["id"]) if dash else g["tv"],
               "Actions": f"view, TradingView, {g['tv']}"}
    req = urllib.request.Request(f"https://ntfy.sh/{topic}", data="\n".join(lines).encode(), method="POST",
                                 headers=headers)
    (opener or urllib.request.urlopen)(req, timeout=20)


def push_zone_alerts(zones, cfg, topic, opener=None) -> int:
    topic = clean_topic(topic)
    seen = set(json.loads(PUSHED.read_text())) if PUSHED.exists() else set()
    seed = not any(k.startswith("zone|") for k in seen)      # first zone scan: record, don't flood
    cap = int(cfg["alerts"].get("zone_max_per_scan", 8))
    sent, extra, stop = 0, [], False
    new = [(k, g, w) for k, g, w in zone_alert_events(zones, cfg) if k not in seen]
    for key, g, what in new:
        seen.add(key)                       # never retried: a missed zone alert is better than a flood
        if seed or not topic or stop:
            continue
        if sent >= cap:
            extra.append(g["symbol"])
            continue
        try:
            send_zone_ntfy(topic, g, what, opener=opener)
            sent += 1
        except Exception as e:  # noqa: BLE001
            print(f"  zone push failed for {g['symbol']}: {e}")
            stop = "429" in str(e)
    if extra and topic and not stop:
        try:
            names = ", ".join(dict.fromkeys(extra))
            req = urllib.request.Request(f"https://ntfy.sh/{topic}", method="POST",
                                         data=f"{len(extra)} more zone alerts: {names}".encode(),
                                         headers={"Title": "More supply & demand hits", "Tags": "bell",
                                                  "Click": (dashboard_url() + "#zones") if dashboard_url() else ""})
            (opener or urllib.request.urlopen)(req, timeout=20)
            sent += 1
        except Exception as e:  # noqa: BLE001
            print(f"  zone summary push failed: {e}")
    if seed:
        seen.add("zone|seeded")
    PUSHED.parent.mkdir(parents=True, exist_ok=True)
    PUSHED.write_text(json.dumps(sorted(seen), indent=0))
    return sent


def write_zones(now, zones: dict, zcandles: dict, cfg) -> None:
    both = zones["stocks"] + zones["crypto"]
    allz = sorted([z for z in both if z.get("list") != "watch"], key=lambda z: z["touch"], reverse=True) + \
        sorted([z for z in both if z.get("list") == "watch"], key=lambda z: z.get("distance") or 0)
    (OUT.parent / 'zones.json').parent.mkdir(parents=True, exist_ok=True)
    (OUT.parent / 'zones.json').write_text(json.dumps({"generated_at": iso(now), "zones": allz}, separators=(",", ":")))
    (OUT.parent / 'zones').mkdir(parents=True, exist_ok=True)
    keep = {z["chart"] for z in allz if z.get("chart")}
    for f in (OUT.parent / 'zones').glob("*.json"):
        if f.stem not in keep:
            f.unlink()
    for name, c in zcandles.items():
        if name in keep:
            ((OUT.parent / 'zones') / f"{name}.json").write_text(json.dumps(c, separators=(",", ":")))


# ------------------------------------------------------------------ main ---
def main(now: datetime | None = None, opener=None) -> dict:
    now = now or datetime.now(timezone.utc)
    cfg = json.loads((ROOT / "config.json").read_text())
    forced = [m.strip().lower() for m in os.environ.get("SCAN_MARKETS", "").split(",") if m.strip()]
    if "all" in forced:
        forced = ["stocks", "crypto"]
    prev = json.loads(OUT.read_text()) if OUT.exists() else {"signals": [], "markets": {}}
    markets = dict(prev.get("markets", {}))
    signals = {"stocks": [s for s in prev.get("signals", []) if s["market"] == "stocks"],
               "crypto": [s for s in prev.get("signals", []) if s["market"] == "crypto"]}
    zprev = json.loads((OUT.parent / 'zones.json').read_text()) if (OUT.parent / 'zones.json').exists() else {"zones": []}
    zones = {m: [z for z in zprev.get("zones", []) if z["market"] == m] for m in ("stocks", "crypto")}
    zcandles: dict = {}

    todo = []
    if cfg["stocks"].get("enabled") and (("stocks" in forced) if forced else stock_market_open(now)):
        todo.append(("stocks", scan_stocks))
    if cfg["crypto"].get("enabled") and (("crypto" in forced) if forced else True):
        todo.append(("crypto", scan_crypto))

    for name, fn in todo:
        t0 = time.time()
        print(f"scanning {name}…")
        try:
            sigs, info = fn(cfg)
            signals[name] = sigs
            if "zones" in info:
                zones[name] = info["zones"]
                zcandles.update(info.get("zone_candles", {}))
            markets[name] = {"scanned_at": iso(now), "symbols": info["symbols"], "signals": len(sigs),
                             "missing": info.get("missing", []),
                             "seconds": round(time.time() - t0, 1), "error": None}
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            markets.setdefault(name, {})
            markets[name]["error"] = f"{type(e).__name__}: {e}"
            markets[name]["error_at"] = iso(now)
    for name in ("stocks", "crypto"):
        if name not in [n for n, _ in todo]:
            markets.setdefault(name, {})["skipped_at"] = iso(now)

    allsig = sorted(signals["stocks"] + signals["crypto"], key=lambda s: s["time"], reverse=True)
    sent = push_alerts([s for name, _ in todo for s in signals[name]], cfg, os.environ.get("NTFY_TOPIC", "").strip(),
                       opener=opener)
    sent += push_zone_alerts([z for name, _ in todo for z in zones[name]], cfg,
                             os.environ.get("NTFY_TOPIC", "").strip(), opener=opener)
    write_zones(now, zones, zcandles, cfg)
    out = {"generated_at": iso(now), "markets": markets, "alerts_sent": sent,
           "config": {**{k: cfg[k] for k in ("stocks", "crypto", "rules", "alerts", "display")}, "zones": zone_cfg(cfg)},
           "signals": allsig}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {len(allsig)} signals ({sum(s['live'] for s in allsig)} live), sent {sent} alert(s)")
    return out


def explain_symbol(symbol: str, tfs: list[str] | None = None) -> None:
    """Print why the scanner did or didn't find setups on one symbol."""
    from wyckoff_scanner.explain import explain
    cfg = json.loads((ROOT / "config.json").read_text())
    symbol = symbol.strip().upper()
    if "-" in symbol:                                   # Coinbase pair
        from wyckoff_scanner.data import coinbase_frames
        frames = coinbase_frames(symbol, tfs or cfg["crypto"]["timeframes"])
    else:
        from wyckoff_scanner.alpaca import TF_SOURCE, AlpacaClient, BarCache, stock_frames, update_cache
        tfs = tfs or cfg["stocks"]["timeframes"]
        client = AlpacaClient(feed=cfg["stocks"].get("feed", "sip"))
        cache = BarCache(str(ROOT / ".bar_cache"))
        for src in sorted({TF_SOURCE[t] for t in tfs}):
            update_cache(client, cache, [symbol], src)
        frames = stock_frames(cache, symbol, tfs)
    if not frames:
        print(f"No candles found for {symbol}.")
        return
    explain(symbol, frames, cfg["rules"], last=int(os.environ.get("EXPLAIN_LAST", "12")))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--explain":
        explain_symbol(sys.argv[2], sys.argv[3].split() if len(sys.argv) > 3 and sys.argv[3].strip() else None)
        sys.exit(0)
    sys.exit(0 if main() else 1)

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
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from wyckoff_scanner.alerts import tradingview_url
from wyckoff_scanner.confluence import zone_at_tap3
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
            tap3_zone = None
            if e.model == "model1" and r.tap2_idx is not None:
                acc_ = r.side == "accumulation"
                a, b = r.tap2_idx + 1, max(ext_i, r.tap2_idx + 1)
                seg = df["high"].to_numpy()[a:b] if acc_ else -df["low"].to_numpy()[a:b]
                hi_i = a + int(np.argmax(seg)) if len(seg) else a
                tol = cfg["rules"].get("tap3_zone_tolerance", 0.05) * abs(r.reaction_price - r.climax_price)
                z = zone_at_tap3(frames, r.side, idx[min(hi_i, len(idx) - 1)],
                                 e.extreme if e.extreme is not None else e.price, tol)
                if z is not None:
                    tap3_zone = {"tf": z["tf"], "lo": rnd(z["lo"]), "hi": rnd(z["hi"]), "inside": z["inside"],
                                 "distance": rnd(z["distance"]), "ob": iso(z["ob"]), "fvg": iso(z["fvg"])}
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
                "zone": zone, "tap3_zone": tap3_zone, "liquidity": liq, "markers": markers, "timeline": timeline,
                "candles": candles,
            })
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
    for src in sorted({TF_SOURCE[t] for t in sc["timeframes"]}):
        print(f"  downloading {src} for {len(syms)} stocks")
        update_cache(client, cache, syms, src)
    sigs = []
    for u in universe:
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
    return sigs, {"symbols": len(syms), "missing": missing}


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
    sigs = []
    for p in prods:
        try:
            frames = coinbase_frames(p["product_id"], cc["timeframes"])
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
    return sigs, {"symbols": len(prods), "missing": missing}


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
    if s.get("tap3_zone"):
        z = s["tap3_zone"]
        kind = "demand" if s["side"] == "accumulation" else "supply"
        lines.append(f"+ Tap 3 {'in' if z['inside'] else 'near'} fresh {z['tf']} {kind} {z['lo']:g}-{z['hi']:g}")
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
    out = {"generated_at": iso(now), "markets": markets, "alerts_sent": sent,
           "config": {k: cfg[k] for k in ("stocks", "crypto", "rules", "alerts", "display")},
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

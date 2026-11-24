"""Phone push notifications via ntfy (https://ntfy.sh) — free, no account.

Setup on the phone: install the ntfy app, tap "+", subscribe to a topic name
that's hard to guess (anyone who knows the topic can read it), e.g.
"wyckoff-taps-7f3k9q".  Tapping a notification opens the TradingView chart.
"""
from __future__ import annotations

import urllib.request

MODEL_TEXT = {"model1": "Model 1", "model2_liquidity": "Model 2 – extreme liquidity",
              "model2_demand": "Model 2 – extreme demand", "model2_supply": "Model 2 – extreme supply"}


def alert_text(a: dict) -> tuple[str, str]:
    """(title, body) for one tap-3 alert."""
    side = "Accumulation" if a["side"] == "accumulation" else "Distribution"
    title = f"{a['symbol']} {a['timeframe']} - Tap 3 {MODEL_TEXT.get(a['model'], a['model'])}"
    lines = [f"{side}, {a['tap3_state']} @ {a['price']:.6g}"]
    if a.get("level") is not None:
        lines.append(f"Level hit: {a['level']:.6g}")
    if a.get("stop") is not None:
        lines.append(f"Tap 3 extreme / stop: {a['stop']:.6g}")
    rng = a["range"]
    lines.append(f"Range {min(rng['climax'], rng['reaction']):.6g} - {max(rng['climax'], rng['reaction']):.6g}")
    if a["red_flags"]:
        lines.append("Flags: " + ", ".join(a["red_flags"]))
    lines.append(a.get("time_local") or f"{a['time'][:16]} UTC")
    return title, "\n".join(lines)


def _latin1(s: str) -> str:
    # HTTP header values must be latin-1
    return s.encode("latin-1", "replace").decode("latin-1")


def send_ntfy(topic: str, alert: dict, server: str = "https://ntfy.sh", opener=None) -> None:
    title, body = alert_text(alert)
    req = urllib.request.Request(
        f"{server.rstrip('/')}/{topic}", data=body.encode("utf-8"), method="POST",
        headers={"Title": _latin1(title), "Click": alert["chart"],
                 "Tags": "chart_with_upwards_trend" if alert["side"] == "accumulation"
                 else "chart_with_downwards_trend",
                 "Priority": "high"})
    (opener or urllib.request.urlopen)(req, timeout=20)


def push_new(topic: str, alerts: list[dict], seen_path: str = ".pushed_alerts.json",
             opener=None) -> int:
    """Push alerts not pushed before.  Returns how many were sent."""
    from .alerts import SeenStore
    seen = SeenStore(seen_path)
    fresh = seen.new(alerts)
    sent = 0
    for a in fresh:
        try:
            send_ntfy(topic, a, opener=opener)
            sent += 1
        except Exception as e:  # noqa: BLE001
            seen.ids.discard(a["id"])          # retry next run
            print(f"push failed for {a['symbol']}: {e}")
    seen.save()
    return sent

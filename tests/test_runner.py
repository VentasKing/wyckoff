import json, os, sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import pytest

import scan
from synth import mirror_prices
from test_alpaca import FakeAlpaca, daily
from test_scanner import model1_live, mtf_case


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Run scan.py in a temp folder with fake Alpaca + Coinbase."""
    (tmp_path / "config.json").write_text((scan.ROOT / "config.json").read_text())
    monkeypatch.setattr(scan, "ROOT", tmp_path)
    monkeypatch.setattr(scan, "OUT", tmp_path / "docs" / "data" / "signals.json")
    monkeypatch.setattr(scan, "PUSHED", tmp_path / "state" / "pushed.json")
    cfg = json.loads((tmp_path / "config.json").read_text())
    cfg["stocks"]["tickers"] = ["AAPL", "IBM"]
    cfg["stocks"]["timeframes"] = ["1D"]
    cfg["crypto"]["products"] = ["BTC-USD", "ETH-USD", "NOPE-USD"]
    cfg["crypto"]["timeframes"] = ["15m", "1h"]
    cfg["display"]["history_days"] = 100000          # synthetic data is dated 2024-2025
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    import wyckoff_scanner.alpaca as A
    import wyckoff_scanner.data as D
    live = model1_live(); live.bar(95, 101, 94.5, 100.5)
    fake = FakeAlpaca({("AAPL", "1Day"): daily(live.df())}, page=50)
    orig = A.AlpacaClient
    monkeypatch.setattr(A, "AlpacaClient", lambda **kw: orig(key="PK1", secret="s", getter=fake))
    h1, m15 = mtf_case(False)
    frames = {"BTC-USD": {"15m": m15, "1h": h1},
              "ETH-USD": {"15m": mirror_prices(m15, 300), "1h": mirror_prices(h1, 300)}}
    monkeypatch.setattr(D, "coinbase_products", lambda *a, **k: [
        {"product_id": p, "base_currency_id": p.split("-")[0], "quote_currency_id": "USD"} for p in frames])
    monkeypatch.setattr(D, "_coinbase_native", lambda pid, tf, bars: frames[pid][tf])
    monkeypatch.setenv("GITHUB_REPOSITORY", "someone/wyckoff-scanner")
    return tmp_path


def test_scan_writes_dashboard_data_and_alerts_once(sandbox, monkeypatch):
    sent = []
    opener = lambda req, timeout: sent.append(req)
    monkeypatch.setenv("NTFY_TOPIC", "my-topic")
    monday_noon_ny = datetime(2026, 11, 23, 17, 0, tzinfo=timezone.utc)     # 12:00 New York

    out = scan.main(now=monday_noon_ny, opener=opener)
    sig = {(s["symbol"], s["model"]) for s in out["signals"]}
    assert ("AAPL", "model1") in sig and ("BTC-USD", "model2_liquidity") in sig
    assert ("ETH-USD", "model2_liquidity") in sig                         # distribution side works too
    assert out["markets"]["stocks"]["symbols"] == 2 and out["markets"]["crypto"]["symbols"] == 2
    s = next(x for x in out["signals"] if x["symbol"] == "BTC-USD")
    assert s["liquidity"]["tf"] == "15m" and s["candles"] and len(s["markers"]) == 4
    assert s["tv"].endswith("COINBASE%3ABTCUSD&interval=60")
    assert sent == []                                                      # first run: no flood

    # a new event on the next run is pushed once, with dashboard + TradingView links
    pushed = json.loads(scan.PUSHED.read_text())
    scan.PUSHED.write_text(json.dumps([k for k in pushed if not k.startswith(s["id"])]))
    scan.main(now=monday_noon_ny, opener=opener)
    assert len(sent) == 1
    req = sent[0]
    assert req.full_url == "https://ntfy.sh/my-topic"
    assert req.get_header("Click") == f"https://someone.github.io/wyckoff-scanner/#{s['id']}"
    assert "TradingView" in req.get_header("Actions") and "BTC-USD 1h" in req.get_header("Title")
    scan.main(now=monday_noon_ny, opener=opener)
    assert len(sent) == 1                                                  # never twice


def test_stocks_skipped_outside_market_hours_but_kept(sandbox):
    monday_noon_ny = datetime(2026, 11, 23, 17, 0, tzinfo=timezone.utc)
    sunday = datetime(2026, 11, 22, 17, 0, tzinfo=timezone.utc)
    scan.main(now=monday_noon_ny)
    out = scan.main(now=sunday)
    assert out["markets"]["stocks"]["scanned_at"] == "2026-11-23T17:00:00Z"
    assert out["markets"]["stocks"]["skipped_at"] == "2026-11-22T17:00:00Z"
    assert any(s["symbol"] == "AAPL" for s in out["signals"])             # previous stock results kept
    assert out["markets"]["crypto"]["scanned_at"] == "2026-11-22T17:00:00Z"


def test_market_error_is_recorded_not_fatal(sandbox, monkeypatch):
    import wyckoff_scanner.alpaca as A
    monkeypatch.setattr(A, "AlpacaClient", lambda **kw: A.AlpacaClient.__wrapped__ if False else type(
        "X", (), {"key": "", "__init__": lambda self, **k: None})())
    out = scan.main(now=datetime(2026, 11, 23, 17, 0, tzinfo=timezone.utc))
    assert "not set" in out["markets"]["stocks"]["error"]
    assert out["markets"]["crypto"]["error"] is None


def test_topic_accepts_full_address():
    for t in ["wyckoff", "ntfy.sh/wyckoff", "https://ntfy.sh/wyckoff", " https://ntfy.sh/wyckoff/ "]:
        assert scan.clean_topic(t) == "wyckoff"

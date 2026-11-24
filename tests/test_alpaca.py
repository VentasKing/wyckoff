import os, sys, argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import pytest

from test_scanner import model1_path
from wyckoff_scanner.alpaca import AlpacaClient, BarCache, stock_frames, update_cache
from wyckoff_scanner.session import session_resample
from wyckoff_scanner.__main__ import run_stocks


ASSETS = [
    {"symbol": "AAPL", "exchange": "NASDAQ", "name": "Apple Inc. Common Stock", "tradable": True},
    {"symbol": "SPY", "exchange": "ARCA", "name": "SPDR S&P 500 ETF Trust", "tradable": True},
    {"symbol": "IBM", "exchange": "NYSE", "name": "IBM", "tradable": True},
    {"symbol": "CBOEX", "exchange": "BATS", "name": "Some ETF", "tradable": True},
    {"symbol": "XYZW", "exchange": "NASDAQ", "name": "XYZ Corp Warrants", "tradable": True},
    {"symbol": "PINK", "exchange": "OTC", "name": "Pink sheet co", "tradable": True},
]


class FakeAlpaca:
    """Mimics /assets and multi-symbol /stocks/bars with cross-symbol paging."""

    def __init__(self, data: dict, page=4):
        self.data, self.page, self.calls = data, page, []

    def __call__(self, url, params):
        self.calls.append((url, dict(params)))
        if url.endswith("/assets"):
            return ASSETS
        tf, start = params["timeframe"], pd.Timestamp(params["start"])
        flat = []
        for sym in sorted(params["symbols"].split(",")):
            df = self.data.get((sym, tf))
            if df is None:
                continue
            for t, r in df[df.index >= start].iterrows():
                flat.append((sym, {"t": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "o": r.open, "h": r.high,
                                   "l": r.low, "c": r.close, "v": 100}))
        off = int(params.get("page_token") or 0)
        chunk = flat[off: off + self.page]
        out = {}
        for sym, b in chunk:
            out.setdefault(sym, []).append(b)
        nxt = str(off + self.page) if off + self.page < len(flat) else None
        return {"bars": out, "next_page_token": nxt}


def daily(df):
    df = df.copy()
    df.index = pd.date_range("2025-01-01 05:00", periods=len(df), freq="1D", tz="UTC")
    return df


def test_asset_universe_and_tv_prefixes():
    c = AlpacaClient(key="PK1", secret="s", getter=FakeAlpaca({}))
    u = {a["symbol"]: a["tv"] for a in c.assets()}
    assert u == {"AAPL": "NASDAQ:AAPL", "SPY": "AMEX:SPY", "IBM": "NYSE:IBM", "CBOEX": "CBOE:CBOEX"}
    assert "SPY" not in {a["symbol"] for a in c.assets(include_etfs=False)}


def test_paging_across_symbols():
    df = daily(model1_path().df())
    fake = FakeAlpaca({("AAPL", "1Day"): df, ("IBM", "1Day"): df}, page=7)
    c = AlpacaClient(key="PK1", secret="s", getter=fake)
    got = c.bars(["AAPL", "IBM"], "1Day", datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert len(got["AAPL"]) == len(df) and len(got["IBM"]) == len(df)
    assert got["IBM"]["low"].min() == pytest.approx(df["low"].min())
    assert "end" not in fake.calls[0][1]          # free plan: let Alpaca default end to 15 min ago


def test_cache_is_incremental(tmp_path):
    df = daily(model1_path().df())
    fake = FakeAlpaca({("AAPL", "1Day"): df}, page=1000)
    c = AlpacaClient(key="PK1", secret="s", getter=fake)
    cache = BarCache(tmp_path)
    now = df.index[-1].to_pydatetime()
    update_cache(c, cache, ["AAPL"], "1Day", now=now)
    assert len(cache.load("AAPL", "1Day")) == len(df)
    update_cache(c, cache, ["AAPL"], "1Day", now=now)
    assert pd.Timestamp(fake.calls[-1][1]["start"]) == df.index[-1]   # only re-asks from last bar
    assert len(cache.load("AAPL", "1Day")) == len(df)


def test_session_resample_matches_tradingview():
    idx = pd.date_range("2025-03-03 04:00", "2025-03-03 19:55", freq="5min", tz="America/New_York")
    df = pd.DataFrame({"open": range(len(idx)), "high": range(len(idx)), "low": range(len(idx)),
                       "close": range(len(idx))}, index=idx.tz_convert("UTC"), dtype=float)
    h4 = session_resample(df, 240).index.tz_convert("America/New_York")
    assert [t.strftime("%H:%M") for t in h4] == ["09:30", "13:30"]
    h1 = session_resample(df, 60)
    assert [t.strftime("%H:%M") for t in h1.index.tz_convert("America/New_York")] == \
        ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
    first_rth = df.index.tz_convert("America/New_York").get_loc(pd.Timestamp("2025-03-03 09:30", tz="America/New_York"))
    assert h1.iloc[0]["open"] == df.iloc[first_rth]["open"]          # pre-market excluded


def test_stocks_command_end_to_end(tmp_path, capsys):
    df = daily(model1_path().df())
    fake = FakeAlpaca({("AAPL", "1Day"): df}, page=10)
    c = AlpacaClient(key="PK1", secret="s", getter=fake)
    args = argparse.Namespace(tfs=["1D"], symbols=None, limit=None, no_etfs=False, feed="sip",
                              cache=str(tmp_path / "c"), no_update=False,
                              seen=str(tmp_path / "seen.json"), all=False)
    assert run_stocks(args, client=c) == 2
    out = capsys.readouterr().out
    assert "https://www.tradingview.com/chart/?symbol=NASDAQ%3AAAPL&interval=D" in out
    assert run_stocks(args, client=c) == 0                         # already alerted


def test_falls_back_to_other_trading_server():
    from wyckoff_scanner.alpaca import AlpacaError, TRADING_LIVE
    def getter(url, params):
        if "paper-api" in url:
            raise AlpacaError(401, '{"message": "unauthorized."}', url)
        return ASSETS
    c = AlpacaClient(key=" PKXYZ123 \n", secret="sec ", getter=getter)
    assert c.key == "PKXYZ123" and c.secret == "sec"
    assert len(c.assets()) == 4
    assert c.trading == TRADING_LIVE


def test_bad_keys_give_plain_explanation():
    from wyckoff_scanner.alpaca import AlpacaError
    def getter(url, params):
        raise AlpacaError(401, '{"message": "unauthorized."}', url)
    c = AlpacaClient(key="PKXYZ123", secret="abc", getter=getter)
    with pytest.raises(AlpacaError) as e:
        c.assets()
    msg = str(e.value)
    assert "swapped" in msg and "PKXY…23" in msg and "secret length 3" in msg


def test_ntfy_push_once_with_chart_link(tmp_path):
    from test_scanner import model1_live
    from wyckoff_scanner import detect_ranges, tap3_alerts
    from wyckoff_scanner.notify import push_new
    p = model1_live(); p.bar(95, 101, 94.5, 100.5)
    rs = [r for r in detect_ranges(p.df(), "1D") if "inside_larger_range" not in r.red_flags]
    alerts = tap3_alerts("AAPL", "NASDAQ:AAPL", rs, live_only=True)
    sent = []
    opener = lambda req, timeout: sent.append(req)
    seen = str(tmp_path / "pushed.json")
    assert push_new("my-topic", alerts, seen, opener=opener) == len(alerts) >= 1
    req = sent[0]
    assert req.full_url == "https://ntfy.sh/my-topic"
    assert req.get_header("Click") == "https://www.tradingview.com/chart/?symbol=NASDAQ%3AAAPL&interval=D"
    assert "Tap 3" in req.get_header("Title") and b"stop" in req.data
    assert push_new("my-topic", alerts, seen, opener=opener) == 0      # no repeats


def test_coinbase_top_products_ranked_by_dollar_volume():
    from wyckoff_scanner.data import coinbase_top_products
    prods = [
        {"product_id": "BTC-USD", "base_currency_id": "BTC", "quote_currency_id": "USD", "approximate_quote_24h_volume": "900000000"},
        {"product_id": "USDT-USD", "base_currency_id": "USDT", "quote_currency_id": "USD", "approximate_quote_24h_volume": "999999999"},
        {"product_id": "ETH-BTC", "base_currency_id": "ETH", "quote_currency_id": "BTC", "approximate_quote_24h_volume": "5"},
        {"product_id": "DOGE-USD", "base_currency_id": "DOGE", "quote_currency_id": "USD", "volume_24h": "1000000", "price": "0.4"},
        {"product_id": "ETH-USD", "base_currency_id": "ETH", "quote_currency_id": "USD", "approximate_quote_24h_volume": "500000000"},
        {"product_id": "OLD-USD", "base_currency_id": "OLD", "quote_currency_id": "USD", "approximate_quote_24h_volume": "1e12", "view_only": True},
    ]
    assert [p["product_id"] for p in coinbase_top_products(None, products=prods)] == ["BTC-USD", "ETH-USD", "DOGE-USD"]
    assert [p["product_id"] for p in coinbase_top_products(2, products=prods)] == ["BTC-USD", "ETH-USD"]


def test_coinbase_frames_downloads_each_granularity_once():
    import numpy as np
    from wyckoff_scanner.data import coinbase_frames
    calls = []
    def fetch(pid, tf, bars):
        calls.append((tf, bars))
        freq = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "1D": "1D"}[tf]
        idx = pd.date_range("2024-01-01", periods=bars, freq=freq, tz="UTC")
        x = np.linspace(100, 110, bars)
        return pd.DataFrame({"open": x, "high": x + 1, "low": x - 1, "close": x}, index=idx)
    fr = coinbase_frames("BTC-USD", ["5m", "15m", "30m", "1h", "4h", "1D", "1W"], fetch=fetch)
    assert sorted(t for t, _ in calls) == ["15m", "1D", "1h", "30m", "5m"]   # 5 downloads, not 7+
    assert dict(calls)["1h"] == 1600 and dict(calls)["1D"] == 2800
    assert len(fr["4h"]) == 400 and len(fr["1h"]) == 400 and len(fr["5m"]) == 1000


def test_most_actives():
    seen = {}
    def getter(url, params):
        seen["url"], seen["params"] = url, params
        return {"most_actives": [{"symbol": "NVDA", "volume": 5}, {"symbol": "TSLA", "volume": 4}]}
    c = AlpacaClient(key="PK1", secret="s", getter=getter)
    assert c.most_actives(250) == ["NVDA", "TSLA"]
    assert seen["url"].endswith("/v1beta1/screener/stocks/most-actives") and seen["params"]["top"] == 100

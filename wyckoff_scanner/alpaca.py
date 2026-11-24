"""Alpaca free-tier loader for US stocks.

Free plan facts this module is built around:
  * historical bars from the SIP feed (all US exchanges) are free as long as
    the request doesn't ask for the most recent 15 minutes, so `end` is left
    out and Alpaca defaults it to 15 minutes ago
  * 200 requests/minute
  * one request can carry many symbols; `limit` (max 10,000) counts bars across
    all symbols, and results page via next_page_token

Keys: set APCA_API_KEY_ID and APCA_API_SECRET_KEY (a free paper account works).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .session import session_resample
from .timeframes import resample

DATA = "https://data.alpaca.markets/v2"
TRADING_PAPER = "https://paper-api.alpaca.markets/v2"
TRADING_LIVE = "https://api.alpaca.markets/v2"

# Alpaca exchange code -> TradingView prefix
TV_PREFIX = {"NASDAQ": "NASDAQ", "NYSE": "NYSE", "AMEX": "AMEX", "ARCA": "AMEX",
             "NYSEARCA": "AMEX", "BATS": "CBOE"}

# Which Alpaca bars each scanner timeframe is built from, and how
SOURCES = {          # source: (alpaca timeframe, calendar days of history to keep)
    "5Min": ("5Min", 30),
    "30Min": ("30Min", 200),
    "1Day": ("1Day", 365 * 5),
}
TF_SOURCE = {"5m": "5Min", "15m": "5Min", "30m": "30Min", "1h": "30Min", "4h": "30Min",
             "1D": "1Day", "1W": "1Day"}

SKIP_NAME_WORDS = (" warrant", " warrants", " unit", " units", " right", " rights")


class AlpacaError(RuntimeError):
    def __init__(self, code: int, body: str, url: str = ""):
        self.code, self.body, self.url = code, body, url
        super().__init__(f"Alpaca {code}: {body}")


AUTH_HELP = """Alpaca rejected the request ({code}): {body}

Most common causes:
  1. Key ID and secret swapped. APCA_API_KEY_ID is the shorter one (often starts with PK);
     APCA_API_SECRET_KEY is the long one.
  2. A space or line break was copied along with a key.
  3. The secret was lost. Alpaca only shows it once; if unsure, click "Regenerate" on the
     Alpaca dashboard (Paper Trading -> API Keys) and copy both values again.
  4. On GitHub: the values were pasted into the wrong secret (the secret key is usually
     about 40 characters long; the key ID is shorter).
Keys this run used: key id {kid} ({klen} characters), secret length {slen}."""


class AlpacaClient:
    def __init__(self, key: str | None = None, secret: str | None = None, feed: str = "sip",
                 per_minute: int = 190, trading_base: str | None = None, getter=None):
        self.key = (key or os.environ.get("APCA_API_KEY_ID") or "").strip()
        self.secret = (secret or os.environ.get("APCA_API_SECRET_KEY") or "").strip()
        self.feed = feed
        self.min_gap = 60.0 / per_minute
        self._last = 0.0
        self.trading = trading_base or os.environ.get("APCA_TRADING_BASE") or (
            TRADING_LIVE if self.key.startswith("AK") else TRADING_PAPER)
        self._getter = getter or self._http_get

    # ------------------------------------------------------------ http ------
    def _http_get(self, url: str, params: dict) -> dict:
        q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        req = urllib.request.Request(f"{url}?{q}" if q else url, headers={
            "APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret,
            "Accept": "application/json", "User-Agent": "wyckoff-scanner"})
        for attempt in range(6):
            wait = self._last + self.min_gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code == 429 or e.code >= 500:
                    time.sleep(min(60, 2 ** attempt))
                    continue
                raise AlpacaError(e.code, e.read().decode(errors="replace")[:300], url) from None
            except urllib.error.URLError as e:
                if attempt < 2:
                    time.sleep(2)
                    continue
                raise AlpacaError(0, f"could not connect to {url}: {e.reason}", url) from None
        raise AlpacaError(429, "too many retries (rate limited)", url)

    def get(self, url, params):
        return self._getter(url, params)

    # ----------------------------------------------------------- assets -----
    def assets(self, include_etfs: bool = True, skip_derivative_names: bool = True) -> list[dict]:
        """Active, tradable US stocks on NASDAQ/NYSE/AMEX/ARCA/BATS (no OTC)."""
        rows = self._assets_rows()
        out = []
        for a in rows:
            ex = a.get("exchange")
            if ex not in TV_PREFIX or not a.get("tradable", True):
                continue
            name = (a.get("name") or "").lower()
            if skip_derivative_names and any(name.endswith(w) or f"{w} " in name for w in SKIP_NAME_WORDS):
                continue
            if not include_etfs and ex in ("ARCA", "NYSEARCA"):
                continue
            out.append({"symbol": a["symbol"], "exchange": ex, "name": a.get("name", ""),
                        "tv": tv_symbol(a["symbol"], ex)})
        return sorted(out, key=lambda x: x["symbol"])

    def _assets_rows(self):
        """Try the trading server that matches the key, then the other one
        (paper keys only work on paper-api, live keys only on api)."""
        params = {"status": "active", "asset_class": "us_equity"}
        bases = [self.trading] + [b for b in (TRADING_PAPER, TRADING_LIVE) if b != self.trading]
        err = None
        for base in bases:
            try:
                rows = self.get(f"{base}/assets", params)
                self.trading = base
                return rows
            except AlpacaError as e:
                if e.code not in (401, 403):
                    raise
                err = e
        raise AlpacaError(err.code, self.explain(err), err.url)

    def explain(self, err: "AlpacaError") -> str:
        if err.code in (401, 403):
            kid = (self.key[:4] + "…" + self.key[-2:]) if len(self.key) > 6 else (repr(self.key) or "(empty)")
            return AUTH_HELP.format(code=err.code, body=err.body.strip()[:200], kid=kid,
                                    klen=len(self.key), slen=len(self.secret))
        return str(err)

    def most_actives(self, top: int = 100, by: str = "volume") -> list[str]:
        """Alpaca's screener: most active stocks right now by volume (or trades)."""
        data = self.get("https://data.alpaca.markets/v1beta1/screener/stocks/most-actives",
                        {"by": by, "top": min(int(top), 100)})
        return [x["symbol"] for x in data.get("most_actives", [])]

    # ------------------------------------------------------------- bars -----
    def bars(self, symbols: list[str], timeframe: str, start: datetime,
             adjustment: str = "split") -> dict[str, pd.DataFrame]:
        params = {"symbols": ",".join(symbols), "timeframe": timeframe,
                  "start": start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "limit": 10000, "adjustment": adjustment, "feed": self.feed}
        rows: dict[str, list] = {}
        token = None
        while True:
            data = self.get(f"{DATA}/stocks/bars", dict(params, page_token=token))
            for sym, bars in (data.get("bars") or {}).items():
                rows.setdefault(sym, []).extend(bars)
            token = data.get("next_page_token")
            if not token:
                break
        return {s: _to_df(b) for s, b in rows.items() if b}


def tv_symbol(symbol: str, exchange: str) -> str:
    return f"{TV_PREFIX.get(exchange, exchange)}:{symbol}"


def _to_df(bars: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(bars)
    out = pd.DataFrame({"open": df["o"].astype(float), "high": df["h"].astype(float),
                        "low": df["l"].astype(float), "close": df["c"].astype(float),
                        "volume": df["v"].astype(float)})
    out.index = pd.to_datetime(df["t"], utc=True)
    out.index.name = None
    return out[~out.index.duplicated(keep="last")].sort_index()


# ------------------------------------------------------------------ cache --
class BarCache:
    """One pickle per (symbol, source).  Later runs only download new bars."""

    def __init__(self, root: str = ".bar_cache"):
        self.root = Path(root)

    def _path(self, sym, src):
        return self.root / src / f"{sym.replace('/', '_')}.pkl"

    def load(self, sym, src) -> pd.DataFrame | None:
        p = self._path(sym, src)
        return pd.read_pickle(p) if p.exists() else None

    def save(self, sym, src, df):
        p = self._path(sym, src)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(p)


def update_cache(client: AlpacaClient, cache: BarCache, symbols: list[str], source: str,
                 batch: int = 100, now: datetime | None = None, progress=None) -> None:
    """Download missing bars for `symbols` into the cache (incremental)."""
    tf, keep_days = SOURCES[source]
    now = now or datetime.now(timezone.utc)
    floor = now - timedelta(days=keep_days)
    for b in range(0, len(symbols), batch):
        chunk = symbols[b:b + batch]
        existing = {s: cache.load(s, source) for s in chunk}
        lasts = [df.index[-1] for df in existing.values() if df is not None and len(df)]
        # re-fetch from the oldest "last bar" in the batch (the last bar may have been partial)
        start = min(lasts) if len(lasts) == len(chunk) else floor
        fresh = client.bars(chunk, tf, start.to_pydatetime() if hasattr(start, "to_pydatetime") else start)
        for s in chunk:
            old, new = existing[s], fresh.get(s)
            if new is None and old is None:
                continue
            df = new if old is None else (old if new is None else
                                          pd.concat([old[old.index < new.index[0]], new]))
            df = df[df.index >= pd.Timestamp(floor)]
            cache.save(s, source, df)
        if progress:
            progress(min(b + batch, len(symbols)), len(symbols))


def stock_frames(cache: BarCache, symbol: str, tfs: list[str]) -> dict[str, pd.DataFrame]:
    """Build scanner timeframes from cached Alpaca bars, aligned to TradingView."""
    frames = {}
    loaded: dict[str, pd.DataFrame | None] = {}
    for tf in tfs:
        src = TF_SOURCE[tf]
        if src not in loaded:
            loaded[src] = cache.load(symbol, src)
        df = loaded[src]
        if df is None or df.empty:
            continue
        if tf in ("5m", "15m", "30m", "1h", "4h"):
            frames[tf] = session_resample(df, {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}[tf])
        elif tf == "1D":
            frames[tf] = df
        elif tf == "1W":
            frames[tf] = resample(df, "1W")
    return frames

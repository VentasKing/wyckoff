"""CLI.

  python -m wyckoff_scanner csv  path.csv --symbol AAPL --tv NASDAQ:AAPL --tf 1D
  python -m wyckoff_scanner coinbase --type SPOT --tfs 1D 4h 1h --limit 20
"""
from __future__ import annotations

import argparse
import sys

from .alerts import SeenStore, format_alert, tap3_alerts
from .timeframes import scan_frames


def main(argv=None):
    p = argparse.ArgumentParser(prog="wyckoff_scanner")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("csv", help="scan one CSV of OHLC bars")
    c.add_argument("path")
    c.add_argument("--symbol", required=True)
    c.add_argument("--tv", help="TradingView symbol, e.g. NASDAQ:AAPL")
    c.add_argument("--tf", default="1D")

    cb = sub.add_parser("coinbase", help="scan Coinbase products")
    cb.add_argument("--type", default="SPOT", choices=["SPOT", "FUTURE"])
    cb.add_argument("--expiry", choices=["PERPETUAL", "EXPIRING"])
    cb.add_argument("--quote", default="USD", help="spot quote currency filter")
    cb.add_argument("--tfs", nargs="+", default=["1h", "4h", "1D", "1W"])
    cb.add_argument("--bars", type=int, default=400)
    cb.add_argument("--limit", type=int, help="only first N products (testing)")
    cb.add_argument("--seen", default=".seen_alerts.json")
    cb.add_argument("--all", action="store_true", help="show all tap 3s, not just new ones")

    st = sub.add_parser("stocks", help="scan US stocks via Alpaca (free tier)")
    st.add_argument("--tfs", nargs="+", default=["1h", "4h", "1D", "1W"])
    st.add_argument("--symbols", nargs="+", help="only these tickers (default: all NASDAQ/NYSE/AMEX/ARCA/BATS)")
    st.add_argument("--limit", type=int, help="only first N tickers (testing)")
    st.add_argument("--no-etfs", action="store_true")
    st.add_argument("--feed", default="sip", choices=["sip", "iex"],
                    help="sip = all exchanges, 15-min delayed on free plan; iex = real-time, IEX only")
    st.add_argument("--cache", default=".bar_cache")
    st.add_argument("--no-update", action="store_true", help="scan cached bars without downloading")
    st.add_argument("--seen", default=".seen_alerts.json")
    st.add_argument("--all", action="store_true", help="show all tap 3s, not just new ones")

    a = p.parse_args(argv)

    if a.cmd == "stocks":
        return run_stocks(a)

    if a.cmd == "csv":
        from .data import load_csv
        df = load_csv(a.path)
        ranges = scan_frames({a.tf: df})
        for r in ranges:
            print(f"{r.side:12s} {r.tf:4s} climax={r.climax_price:.6g} reaction={r.reaction_price:.6g} "
                  f"status={r.status} flags={r.red_flags}")
        for al in tap3_alerts(a.symbol, a.tv or a.symbol, ranges):
            print(format_alert(al))
        return

    from .data import coinbase_candles, coinbase_products, coinbase_tv_symbol
    prods = coinbase_products(a.type, a.expiry)
    if a.type == "SPOT":
        prods = [x for x in prods if x.get("quote_currency_id") == a.quote]
    if a.limit:
        prods = prods[: a.limit]
    seen = SeenStore(a.seen)
    for prod in prods:
        pid = prod["product_id"]
        try:
            frames = {tf: coinbase_candles(pid, tf, a.bars) for tf in a.tfs}
        except Exception as e:  # noqa: BLE001
            print(f"skip {pid}: {e}", file=sys.stderr)
            continue
        ranges = scan_frames(frames)
        alerts = tap3_alerts(pid, coinbase_tv_symbol(prod), ranges)
        for al in (alerts if a.all else seen.new(alerts)):
            print(format_alert(al))
    seen.save()


def run_stocks(a, client=None):
    from .alpaca import TF_SOURCE, AlpacaClient, BarCache, stock_frames, update_cache

    client = client or AlpacaClient(feed=a.feed)
    if not client.key:
        sys.exit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (free Alpaca paper account).")
    cache = BarCache(a.cache)

    universe = client.assets(include_etfs=not a.no_etfs)
    if a.symbols:
        want = {s.upper() for s in a.symbols}
        universe = [u for u in universe if u["symbol"] in want]
    if a.limit:
        universe = universe[: a.limit]
    syms = [u["symbol"] for u in universe]
    print(f"{len(syms)} tickers, timeframes {a.tfs}", file=sys.stderr)

    if not a.no_update:
        for src in sorted({TF_SOURCE[t] for t in a.tfs}):
            def prog(done, total, src=src):
                print(f"\r  downloading {src}: {done}/{total}", end="", file=sys.stderr)
            update_cache(client, cache, syms, src, progress=prog)
            print(file=sys.stderr)

    seen = SeenStore(a.seen)
    count = 0
    for u in universe:
        frames = stock_frames(cache, u["symbol"], a.tfs)
        if not frames:
            continue
        alerts = tap3_alerts(u["symbol"], u["tv"], scan_frames(frames))
        for al in (alerts if a.all else seen.new(alerts)):
            print(format_alert(al))
            count += 1
    seen.save()
    print(f"{count} tap-3 alert(s)", file=sys.stderr)
    return count


if __name__ == "__main__":
    main()

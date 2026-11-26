# Wyckoff tap scanner — rules & internals

Detects Wyckoff accumulation / distribution ranges and alerts when a **third tap**
prints (Model 1, Model 2 extreme liquidity, Model 2 extreme demand/supply), with a
TradingView link that opens on the range's timeframe.

```
pip install pandas numpy pytest
python -m pytest tests                     # 13 tests on synthetic setups
python -m wyckoff_scanner csv bars.csv --symbol AAPL --tv NASDAQ:AAPL --tf 1D

# US stocks (free Alpaca paper account keys)
export APCA_API_KEY_ID=...  APCA_API_SECRET_KEY=...
python -m wyckoff_scanner stocks --symbols AAPL MSFT NVDA --tfs 1h 4h 1D 1W
python -m wyckoff_scanner stocks --tfs 4h 1D 1W            # whole market
python -m wyckoff_scanner coinbase --type SPOT --tfs 1h 4h 1D 1W --limit 20
python -m wyckoff_scanner coinbase --type FUTURE --expiry PERPETUAL --tfs 4h 1D
```
CSV needs columns `time, open, high, low, close` (volume optional).
The Coinbase command remembers alerts in `.seen_alerts.json` so reruns only show
new third taps (`--all` shows everything).

## Data
- **Stocks – Alpaca free tier.** Universe = active, tradable NASDAQ / NYSE / NYSE American /
  NYSE Arca / Cboe listings (OTC, warrants, units and rights excluded). TradingView prefixes:
  NASDAQ→`NASDAQ:`, NYSE→`NYSE:`, NYSE American & Arca→`AMEX:`, Cboe BZX→`CBOE:`.
  Uses the all-exchange SIP feed, which is free when 15 minutes delayed (`--feed iex` for
  real-time IEX-only prices). Bars are cached in `.bar_cache/`; reruns only download new bars.
  Intraday bars are cut to regular hours and rebuilt from the 09:30 open so 1h/4h candles
  match TradingView's default stock charts: 5m/15m from 5-min bars, 30m/1h/4h from 30-min bars,
  1D native, 1W from daily.
- **Crypto – Coinbase public API** (free, no key). Universe = USD spot pairs ranked by 24h dollar volume (top N or all), stablecoins excluded. Each native granularity is downloaded once per coin (4h built from 1h, 1W from 1D).

First full-market download is the slow part at 200 requests/min: roughly 5–10 min for
daily, 20–30 min each for 30-min and 5-min history. After that, updates take a few minutes.

## Rules as implemented (accumulation; distribution is the exact mirror)

| Step | Rule |
|---|---|
| Market structure | **Two-candle rule**: two candles where the 2nd trades beyond the 1st (lower low going down, higher high going up). Inside candles don't count. Optional `require_color` also requires bearish/bullish closes. |
| Tap 1 – SC | A swing low that makes a new low, with valid two-candle structure into it, and **closes below every close of the previous `trend_lookback` candles** (default 50; the climax close is the lower close of the climax candle and the one before it): tap 1 ends the trend, so in an uptrend ranges are drawn from the top down and in a downtrend from the bottom up. |
| AR | Highest high after the SC before price trades into the EQ. Should **break structure** (exceed the latest swing high before the SC on the range's own timeframe). If it doesn't, the range is kept with red flag `ar_no_break_of_structure` (`require_bos=True` drops it). Needs two candles up into it and two down out of it. |
| Range confirmed | First candle that trades into the **EQ** (50% of SC→AR). |
| Tap 2 | Trades below range low. Invalid if any candle **closes** beyond `low − k·range` (k = 0.25; 0.30 on 1W/1M). Completes only on enough displacement by a candle **after** the one that made the tap-2 extreme: `either` (default) = quarter or two_closes, whichever comes first; `quarter` = close past the midpoint of range low and EQ; `two_closes` = two consecutive closes back inside; `close` = one close back inside. Until then further pushes extend tap 2. Wick-only vs closed-outside is recorded. |
| Tap 3 – Model 1 | Trades below the tap-2 low. Invalid if a candle closes beyond `tap2 low − k·range`. |
| Tap 3 – Model 2 liquidity | Sweeps the **extreme liquidity**: the first two-candle structure low (two bearish candles trading lower, then two trading higher) formed between the tap-2 extreme and the high between taps 2 and 3, taken from the **highest timeframe** (the range's own, then lower ones) that shows such structure. Swept before tap 2 completes = mitigated, next one used. |
| Tap 3 – Model 2 demand | Returns into the **extreme demand**: the first order block (last bearish candle since the tap-2 extreme) whose move left a bullish FVG, from the **highest timeframe** that has one. The source candles (order block, FVG) are recorded and shown. Same mitigation rule. |
| Model 2 tap 3 | The first extreme level hit is a **potential** tap 3 (later level hits are noted on it, not new taps). It's **confirmed** when, on the level's timeframe, a candle closes beyond the last two-candle structure high before the tap's extreme. Price going past tap 2 turns it into a Model 1. |
| Tap 3 confirmation & stop | A tap 3 is *forming* until a later candle closes back past its level (Model 1: back inside the range; M2 liquidity: back above the swept level; M2 demand: back above the zone). Before that, further pushes move its extreme. Once *confirmed*, the extreme is the stop: any trade through it fails the tap. A failed Model 1 invalidates the range. **Model 2 taps are level-hit alerts only** (you judge the break of structure), and are retired if price goes on to make a Model 1. |
| Red flag | Range high traded through before tap 3 (`range_high_not_fresh`). |
| Order block at tap 3 (info only) | A Model 1 tap 3 whose extreme lands in a fresh order block — last opposite candle before a move that left a FVG, on any scanned timeframe, untouched between the FVG and the high before tap 3 — or within `tap3_zone_tolerance` × the range of one, gets a ◆ badge. Highest timeframe wins. It never filters a setup out. |
| Bojan low / high (info only) | Looks like a Wyckoff model on a lower timeframe, each wick end a tap. 3 or 4 consecutive candles, each with at least `bojan_min_wick` of its range as a lower wick (upper for a Bojan high), the deepest wick at least half its range. Model 1 shape: each wick lower and longer than the one before, the last deepest. Model 2 shape: lower and longer up to the deepest wick, then one shorter wick holding above it but below the first candle's low. No close below the first candle's low. Not in a cluster: none of the 5 candles either side has a low within half the deepest wick's length. Marked ◉ at tap 3 (deepest wick at the tap 3 extreme) or at the LPS/LPSY after tap 3 is confirmed, on the model's timeframe or up to `bojan_timeframes_below` below. |
| Rolled model | When a confirmed Model 1 tap 3 is traded through, that setup is invalid, but a new model starts from it: the old tap 2 becomes tap 1, the highest high between the old taps 2 and 3 is the AR, the old tap 3 becomes tap 2, and the new deviation is the potential tap 3. All the usual rules then apply. Flag: `rolled_from_earlier_model`. |
| Tap 3 spacing | Candles from tap 2 to tap 3 must be at least `min_tap3_spacing` (default 0.25) × the candles from tap 1 to tap 2. A deviation of tap 2 before then is tap 2 extending (tap 2 moves to the new extreme and must displace back in again); Model 2 level hits before then are ignored. |
| Opposite end deviated twice | A second, separate deviation of the range high before tap 3 (price must come back in between, by the tap-2 return rule) invalidates the setup: it works as the opposite model. Limit = `max_opposite_deviations` (default 1). |
| Invalidated | Close beyond the opposite end's deviation limit before tap 3. |
| Completed | Close above the range high after a tap 3. |
| Timeframe | Each TF is scanned; when the same range (same climax & reaction prices) appears on several TFs it's classified on the **highest**, and limits are judged on that TF's closes. |
| Superseded | A newer range on the same side that confirms **before** an older range's tap 2 replaces it (status `superseded`). Ranges already past tap 2 keep running. |
| Nesting | A range can form inside a larger one (e.g. the tap-2 low starting an inner range). It's kept with red flag `inside_larger_range` (`allow_nested=False` drops it). |

## Layout
```
wyckoff_scanner/
  structure.py   two-candle rule, swing detection
  ranges.py      range detection + tap state machine (Model 1 / Model 2)
  timeframes.py  resampling, multi-TF scan, highest-TF assignment
  alerts.py      tap-3 alerts (live/recent filters), TradingView links, seen-alert store
  notify.py      phone push via ntfy (tap opens the TradingView chart), no repeats
  chart.py       draws a range with SC, AR, EQ, limits, liquidity/demand and every tap
  levels.py      Model 2 extreme liquidity / demand across timeframes
  data.py        CSV + Coinbase public candles; stock loader is a stub
tests/           synthetic price paths for each scenario
```

## Known gaps / to decide
- Alpaca daily bars may include extended-hours trades; compare a few daily highs/lows with TradingView.
- TradingView symbols for Coinbase futures/perps fall back to the spot symbol.
- "Structure demand" on high timeframes isn't modelled yet (order block + FVG only).
- No expiry for stale ranges; no Model 2 invalidation rule yet.
- Not yet validated on real charts.

import pandas as pd
import pytest

from wyckoff_scanner.zones import find_zones, group_zones, resample_clock, scan_zones, zone_bojans


def frame(rows, freq="1D", start="2026-01-01"):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=pd.date_range(start, periods=len(rows), freq=freq, tz="UTC"))


UP = [(100, 103, 99, 102)] * 3


def test_structure_demand_is_the_whole_leg():
    # two bearish candles trading lower, then a big bullish candle leaving a gap above both
    rows = UP + [(102, 104, 97, 98), (98, 99, 94, 95),         # leg: 94 - 104
                 (95, 112, 95, 111), (111, 118, 106, 117),     # gap: 106 > 104
                 (117, 120, 114, 119)]
    z = [z for z in find_zones(frame(rows), "1D") if z.kind == "structure"]
    assert len(z) == 1 and (z[0].lo, z[0].hi) == (94, 104) and z[0].touch is None


def test_order_block_when_structure_turned_first():
    # down 2, then 2 candles up (new structure), then the displacement
    rows = UP + [(102, 104, 97, 98), (98, 99, 94, 95), (95, 98, 95, 97), (97, 99.5, 96, 99),
                 (99, 112, 97.5, 111), (111, 118, 106, 117), (117, 120, 114, 119)]
    zs = find_zones(frame(rows), "1D")
    assert [z.kind for z in zs] == ["ob"] and (zs[0].lo, zs[0].hi) == (96, 99.5)   # the candle before the big one


def test_first_touch_only_and_break():
    rows = UP + [(102, 104, 97, 98), (98, 99, 94, 95), (95, 112, 95, 111), (111, 118, 106, 117),
                 (117, 120, 114, 119), (119, 119, 103, 108),   # first touch
                 (108, 110, 100, 105), (105, 106, 90, 92)]     # closes below 94: broken
    z = [z for z in find_zones(frame(rows), "1D") if z.kind == "structure"][0]
    assert z.touch == 8 and z.broken == 10


def test_supply_is_the_mirror():
    rows = UP + [(102, 104, 97, 98), (98, 99, 94, 95), (95, 112, 95, 111), (111, 118, 106, 117),
                 (117, 120, 114, 119)]
    m = frame([(-o, -l, -h, -c) for o, h, l, c in rows])
    z = [z for z in find_zones(m, "1D", "supply") if z.kind == "structure"]
    assert (z[0].lo, z[0].hi) == (-104, -94)


def test_bojan_formed_and_forming_in_zone():
    base = UP + [(102, 104, 97, 98), (98, 99, 94, 95), (95, 112, 95, 111), (111, 118, 106, 117),
                 (117, 120, 114, 119), (119, 120, 110, 116)]
    bojan = [(116, 117, 103, 110), (110, 112, 101, 109), (109, 111, 98.5, 108)]   # wicks into 94-104
    df = frame(base + bojan + [(108, 115, 107, 114)] * 6)
    zs = scan_zones({"1D": df}, ["1D"], keep=20)
    assert len(zs) == 1 and [b["state"] for b in zs[0].bojans] == ["formed"]
    df2 = frame(base + bojan[:2])
    z2 = scan_zones({"1D": df2}, ["1D"], keep=20)[0]
    assert [b["state"] for b in z2.bojans] == ["forming"]


def test_grouping_overlaps_across_timeframes():
    from wyckoff_scanner.zones import Zone
    a = Zone("demand", "ob", "1W", 90, 100, 0, 0, 2)
    b = Zone("demand", "structure", "1D", 95, 98, 0, 1, 3)
    c = Zone("demand", "ob", "4h", 60, 62, 0, 0, 2)
    d = Zone("supply", "ob", "1D", 96, 99, 0, 0, 2)
    gs = group_zones([b, a, c, d])
    assert sorted([[z.tf for z in g] for g in gs]) == [["1D"], ["1W", "1D"], ["4h"]]
    # chained overlaps no longer merge: 1W 90-100 and 1D 99-120 overlap by 1 of 10
    e = Zone("demand", "ob", "1D", 99, 120, 0, 0, 2)
    assert len(group_zones([a, e])) == 2


def test_clock_candles():
    h = frame([(1, 2, 0, 1)] * 48, freq="1h")
    assert list(resample_clock(h, "18h").index.hour[:3]) == [0, 18, 0]
    assert len(resample_clock(h, "5h")) == 10


def test_eth_3w_order_block_is_the_candle_before_the_big_one():
    """ETH-USD 3W, Nov 2026: red Jun 1, green Jun 22, green Jul 13, big green Aug 3.
    The OB is Jul 13 (1748-1979), not the red Jun 1 candle; it closed through
    on Nov 16, so it isn't listed."""
    import json, pathlib
    rows = json.loads((pathlib.Path(__file__).parent / "data" / "eth_3w_2026-11-26.json").read_text())
    df = pd.DataFrame([r[1:] for r in rows], columns=["open", "high", "low", "close"],
                      index=pd.to_datetime([r[0] for r in rows], unit="s", utc=True))
    zs = [z for z in find_zones(df, "3W") if df.index[z.first] >= pd.Timestamp("2026-06-01", tz="UTC")]
    assert [(z.kind, z.lo, z.hi) for z in zs][:1] == [("ob", 1748.05, 1979.0)]
    assert zs[0].broken is not None
    assert not [z for z in scan_zones({"3W": df}, ["3W"]) if z.lo == 1505.0]


def test_zone_left_and_retouched_is_not_listed():
    rows = UP + [(102, 104, 97, 98), (98, 99, 94, 95), (95, 112, 95, 111), (111, 118, 106, 117),
                 (117, 120, 114, 119), (119, 119, 103, 108),   # first touch
                 (108, 116, 105, 115)] + [(115, 120, 110, 118)] * 8 + [(118, 118, 101, 103)]   # left, came back
    assert scan_zones({"1D": frame(rows)}, ["1D"], keep=3) == []
    assert scan_zones({"1D": frame(rows[:10])}, ["1D"], keep=3)      # recent first touch is listed


def test_qnt_18h_near_miss_counts_as_touch():
    """QNT-USD 18h: OB 141.35-143.62; a wick to 143.70 two candles later is the first touch."""
    rows = [(150, 152.5, 147.2, 149.5), (149.5, 149.8, 134.8, 136.5), (136.5, 140.7, 133.9, 138.9),
            (138.8, 143.7, 126.8, 142.7), (142.7, 144.1, 140.6, 143.2), (143.2, 144.0, 137.3, 142.8),
            (142.81, 143.62, 141.35, 143.47), (143.56, 172.9, 139.85, 161.74), (161.8, 162.8, 153.53, 153.89),
            (153.98, 156.63, 143.70, 146.87), (146.97, 148.5, 143.86, 146.97), (146.62, 147.1, 135.0, 143.04)]
    df = frame(rows, freq="18h")
    z = [z for z in find_zones(df, "18h", touch_tol=0.05) if z.lo == 141.35][0]
    assert z.touch == 9
    z0 = [z for z in find_zones(df, "18h") if z.lo == 141.35][0]
    assert z0.touch == 11


def test_imbalance_candle_must_be_double_the_zone():
    # leg 94-104 (10 high); displacement 95-112 is 17 high: < 2x -> no zone at ratio 2, zone at ratio 1.5
    rows = UP + [(102, 104, 97, 98), (98, 99, 94, 95), (95, 112, 95, 111), (111, 118, 106, 117), (117, 120, 114, 119)]
    assert [z for z in find_zones(frame(rows), "1D", min_ratio=2) if z.kind == "structure"] == []
    assert [z for z in find_zones(frame(rows), "1D", min_ratio=1.5) if z.kind == "structure"]
    big = rows[:5] + [(95, 125, 95, 124), (124, 130, 106, 129), (129, 131, 126, 130)]           # 30 high: >= 2x
    assert [z for z in find_zones(frame(big), "1D", min_ratio=2) if z.kind == "structure"]


def test_watch_list_untouched_within_distance_with_own_ratio():
    from wyckoff_scanner.zones import scan_watch
    # leg 94-104 (10), displacement 95-125 (30 = 3x), price drifts up and away, never back
    rows = UP + [(102, 104, 97, 98), (98, 99, 94, 95), (95, 125, 95, 124), (124, 130, 106, 129)] + \
        [(129, 131, 126, 130)] * 5
    df = frame(rows)
    w = scan_watch({"1D": df}, ["1D"], min_ratio=3, max_distance=30)
    assert [(z.kind, z.lo, z.hi, round(z.distance, 1)) for z in w if z.kind == "structure"] == [("structure", 94, 104, 20.0)]
    assert not [z for z in scan_watch({"1D": df}, ["1D"], min_ratio=3.5, max_distance=30) if z.kind == "structure"]
    assert not [z for z in scan_watch({"1D": df}, ["1D"], min_ratio=3, max_distance=10) if z.kind == "structure"]
    assert not [z for z in scan_watch({"1D": df}, ["1D"], min_ratio=3, max_distance=30, kinds=("ob",)) if z.kind == "structure"]

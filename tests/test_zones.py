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
                 (99, 112, 98.5, 111), (111, 118, 106, 117), (117, 120, 114, 119)]
    zs = find_zones(frame(rows), "1D")
    assert [z.kind for z in zs] == ["ob"] and (zs[0].lo, zs[0].hi) == (94, 99)


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


def test_clock_candles():
    h = frame([(1, 2, 0, 1)] * 48, freq="1h")
    assert list(resample_clock(h, "18h").index.hour[:3]) == [0, 18, 0]
    assert len(resample_clock(h, "5h")) == 10

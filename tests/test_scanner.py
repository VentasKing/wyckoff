import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

from synth import Path, base_range, mirror_prices
from wyckoff_scanner import detect_ranges, find_swings, scan_frames, tap3_alerts
from wyckoff_scanner.structure import has_two_candle_leg


def only(ranges, side="accumulation"):
    """The single outer (non-nested) range on `side`."""
    rs = [r for r in ranges if r.side == side and "inside_larger_range" not in r.red_flags]
    assert len(rs) == 1, [(r.side, r.climax_price, r.reaction_price) for r in ranges]
    return rs[0]


def models(r):
    return [e.model for e in r.tap3_events()]


# ---------------------------------------------------------------- structure
def test_two_candle_rule_rejects_inside_doji():
    p = Path(100)
    p.leg(110, 3)                    # up leg
    p.bar(110, 110.5, 107, 108)      # bearish, lower low
    p.bar(108, 109, 107.5, 108.2)    # inside candle: low NOT lower -> breaks the leg
    df = p.df()
    assert not has_two_candle_leg(df, 3, len(df) - 1, "down")
    p.bar(108.2, 108.4, 106, 106.5)  # trades below the inside candle -> valid pair
    assert has_two_candle_leg(p.df(), 3, len(p.rows) - 1, "down")


def test_swings_alternate():
    sw = find_swings(base_range().df())
    kinds = [s.kind for s in sw]
    assert all(a != b for a, b in zip(kinds, kinds[1:]))


# ---------------------------------------------------------------- ranges
def test_range_confirmed_at_eq():
    r = only(detect_ranges(base_range().df(), "1D"))
    assert r.climax_price == pytest.approx(99.7)
    assert r.reaction_price == pytest.approx(140.3)
    assert r.eq == pytest.approx(120.0)
    assert r.dev_limit == pytest.approx(99.7 - 0.25 * 40.6)
    assert r.status == "awaiting_tap2"


def test_ar_without_bos_is_flagged_not_dropped():
    p = Path(200)
    p.leg(120, 8).leg(150, 3).leg(100, 3)   # lower high at ~150.3
    p.leg(140, 4).leg(118, 3)               # AR 140.3 does NOT break 150.3
    r = only(detect_ranges(p.df(), "1D"))
    assert "ar_no_break_of_structure" in r.red_flags
    strict = detect_ranges(p.df(), "1D", require_bos=True)
    assert [r for r in strict if r.side == "accumulation"] == []


def test_aggressive_move_without_prior_structure():
    p = Path(200)
    p.leg(202, 10)                           # some history before the drop
    p.leg(100, 6)                            # straight drop, no swing highs before the SC
    p.leg(140, 4).leg(118, 3)
    r = only(detect_ranges(p.df(), "1D"))
    assert r.climax_price == pytest.approx(99.7)
    assert "ar_no_break_of_structure" in r.red_flags


def nested_path():
    """Like model1_path, but tap 2 CLOSES below the range low (within the limit),
    so it is a new low by closes and can start a range of its own."""
    p = base_range()
    p.bar(103, 103.5, 96, 97)           # tap 2 closes below 99.7 (limit 89.55)
    p.bar(97, 101, 96.5, 100.5)
    p.leg(115, 3).leg(109, 2).leg(127, 3)
    p.leg(98, 5)
    p.bar(98, 98.5, 93, 99)
    p.leg(145, 6)
    return p


def test_range_within_range_is_kept_and_flagged():
    rs = [r for r in detect_ranges(nested_path().df(), "1D") if r.side == "accumulation"]
    inner = [r for r in rs if "inside_larger_range" in r.red_flags]
    assert len(rs) >= 2 and inner
    assert inner[0].climax_price == pytest.approx(96)        # tap 2 low starts the inner range
    flat = detect_ranges(nested_path().df(), "1D", allow_nested=False)
    assert len([r for r in flat if r.side == "accumulation"]) == 1


def test_wick_only_tap2_does_not_start_a_range_by_closes():
    rs = [r for r in detect_ranges(model1_path().df(), "1D") if r.side == "accumulation"]
    assert [r.climax_price for r in rs] == [pytest.approx(99.7)]


def test_weekly_uses_030():
    r = only(detect_ranges(base_range().df(), "1W"))
    assert r.k == 0.30


def model1_path():
    p = base_range()
    p.bar(103, 103.5, 96, 102)        # tap 2: wick to 96, closes back inside
    p.leg(115, 3)                     # rally (creates bullish FVG; OB = tap-2 candle)
    p.leg(109, 2)                     # first two bearish candles -> liquidity ~108.7
    p.leg(127, 3)                     # two candles up -> liquidity confirmed
    p.leg(98, 5)                      # sweeps liquidity, taps demand
    p.bar(98, 98.5, 93, 99)           # deviates tap-2 low (96) -> model 1
    p.leg(145, 6)                     # breaks range high -> completed
    return p


def test_model1_full_sequence():
    r = only(detect_ranges(model1_path().df(), "1D"))
    assert r.tap2_price == pytest.approx(96)
    assert r.tap2_closed_outside is False
    assert r.liquidity_level == pytest.approx(108.7)
    assert r.demand_zone == pytest.approx((96, 103.5))
    assert models(r) == ["model2_liquidity", "model1"]       # one Model 2 tap, then Model 1
    m2 = r.tap3_events()[0]
    assert "also tapped extreme demand" in m2.note and m2.failed   # superseded by Model 1
    assert r.model1_limit == pytest.approx(96 - 0.25 * 40.6)
    assert r.status == "completed"
    assert r.red_flags == []


def test_model2_liquidity_only():
    p = base_range()
    p.bar(103, 103.5, 96, 102)
    p.leg(115, 3).leg(109, 2).leg(127, 3)
    p.leg(107.5, 4)                   # sweeps 108.7 only, stays above demand zone top 103.5
    p.leg(135, 4)
    r = only(detect_ranges(p.df(), "1D"))
    assert models(r) == ["model2_liquidity"]
    assert r.status == "tap3"


def test_tap2_closing_beyond_limit_invalidates():
    p = base_range()
    p.bar(103, 103.5, 85, 88)         # closes below 89.55
    r = only(detect_ranges(p.df(), "1D"))
    assert r.status == "invalidated"
    assert "tap 2" in r.invalid_reason


def test_tap2_needs_a_later_candle_back_inside():
    """AMZN case: the tap-2 candle closes inside, the next closes back outside and
    keeps going -> still tap 2 (no tap 3) until a later candle closes inside."""
    p = base_range()
    p.bar(103, 103.5, 96, 102)        # wick below, closes inside
    p.bar(102, 102.5, 95, 97)         # closes back outside, new low
    p.bar(97, 98, 94, 95)             # lower again (within limit 89.55)
    r = only(detect_ranges(p.df(), "1D", tap2_return="close"))
    assert r.status == "tap2_active" and r.tap3_events() == []
    assert r.tap2_price == pytest.approx(94)
    p.bar(95, 101, 94.5, 100.5)       # closes inside, but it's after the extreme -> complete
    r = only(detect_ranges(p.df(), "1D", tap2_return="close"))
    assert r.status == "awaiting_tap3" and r.tap2_price == pytest.approx(94)


def test_tap2_extreme_candle_closing_inside_is_not_enough():
    p = base_range()
    p.bar(103, 103.5, 96, 102)
    r = only(detect_ranges(p.df(), "1D"))
    assert r.status == "tap2_active"


def model1_live():
    p = base_range()
    p.bar(103, 103.5, 96, 102).bar(102, 106, 101.5, 105)       # tap 2 + displacement
    p.leg(120, 3).leg(112, 2).leg(118, 2)
    p.bar(118, 118.5, 93, 95)          # tap 3 (model 1) starts: below 96
    return p


def test_model1_extends_before_confirmation():
    p = model1_live()
    p.bar(95, 95.5, 91, 92)            # lower low, still within limit 85.85, not confirmed yet
    r = only(detect_ranges(p.df(), "1D"))
    e = [e for e in r.tap3_events() if e.model == "model1"][0]
    assert not e.confirmed and e.extreme == pytest.approx(91) and r.status == "tap3"
    p.bar(92, 101, 91.5, 100.5)        # closes back inside the range -> confirmed
    r = only(detect_ranges(p.df(), "1D"))
    e = [e for e in r.tap3_events() if e.model == "model1"][0]
    assert e.confirmed and r.status == "tap3"


def test_model1_stop_hit_invalidates():
    """AAPL case: after tap 3 is confirmed, a wick through its extreme kills the model."""
    p = model1_live()
    p.bar(95, 101, 94.5, 100.5)        # confirmed, extreme 93
    p.leg(110, 2)
    p.bar(110, 110.5, 92.5, 99)        # wick through 93
    r = only(detect_ranges(p.df(), "1D"))
    assert r.status == "invalidated" and "tap 3 extreme" in r.invalid_reason
    from wyckoff_scanner import tap3_alerts
    assert tap3_alerts("X", "X", [r], live_only=True) == []


def test_model2_is_a_level_hit_alert_only():
    """Model 2 taps aren't auto-confirmed or stopped (the trader judges the BOS),
    but they're retired if price goes on to make a Model 1."""
    p = base_range()
    p.bar(103, 103.5, 96, 102).bar(102, 106, 101.5, 105)
    p.leg(115, 3).leg(109, 2).leg(127, 3)       # liquidity 108.7
    p.leg(107.5, 4)                              # sweeps it -> M2 liquidity hit
    p.leg(116, 2)
    p.bar(116, 116.5, 105, 106)                  # trades under the sweep low, above tap 2
    r = only(detect_ranges(p.df(), "1D"))
    m2 = [e for e in r.tap3_events() if e.model == "model2_liquidity"][0]
    assert not m2.confirmed and not m2.failed and r.status == "tap3"
    from wyckoff_scanner import tap3_alerts
    a = tap3_alerts("X", "X", [r], live_only=True)[0]
    assert a["tap3_state"] in ("potential", "confirmed") and a["level"] == pytest.approx(108.7)
    p.bar(106, 106.5, 95, 97)                    # now deviates tap 2 (96) -> Model 1
    r = only(detect_ranges(p.df(), "1D"))
    m2 = [e for e in r.tap3_events() if e.model == "model2_liquidity"][0]
    assert m2.failed and "model1" in [e.model for e in r.tap3_events()]


def test_tap2_close_outside_within_limit_is_ok():
    p = base_range()
    p.bar(103, 103.5, 94, 95)         # closes outside (below 99.7) but above 89.55
    p.bar(95, 102, 94.5, 101.5)       # back inside
    r = only(detect_ranges(p.df(), "1D", tap2_return="close"))
    assert r.status == "awaiting_tap3"
    assert r.tap2_closed_outside is True
    assert r.tap2_price == pytest.approx(94)


def test_model1_close_beyond_limit_invalidates():
    p = base_range()
    p.bar(103, 103.5, 96, 102)
    p.leg(115, 3)
    p.bar(115, 115.5, 80, 84)         # through tap-2 low, closes below 85.85
    r = only(detect_ranges(p.df(), "1D"))
    assert r.status == "invalidated"
    assert r.tap3_events() == []


def confirmed_only():
    """SC ~99.7 -> AR ~140.3 -> EQ touch, nothing after."""
    p = Path(200)
    p.leg(120, 8).leg(130, 3).leg(100, 3).leg(140, 4).leg(118, 3)
    return p


def test_opposite_end_red_flag_and_invalidation():
    p = confirmed_only()
    p.leg(142, 4)                     # trades above AR 140.3, closes below 150.45
    p.leg(103, 5)
    p.bar(103, 103.5, 96, 102)
    p.bar(102, 113, 101.5, 112)       # a later candle closes 25% back into the range
    r = only(detect_ranges(p.df(), "1D"))
    assert "range_high_not_fresh" in r.red_flags
    assert r.status == "awaiting_tap3"

    p2 = confirmed_only()
    p2.leg(155, 4)                    # closes above 150.45 -> invalid
    r2 = only(detect_ranges(p2.df(), "1D"))
    assert r2.status == "invalidated"


# ---------------------------------------------------------------- distribution
def test_distribution_is_mirror_of_accumulation():
    df = mirror_prices(model1_path().df())
    r = only(detect_ranges(df, "1D"), side="distribution")
    assert r.climax_price == pytest.approx(300 - 99.7)      # BC high
    assert r.reaction_price == pytest.approx(300 - 140.3)   # AR low
    assert r.tap2_price == pytest.approx(300 - 96)
    assert models(r) == ["model2_liquidity", "model1"]
    assert "also tapped extreme demand" in r.tap3_events()[0].note
    assert r.status == "completed"


# ---------------------------------------------------------------- MTF + alerts
def test_highest_timeframe_wins_and_alert_link():
    df = model1_path().df()
    ranges = scan_frames({"1h": df, "4h": df})
    acc = only(ranges)
    assert acc.tf == "4h"
    alerts = tap3_alerts("BTC-USD", "COINBASE:BTCUSD", [acc])
    assert len(alerts) == 2
    assert alerts[0]["chart"] == "https://www.tradingview.com/chart/?symbol=COINBASE%3ABTCUSD&interval=240"


def test_no_model2_taps_after_model1_starts():
    p = model1_live()
    p.bar(95, 101, 94.5, 100.5)
    p.leg(112, 2).leg(101, 2)          # would touch the old demand zone / liquidity again
    r = only(detect_ranges(p.df(), "1D"))
    models_after = [e.model for e in r.tap3_events()
                    if e.idx > [x for x in r.tap3_events() if x.model == "model1"][0].idx]
    assert models_after == []


def test_newer_range_supersedes_old_one_before_tap2():
    """MSFT case (shown as accumulation): an old range never got its tap 2; price
    trended to the opposite end and back, building a new climax near the old
    range low.  The new range should own the taps; the old one is superseded."""
    p = confirmed_only()                      # old range: SC 99.7, AR 140.3
    p.leg(139, 20).leg(120, 15).leg(132, 15)  # wanders near the opposite end for a while
    p.leg(101, 6)                             # long trend down -> new SC ~100.7 (above 99.7)
    p.leg(116, 3).leg(107, 2)                 # new AR ~116.3, EQ ~108.5 touched -> confirmed
    p.leg(103, 2)
    p.bar(103, 103.5, 98, 102).bar(102, 106, 101.5, 105)   # tap 2: sweeps new SC AND old low
    p.leg(112, 2).leg(105, 2).leg(110, 2)
    p.bar(110, 110.3, 96.5, 99).bar(99, 104, 98.5, 103)    # tap 3 model 1, confirmed
    rs = [r for r in detect_ranges(p.df(), "1D") if r.side == "accumulation"]
    old = [r for r in rs if r.climax_price == pytest.approx(99.7)][0]
    new = [r for r in rs if r.climax_price == pytest.approx(100.7)][0]
    assert old.status == "superseded" and old.tap3_events() == []
    assert new.status == "tap3" and "inside_larger_range" not in new.red_flags
    assert [e.model for e in new.tap3_events()] == ["model1"]
    from wyckoff_scanner import tap3_alerts
    live = tap3_alerts("MSFT", "NASDAQ:MSFT", rs, live_only=True)
    assert live and all(a["range"]["climax"] == pytest.approx(100.7) for a in live)


def test_range_past_tap2_is_not_superseded():
    rs = [r for r in detect_ranges(model1_path().df(), "1D") if r.side == "accumulation"]
    outer = only(rs)
    assert outer.status == "completed"       # inner range from the tap-2 low didn't replace it


def test_close_beyond_model1_limit_fails_the_forming_tap3():
    """QQQ case: tap 3 starts, then a candle closes far beyond the Model 1 limit.
    The range is invalidated and the tap must not survive as a valid tap 3."""
    p = model1_live()                     # model 1 forming at 93, limit 85.85
    p.bar(95, 95.5, 80, 81)               # closes way beyond the limit
    r = only(detect_ranges(p.df(), "1D"))
    assert r.status == "invalidated"
    assert all(e.failed for e in r.tap3_events())
    from wyckoff_scanner import tap3_alerts
    a = tap3_alerts("QQQ", "NASDAQ:QQQ", [r])           # history view
    assert a and all(x["tap3_state"] == "failed" for x in a)


def qqq_bad_break():
    """QQQ 1h case: wick below the range low, one weak close back inside, then the
    downtrend continues to just above the deviation limit, then a real rally."""
    p = base_range()                       # SC 99.7, AR 140.3, EQ 120, quarter level 109.85
    p.bar(103, 103.5, 96, 98)              # tap 2 starts
    p.bar(98, 101.5, 97.5, 101)            # weak close back inside (only to 101)
    p.leg(97, 2)
    p.bar(97, 97.5, 91, 95)                # continues to 91 (limit 89.55, closes above it)
    p.bar(95, 101, 94.5, 100.5)
    p.leg(116, 3)                          # real displacement above 109.85
    return p


def test_quarter_return_treats_bad_break_as_one_tap2():
    r = only(detect_ranges(qqq_bad_break().df(), "1D"))          # default: quarter
    assert r.tap2_price == pytest.approx(91)
    assert r.tap3_events() == [] and r.status == "awaiting_tap3"


def test_close_mode_would_call_it_a_tap3():
    r = only(detect_ranges(qqq_bad_break().df(), "1D", tap2_return="close"))
    assert r.tap2_price == pytest.approx(96)
    assert [e.model for e in r.tap3_events()] == ["model1"]


def test_two_closes_mode():
    r = only(detect_ranges(qqq_bad_break().df(), "1D", tap2_return="two_closes"))
    assert r.tap2_price == pytest.approx(91)        # only one close inside before the push
    p = base_range()
    p.bar(103, 103.5, 96, 98).bar(98, 101.5, 97.5, 101).bar(101, 102.5, 100, 102)
    r = only(detect_ranges(p.df(), "1D", tap2_return="two_closes"))
    assert r.status == "awaiting_tap3" and r.tap2_price == pytest.approx(96)


def test_either_mode_is_default_and_accepts_quarter_or_two_closes():
    r = only(detect_ranges(qqq_bad_break().df(), "1D"))
    assert r.tap2_return == "either" and r.tap2_price == pytest.approx(91)
    p = base_range()
    p.bar(103, 103.5, 96, 98).bar(98, 101.5, 97.5, 101).bar(101, 102.5, 100, 102)  # two closes
    assert only(detect_ranges(p.df(), "1D")).status == "awaiting_tap3"
    p = base_range()
    p.bar(103, 103.5, 96, 98).bar(98, 111, 97.5, 110.5)                           # one big close
    assert only(detect_ranges(p.df(), "1D")).status == "awaiting_tap3"


def mtf_case(pullback_on_1h: bool):
    """1h range; after tap 2 the 1h rally has no two-candle structure, but one
    15m candle group inside the rally does.  Then price drops into it."""
    from synth import expand
    p = base_range()
    p.bar(103, 103.5, 96, 102)                     # tap 2
    p.leg(110, 2)
    if pullback_on_1h:
        p.leg(116, 1).leg(113, 2).leg(122, 2)      # 1h structure too (low ~112.7)
        ov_bar = None
    else:
        p.leg(122, 2)
    rally_bar = p.idx() - 1                        # the 110 -> 116 hour
    p.leg(104.5, 4)                                # drop: sweeps the 15m level, above tap 2
    p.leg(118, 3)
    h1 = p.df(freq="1h")
    ov = {} if pullback_on_1h else {rally_bar: [
        (110, 113.55, 109.7, 113.5),               # up
        (113.5, 113.6, 111.95, 112),               # bearish, lower low
        (112, 112.05, 110.95, 111),                # bearish, lower low -> 15m liquidity 110.95
        (111, 116.3, 110.98, 116)]}                # up; next hour's higher high confirms
    m15 = expand(h1, 15, ov)
    return h1, m15


def test_extreme_liquidity_from_lower_timeframe():
    from wyckoff_scanner.timeframes import resample, scan_frames
    h1, m15 = mtf_case(pullback_on_1h=False)
    assert resample(m15, "1h").round(6).equals(h1.round(6))      # 15m really aggregates to the 1h
    rs = scan_frames({"15m": m15, "1h": h1})
    r = [x for x in rs if x.side == "accumulation" and x.tf == "1h"
         and "inside_larger_range" not in x.red_flags][0]
    assert r.liquidity_tf == "15m" and r.liquidity_level == pytest.approx(110.95)
    m2 = [e for e in r.tap3_events() if e.model == "model2_liquidity"]
    assert m2 and "(15m)" in m2[0].note
    # scanning the 1h alone can't see it
    alone = only(detect_ranges(h1, "1h"))
    assert "model2_liquidity" not in [e.model for e in alone.tap3_events()]


def test_highest_timeframe_with_structure_wins():
    from wyckoff_scanner.timeframes import scan_frames
    h1, m15 = mtf_case(pullback_on_1h=True)
    rs = scan_frames({"15m": m15, "1h": h1})
    r = [x for x in rs if x.side == "accumulation" and x.tf == "1h"
         and "inside_larger_range" not in x.red_flags][0]
    assert r.liquidity_tf == "1h" and r.liquidity_level == pytest.approx(112.7)


def test_structure_after_the_high_does_not_count():
    """The drop toward tap 3 makes bearish pairs; they must not become the level."""
    h1, _ = mtf_case(pullback_on_1h=False)
    r = only(detect_ranges(h1, "1h"))
    assert r.liquidity_level is None


def test_model2_potential_then_confirmed_by_structure_break():
    p = base_range()
    p.bar(103, 103.5, 96, 102).bar(102, 106, 101.5, 105)
    p.leg(115, 3).leg(109, 2).leg(127, 3)       # liquidity 108.7
    p.leg(118, 2).leg(121, 2)                   # two-candle bounce -> lower high ~121.3
    p.leg(107.5, 3)                             # sweeps 108.7 -> potential tap 3
    r = only(detect_ranges(p.df(), "1D"))
    m2 = r.tap3_events()[0]
    assert m2.model == "model2_liquidity" and not m2.confirmed
    p.leg(113, 2).leg(123, 2)                   # closes above the 121.3 lower high -> BOS
    r = only(detect_ranges(p.df(), "1D"))
    m2 = r.tap3_events()[0]
    assert m2.confirmed and m2.extreme == pytest.approx(107.2)


def test_optional_origin_rule():
    """Off by default. When switched on, a FVG that only appears well after tap 2
    (price drifted first) is not the extreme demand on that timeframe."""
    from wyckoff_scanner import levels
    p = base_range()
    p.bar(103, 103.5, 96, 102)                  # tap 2
    p.bar(102, 103, 101, 102.5).bar(102.5, 103.2, 101.8, 102).bar(102, 103.1, 101.6, 102.8)
    p.bar(102.8, 103.3, 102, 102.4)             # 4 candles of drift, no gap
    p.leg(120, 3)                               # impulse with FVGs, far from the origin
    p.leg(110, 3)
    r = only(detect_ranges(p.df(), "1D"))
    assert r.demand_zone is not None and r.demand_src["ob"] > 0      # default: accepted
    levels.REQUIRE_ORIGIN = True
    try:
        r = only(detect_ranges(p.df(), "1D"))
        assert r.demand_zone is None
    finally:
        levels.REQUIRE_ORIGIN = False


def test_zone_source_candles_are_recorded():
    r = only(detect_ranges(model1_path().df(), "1D"))
    df = model1_path().df()
    import pandas as pd
    ob_t = pd.Timestamp(r.demand_src["ob"], tz="UTC")
    assert df.loc[ob_t, "high"] == pytest.approx(r.demand_zone[1])         # OB candle = zone top
    assert pd.Timestamp(r.liquidity_src["low"], tz="UTC") in df.index


def test_pullback_in_uptrend_is_not_a_selling_climax():
    """SOL case: a few red candles inside an uptrend make a swing low, but price was
    lower earlier, so it can't be tap 1 of an accumulation."""
    p = Path(60)
    p.leg(90, 30)                         # strong uptrend
    p.leg(88, 2).leg(93, 3)               # small higher low ~87.7
    p.leg(86, 3)                          # 3-candle pullback -> swing low ~85.7 (below 87.7)
    p.leg(100, 3)                         # rally (AR-like)
    p.leg(92.5, 3)                        # back to the EQ
    acc = [r for r in detect_ranges(p.df(), "30m") if r.side == "accumulation"]
    assert acc == []
    off = [r for r in detect_ranges(p.df(), "30m", trend_lookback=0) if r.side == "accumulation"]
    assert off and off[0].climax_price == pytest.approx(85.7)     # what the old rule found


def test_buying_climax_at_end_of_uptrend_still_found():
    p = Path(60)
    p.leg(90, 30).leg(84, 3).leg(100, 3)  # top of the uptrend ~100.3
    p.leg(80, 4).leg(91, 3)               # AR low, back up to the EQ
    dist = [r for r in detect_ranges(p.df(), "30m") if r.side == "distribution"]
    assert dist and dist[0].climax_price == pytest.approx(100.3)


def test_trend_rule_uses_closes_not_wicks():
    """An old candle WICKED below the climax but closed above it: still a downtrend
    by closes, so the climax counts. The reverse (old close below) does not."""
    def build(old_close):
        p = Path(130)
        p.leg(125, 10)
        p.bar(125, 125.5, 95, old_close)        # old candle: long wick to 95
        p.leg(130, 4).leg(120, 6).leg(126, 3).leg(100, 4)   # downtrend into SC ~99.7
        p.leg(140, 4).leg(118, 3)               # AR, back to the EQ
        return [r for r in detect_ranges(p.df(), "1D") if r.side == "accumulation"
                and r.climax_price == pytest.approx(99.7)]
    assert build(old_close=124)                 # wick below, close far above -> valid
    assert not build(old_close=98)              # an older CLOSE below the climax -> not a new low by closes


def test_opposite_end_deviated_once_is_a_red_flag_only():
    p = confirmed_only()                     # SC 99.7, AR 140.3, return level 130.15
    p.leg(142, 4).leg(125, 4)                # one deviation of the range high, back in
    r = only(detect_ranges(p.df(), "4h"))
    assert r.status == "awaiting_tap2" and "range_high_not_fresh" in r.red_flags


def test_opposite_end_deviated_twice_invalidates():
    """BAT case: the range high was deviated twice, so the accumulation also reads
    as a distribution. Ignore it."""
    p = confirmed_only()
    p.leg(142, 4).leg(125, 4)                # deviation 1, back in
    p.leg(143, 4)                            # deviation 2
    p.leg(103, 6).bar(103, 103.5, 96, 102).bar(102, 113, 101.5, 112)   # a later tap 2
    r = only(detect_ranges(p.df(), "4h"))
    assert r.status == "invalidated" and "range high was deviated 2 times" in r.invalid_reason
    assert r.tap2_price is None              # nothing after it counts


def test_back_to_back_wicks_are_one_deviation():
    p = confirmed_only()
    p.leg(139, 3)
    p.bar(139, 141.5, 138, 139.5).bar(139.5, 141.8, 138.5, 139)        # two wicks, no return between
    p.leg(125, 4)
    r = only(detect_ranges(p.df(), "4h"))
    assert r.status == "awaiting_tap2"


def test_fresh_opposite_end_can_be_required():
    p = confirmed_only()
    p.leg(142, 4).leg(125, 4)
    r = only(detect_ranges(p.df(), "4h", opposite_max=0))
    assert r.status == "invalidated"


def test_distribution_mirror_uses_range_low_wording():
    p = confirmed_only()
    p.leg(142, 4).leg(125, 4).leg(143, 4)
    r = only(detect_ranges(mirror_prices(p.df()), "4h"), side="distribution")
    assert r.status == "invalidated" and "range low" in r.invalid_reason


def test_tap3_too_soon_after_tap2_extends_tap2():
    """SAND case: a new low a few candles after tap 2 is tap 2 extending, not a tap 3."""
    p = model1_live()
    r0 = only(detect_ranges(p.df(), "1D"))
    t2 = r0.tap2_idx
    r = only(detect_ranges(p.df(), "1D", tap3_spacing=1.0))   # tap 3 would need tap1->tap2 candles
    assert not [e for e in r.tap3_events() if e.model == "model1"]
    assert r.tap2_idx > t2 and any(e.kind == "tap2_extended" for e in r.events)
    assert r.status in ("tap2_active", "awaiting_tap3")
    r = only(detect_ranges(p.df(), "1D", tap3_spacing=0.0))   # off: same as before
    assert [e for e in r.tap3_events() if e.model == "model1"]


def test_stopped_model1_rolls_into_a_new_model():
    """BTC-USD 30m, Nov 24 2026: tap 3 (59,190) confirmed, then deviated. The old
    tap 2 (59,300) becomes tap 1, 60,619 the AR, 59,190 tap 2, the new low tap 3."""
    import json, pathlib
    import pandas as pd
    from wyckoff_scanner.timeframes import scan_frames
    rows = json.loads((pathlib.Path(__file__).parent / "data" / "btc_30m_2026-11-24.json").read_text())
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close"])
    df.index = pd.to_datetime(df.pop("t"), unit="s", utc=True)
    rs = scan_frames({"30m": df}, tap3_spacing=0.25, opposite_max=0)
    old = next(r for r in rs if r.climax_price == pytest.approx(59366.13))
    assert old.status == "invalidated"
    new = next(r for r in rs if r.climax_price == pytest.approx(59300.01))
    assert "rolled_from_earlier_model" in new.red_flags
    assert new.reaction_price == pytest.approx(60619.38) and new.tap2_price == pytest.approx(59189.67)
    m1 = [e for e in new.tap3_events() if e.model == "model1"]
    assert m1 and not m1[0].failed and new.status == "tap3"


def test_tap3_into_fresh_order_block():
    import pandas as pd
    from wyckoff_scanner.confluence import zone_at_tap3
    # bars: o, h, l, c
    rows = [(110, 111, 108, 109), (109, 109.5, 100, 101),      # bearish OB 100-109.5
            (101, 106, 101, 105), (105, 112, 104.9, 111),       # bullish move, FVG: low 110.? > OB high? no
            (111, 118, 110, 117),                               # FVG: low 110 > high of bar 2 (106)
            (117, 125, 116, 124), (124, 126, 118, 119),         # high before tap 3 = 126
            (119, 120, 108, 109), (109, 110, 103.2, 108)]       # tap 3 extreme 103.2 in the OB
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                      index=pd.date_range("2026-01-01", periods=len(rows), freq="30min", tz="UTC"))
    z = zone_at_tap3({"30m": df}, "accumulation", df.index[6], 103.2, tol=1.0)
    assert z and z["tf"] == "30m" and z["inside"] and (z["lo"], z["hi"]) == (100, 109.5)
    # touched before the leg into tap 3 -> not fresh
    rows2 = rows[:5] + [(117, 118, 109, 116)] + rows[5:]
    df2 = pd.DataFrame(rows2, columns=df.columns,
                       index=pd.date_range("2026-01-01", periods=len(rows2), freq="30min", tz="UTC"))
    assert zone_at_tap3({"30m": df2}, "accumulation", df2.index[7], 103.2, tol=1.0) is None
    # supply is the mirror
    m = pd.DataFrame({"open": -df["open"], "high": -df["low"], "low": -df["high"], "close": -df["close"]})
    zs = zone_at_tap3({"30m": m}, "distribution", m.index[6], -103.2, tol=1.0)
    assert zs and (zs["lo"], zs["hi"]) == (-109.5, -100)


def _bojan_df(rows, freq="1h"):
    import pandas as pd
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=pd.date_range("2026-01-01", periods=len(rows), freq=freq, tz="UTC"))


def test_bojan_low_shapes_from_the_guide():
    import numpy as np
    from wyckoff_scanner.bojan import find_bojans
    # guide, top-left: middle wick lowest, closes hold
    df = _bojan_df([(120, 121, 100, 111), (111, 115, 94, 112), (112, 117, 97, 113)])
    a = [df[k].to_numpy(float) for k in ("open", "high", "low", "close")]
    assert find_bojans(*a) == [(0, 2, 1)]
    # top-middle: each wick lower
    df = _bojan_df([(120, 121, 100, 109), (109, 114, 97, 112), (112, 118, 94, 113)])
    assert find_bojans(*[df[k].to_numpy(float) for k in ("open", "high", "low", "close")]) == [(0, 2, 2)]
    # no sweep, or a close below the first low -> nothing
    df = _bojan_df([(120, 121, 100, 110), (110, 115, 101, 112), (112, 117, 102, 113)])
    assert find_bojans(*[df[k].to_numpy(float) for k in ("open", "high", "low", "close")]) == []
    df = _bojan_df([(120, 121, 100, 111), (111, 115, 94, 99), (99, 117, 95, 113)])
    assert find_bojans(*[df[k].to_numpy(float) for k in ("open", "high", "low", "close")]) == []


def test_bojan_at_tap3_and_lps():
    from wyckoff_scanner.bojan import bojans_for_tap3
    rows = [(130, 131, 125, 126)] * 3 + [(120, 121, 100, 111), (111, 115, 94, 112), (112, 121, 97, 120),
                                          (120, 125, 119, 124), (124, 126, 118, 125),
                                          (125, 126, 113, 121), (121, 123, 109, 122), (122, 124, 111, 123),
                                          (123, 130, 122, 129)]
    df = _bojan_df(rows)
    got = bojans_for_tap3({"1h": df}, "1h", "accumulation", df.index[4], df.index[5], 94, tol=1,
                          lps_after=df.index[6])
    assert [(b["where"], b["tf"], b["price"]) for b in got] == [("tap3", "1h", 94), ("lps", "1h", 109)]
    m = _bojan_df([(-o, -l, -h, -c) for o, h, l, c in rows])
    got = bojans_for_tap3({"1h": m}, "1h", "distribution", m.index[4], m.index[5], -94, tol=1,
                          lps_after=m.index[6])
    assert [(b["where"], b["price"]) for b in got] == [("tap3", -94), ("lps", -109)]


def test_bojan_middle_wick_and_clusters():
    from wyckoff_scanner.bojan import find_bojans
    arr = lambda df: [df[k].to_numpy(float) for k in ("open", "high", "low", "close")]
    # middle wick shorter than the first -> not a Bojan
    df = _bojan_df([(120, 121, 100, 112), (112, 115, 97, 108), (108, 117, 94, 113)])
    assert find_bojans(*arr(df)) == []
    # PEPE-like: the same three candles inside a cluster of lows at that level -> not a Bojan
    good = [(120, 121, 100, 109), (109, 114, 97, 112), (112, 118, 94, 113)]
    flat = [(118, 119, 95, 117)] * 3
    assert find_bojans(*arr(_bojan_df(flat + good + flat))) == []
    assert find_bojans(*arr(_bojan_df([(130, 131, 125, 126)] * 3 + good + [(113, 125, 112, 124)] * 3))) == [(3, 5, 5)]

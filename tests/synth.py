"""Tiny DSL for building synthetic OHLC paths."""
import pandas as pd


class Path:
    def __init__(self, start: float, wick: float = 0.3):
        self.rows = []
        self.last = start
        self.wick = wick

    def leg(self, to: float, n: int):
        step = (to - self.last) / n
        for _ in range(n):
            o, c = self.last, self.last + step
            self.rows.append((o, max(o, c) + self.wick, min(o, c) - self.wick, c))
            self.last = c
        return self

    def bar(self, o, h, l, c):
        self.rows.append((o, h, l, c))
        self.last = c
        return self

    def idx(self):
        """index of the most recently added bar"""
        return len(self.rows) - 1

    def df(self, freq="1D"):
        idx = pd.date_range("2024-01-01", periods=len(self.rows), freq=freq, tz="UTC")
        return pd.DataFrame(self.rows, columns=["open", "high", "low", "close"], index=idx)


def mirror_prices(df: pd.DataFrame, pivot: float = 300.0) -> pd.DataFrame:
    return pd.DataFrame({"open": pivot - df["open"], "high": pivot - df["low"],
                         "low": pivot - df["high"], "close": pivot - df["close"]}, index=df.index)


def base_range():
    """Down move -> SC ~99.7 -> AR ~140.3 (breaks 130.3 lower high) -> EQ retrace."""
    p = Path(200)
    p.leg(120, 8).leg(130, 3).leg(100, 3)   # lower high at ~130.3, SC at ~99.7
    p.leg(140, 4)                           # AR ~140.3
    p.leg(118, 3)                           # EQ = 120 -> confirmed
    p.leg(126, 2).leg(103, 3)
    return p


def expand(df_1h: pd.DataFrame, minutes: int = 15, overrides: dict | None = None,
           wick: float = 0.05) -> pd.DataFrame:
    """Split each bar into linear sub-bars (same open/high/low/close when
    re-aggregated).  overrides: {bar_index: [(o, h, l, c), ...]} custom sub-bars."""
    k = int(pd.Timedelta(df_1h.index[1] - df_1h.index[0]) / pd.Timedelta(minutes=minutes))
    rows, idx = [], []
    for n, (t, b) in enumerate(df_1h.iterrows()):
        if overrides and n in overrides:
            subs = overrides[n]
        else:
            pts = [b.open + (b.close - b.open) * q / k for q in range(k + 1)]
            subs = [[pts[q], max(pts[q], pts[q + 1]) + wick, min(pts[q], pts[q + 1]) - wick, pts[q + 1]]
                    for q in range(k)]
            if b.close >= b.open:
                subs[0][2], subs[-1][1] = b.low, b.high
            else:
                subs[0][1], subs[-1][2] = b.high, b.low
        for q, sb in enumerate(subs):
            rows.append(tuple(sb))
            idx.append(t + pd.Timedelta(minutes=minutes * q))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=pd.DatetimeIndex(idx))

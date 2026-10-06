"""Η στρατηγική, ίδια με το tradingview/strategy.pine.

Σήματα στο κλείσιμο ημέρας t, είσοδος στο άνοιγμα της t+1 (όπως το TradingView).
Stop και στόχος υπολογίζονται από το κλείσιμο του σήματος με βάση το ATR.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd


@dataclass
class Params:
    ema_fast: int = 50
    ema_slow: int = 200
    use_pullback: bool = True
    rsi_len: int = 14
    rsi_level: int = 40
    use_breakout: bool = True
    brk_len: int = 20
    vol_mult: float = 1.2
    atr_len: int = 14
    sl_atr: float = 2.0
    tp_atr: float = 3.0
    max_bars: int = 20
    trend_exit: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def _rma(s: pd.Series, n: int) -> pd.Series:
    """Κινητός μέσος Wilder (ta.rma του Pine)."""
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def signals(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Υπολογίζει δείκτες και σήματα. Το df έχει Open, High, Low, Close, Volume."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"].fillna(0)
    out = pd.DataFrame(index=df.index)
    out["close"] = c
    out["ema_f"] = _ema(c, p.ema_fast)
    out["ema_s"] = _ema(c, p.ema_slow)

    d = c.diff()
    up, dn = _rma(d.clip(lower=0), p.rsi_len), _rma(-d.clip(upper=0), p.rsi_len)
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    out["rsi"] = rsi.where(dn != 0, 100.0)

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out["atr"] = _rma(tr, p.atr_len)

    vol_avg = v.rolling(20).mean()
    has_vol = vol_avg > 0
    vol_ok = ~has_vol | (v > p.vol_mult * vol_avg)
    hh = h.rolling(p.brk_len).max().shift(1)

    up_trend = (c > out["ema_s"]) & (out["ema_f"] > out["ema_s"])
    cross_up = (out["rsi"] > p.rsi_level) & (out["rsi"].shift() <= p.rsi_level)
    pull = p.use_pullback & up_trend & cross_up
    brk = p.use_breakout & up_trend & (c > hh) & vol_ok

    out["long_sig"] = (pull | brk).fillna(False) & out["atr"].notna()
    out["kind"] = np.where(brk, "breakout", np.where(pull, "pullback", ""))
    out["stop"] = c - p.sl_atr * out["atr"]
    out["tp"] = c + p.tp_atr * out["atr"]
    out["trend_broken"] = c < out["ema_s"]
    out["mom63"] = c / c.shift(63) - 1          # για κατάταξη όταν τα σήματα είναι περισσότερα από τις θέσεις
    return out


def market_ok(market_df: pd.DataFrame, p: Params) -> pd.Series:
    """Φίλτρο αγοράς: ο δείκτης είναι πάνω από τον αργό EMA."""
    c = market_df["Close"]
    return (c > _ema(c, p.ema_slow)).fillna(False)

"""Τεχνικοί δείκτες, σήματα, πιθανότητες και ανάλυση ειδήσεων.

Όλες οι συναρτήσεις εδώ είναι "καθαρές": παίρνουν DataFrame/λίστες και
επιστρέφουν αποτελέσματα, χωρίς δικτυακές κλήσεις. Έτσι ελέγχονται εύκολα.

Το DataFrame τιμών έχει στήλες Open, High, Low, Close, Volume και
χρονολογικό index (όπως το επιστρέφει το yfinance).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Βασικοί δείκτες
# ---------------------------------------------------------------------------


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = _wilder(d.clip(lower=0), n)
    down = _wilder(-d.clip(upper=0), n)
    rs = up / down.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(down != 0, 100.0).where(up.notna())


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, n=20, k=2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    width = (upper - lower)
    pctb = (close - lower) / width.replace(0, np.nan)
    return mid, upper, lower, pctb, width / mid


def stochastic(df: pd.DataFrame, n=14, smooth=3):
    ll = df["Low"].rolling(n, min_periods=n).min()
    hh = df["High"].rolling(n, min_periods=n).max()
    raw = 100 * (df["Close"] - ll) / (hh - ll).replace(0, np.nan)
    k = raw.rolling(smooth).mean()
    d = k.rolling(smooth).mean()
    return k, d


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["Close"].shift()
    return pd.concat(
        [df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, n=14) -> pd.Series:
    return _wilder(true_range(df), n)


def adx(df: pd.DataFrame, n=14):
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    a = atr(df, n).replace(0, np.nan)
    plus_di = 100 * _wilder(plus_dm, n) / a
    minus_di = 100 * _wilder(minus_dm, n) / a
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder(dx, n), plus_di, minus_di


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def mfi(df: pd.DataFrame, n=14) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    flow = tp * df["Volume"]
    pos = flow.where(tp > tp.shift(), 0.0).rolling(n, min_periods=n).sum()
    neg = flow.where(tp < tp.shift(), 0.0).rolling(n, min_periods=n).sum()
    ratio = pos / neg.replace(0, np.nan)
    return (100 - 100 / (1 + ratio)).where(neg != 0, 100.0).where(pos.notna())


def cci(df: pd.DataFrame, n=20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    m = tp.rolling(n, min_periods=n).mean()
    md = tp.rolling(n, min_periods=n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - m) / (0.015 * md.replace(0, np.nan))


def williams_r(df: pd.DataFrame, n=14) -> pd.Series:
    hh = df["High"].rolling(n, min_periods=n).max()
    ll = df["Low"].rolling(n, min_periods=n).min()
    return -100 * (hh - df["Close"]) / (hh - ll).replace(0, np.nan)


def ichimoku(df: pd.DataFrame):
    def mid(n):
        return (df["High"].rolling(n).max() + df["Low"].rolling(n).min()) / 2

    tenkan, kijun = mid(9), mid(26)
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = mid(52).shift(26)
    return tenkan, kijun, span_a, span_b


def psar(df: pd.DataFrame, step=0.02, max_af=0.2):
    """Parabolic SAR. Επιστρέφει (sar, ανοδική_τάση: bool)."""
    h, l = df["High"].to_numpy(float), df["Low"].to_numpy(float)
    n = len(h)
    sar_out = np.full(n, np.nan)
    bull_out = np.zeros(n, dtype=bool)
    if n < 3:
        return pd.Series(sar_out, index=df.index), pd.Series(bull_out, index=df.index)
    bull = h[1] >= h[0]
    af = step
    ep = h[0] if bull else l[0]
    sar = l[0] if bull else h[0]
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if bull:
            sar = min(sar, l[i - 1], l[max(i - 2, 0)])
            if l[i] < sar:
                bull, sar, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, max_af)
        else:
            sar = max(sar, h[i - 1], h[max(i - 2, 0)])
            if h[i] > sar:
                bull, sar, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, max_af)
        sar_out[i] = sar
        bull_out[i] = bull
    return pd.Series(sar_out, index=df.index), pd.Series(bull_out, index=df.index)


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].replace(0, np.nan).fillna(0)
    cum_v = vol.cumsum().replace(0, np.nan)
    return (tp * vol).cumsum() / cum_v


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Προσθέτει όλους τους ημερήσιους δείκτες ως στήλες."""
    out = df.copy()
    c = out["Close"]
    out["SMA20"], out["SMA50"], out["SMA200"] = sma(c, 20), sma(c, 50), sma(c, 200)
    out["EMA20"] = ema(c, 20)
    out["RSI"] = rsi(c)
    out["MACD"], out["MACD_signal"], out["MACD_hist"] = macd(c)
    out["BB_mid"], out["BB_up"], out["BB_low"], out["BB_pctb"], out["BB_width"] = bollinger(c)
    out["STOCH_K"], out["STOCH_D"] = stochastic(out)
    out["ATR"] = atr(out)
    out["ADX"], out["PDI"], out["MDI"] = adx(out)
    out["OBV"] = obv(out)
    out["MFI"] = mfi(out)
    out["CCI"] = cci(out)
    out["WILLR"] = williams_r(out)
    out["TENKAN"], out["KIJUN"], out["SPAN_A"], out["SPAN_B"] = ichimoku(out)
    out["PSAR"], out["PSAR_BULL"] = psar(out)
    out["VOL20"] = out["Volume"].rolling(20, min_periods=20).mean()
    return out


# ---------------------------------------------------------------------------
# Μοτίβα κεριών, στηρίξεις / αντιστάσεις
# ---------------------------------------------------------------------------


def candle_patterns(df: pd.DataFrame) -> list[tuple[str, float]]:
    """Μοτίβα στο τελευταίο κερί. Επιστρέφει [(όνομα, κατεύθυνση -1..1)]."""
    if len(df) < 25:
        return []
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    i = len(df) - 1
    body = abs(c[i] - o[i])
    rng = max(h[i] - l[i], 1e-12)
    upper = h[i] - max(c[i], o[i])
    lower = min(c[i], o[i]) - l[i]
    prior_trend = c[i - 1] - c[i - 6]  # τάση των προηγούμενων 5 ημερών
    found: list[tuple[str, float]] = []

    if body <= 0.1 * rng:
        found.append(("Doji (αναποφασιστικότητα)", 0.0))
    if lower >= 2 * body and upper <= 0.5 * body + 0.1 * rng and prior_trend < 0 and body > 0:
        found.append(("Hammer μετά από πτώση (πιθανή αντιστροφή προς τα πάνω)", 1.0))
    if upper >= 2 * body and lower <= 0.5 * body + 0.1 * rng and prior_trend > 0 and body > 0:
        found.append(("Shooting star μετά από άνοδο (πιθανή αντιστροφή προς τα κάτω)", -1.0))
    prev_red, prev_green = c[i - 1] < o[i - 1], c[i - 1] > o[i - 1]
    if prev_red and c[i] > o[i] and c[i] >= o[i - 1] and o[i] <= c[i - 1]:
        found.append(("Bullish engulfing (ανοδικό κερί «καταπίνει» το προηγούμενο)", 1.0))
    if prev_green and c[i] < o[i] and c[i] <= o[i - 1] and o[i] >= c[i - 1]:
        found.append(("Bearish engulfing (πτωτικό κερί «καταπίνει» το προηγούμενο)", -1.0))
    # Morning / evening star (3 κεριά)
    b2 = abs(c[i - 1] - o[i - 1])
    r2 = max(h[i - 1] - l[i - 1], 1e-12)
    if (c[i - 2] < o[i - 2] and b2 < 0.3 * r2 and c[i] > o[i]
            and c[i] > (o[i - 2] + c[i - 2]) / 2):
        found.append(("Morning star (μοτίβο ανοδικής αντιστροφής)", 1.0))
    if (c[i - 2] > o[i - 2] and b2 < 0.3 * r2 and c[i] < o[i]
            and c[i] < (o[i - 2] + c[i - 2]) / 2):
        found.append(("Evening star (μοτίβο πτωτικής αντιστροφής)", -1.0))
    return found


def pivot_points(df: pd.DataFrame) -> dict[str, float]:
    """Κλασικά pivot points από την τελευταία ολοκληρωμένη συνεδρίαση."""
    if len(df) < 2:
        return {}
    row = df.iloc[-2]
    p = (row["High"] + row["Low"] + row["Close"]) / 3
    r = row["High"] - row["Low"]
    return {"S2": p - r, "S1": 2 * p - row["High"], "P": p, "R1": 2 * p - row["Low"], "R2": p + r}


def swing_levels(df: pd.DataFrame, lookback=180, window=5, tol=0.015):
    """Στηρίξεις/αντιστάσεις από πρόσφατα τοπικά ελάχιστα/μέγιστα, ομαδοποιημένα."""
    d = df.tail(lookback)
    if len(d) < 2 * window + 2:
        return [], []
    highs, lows = d["High"].to_numpy(float), d["Low"].to_numpy(float)
    pts = []
    for i in range(window, len(d) - window):
        if highs[i] == highs[i - window:i + window + 1].max():
            pts.append(highs[i])
        if lows[i] == lows[i - window:i + window + 1].min():
            pts.append(lows[i])
    pts.sort()
    clusters: list[list[float]] = []
    for p in pts:
        if clusters and abs(p - np.mean(clusters[-1])) / p <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    levels = [(float(np.mean(cl)), len(cl)) for cl in clusters]
    price = float(d["Close"].iloc[-1])
    supports = sorted([lv for lv in levels if lv[0] < price], key=lambda x: -x[0])[:3]
    resistances = sorted([lv for lv in levels if lv[0] > price], key=lambda x: x[0])[:3]
    return supports, resistances


# ---------------------------------------------------------------------------
# Σήματα και συνολική τεχνική βαθμολογία
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    group: str
    name: str
    value: str
    verdict: float  # -1 (πτωτικό) ... +1 (ανοδικό)
    weight: float
    note: str


def _f(x, d=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _p(x, d=1):
    """Ποσοστό με πρόσημο σε ελληνική μορφή: +1,5%."""
    return f"{x * 100:+.{d}f}%".replace(".", ",")


def technical_signals(ind: pd.DataFrame) -> list[Signal]:
    """Ημερήσια σήματα από τον πίνακα του add_indicators()."""
    if len(ind) < 30:
        return []
    last, prev = ind.iloc[-1], ind.iloc[-2]
    c = last["Close"]
    sig: list[Signal] = []

    def ok(*vals):
        return all(v is not None and not pd.isna(v) for v in vals)

    # Τάση
    for col, label, w in (("SMA200", "Μακροπρόθεσμη τάση (SMA 200)", 1.5),
                          ("SMA50", "Μεσοπρόθεσμη τάση (SMA 50)", 1.0),
                          ("SMA20", "Βραχυπρόθεσμη τάση (SMA 20)", 0.75)):
        if ok(last[col]):
            diff = c / last[col] - 1
            v = 1.0 if diff > 0 else -1.0
            sig.append(Signal("Τάση", label, f"{_p(diff)} από τον μέσο", v, w,
                              "Η τιμή είναι πάνω από τον μέσο όρο" if v > 0 else
                              "Η τιμή είναι κάτω από τον μέσο όρο"))

    if ok(last["SMA50"], last["SMA200"]):
        golden = last["SMA50"] > last["SMA200"]
        recent = ind[["SMA50", "SMA200"]].dropna().tail(21)
        crossed = len(recent) > 1 and ((recent["SMA50"] > recent["SMA200"]).nunique() > 1)
        note = ("Golden cross: ο SMA50 είναι πάνω από τον SMA200" if golden else
                "Death cross: ο SMA50 είναι κάτω από τον SMA200")
        if crossed:
            note += " (η διασταύρωση έγινε τον τελευταίο μήνα)"
        sig.append(Signal("Τάση", "Διασταύρωση SMA50 / SMA200",
                          "Golden cross" if golden else "Death cross",
                          1.0 if golden else -1.0, 1.0, note))

    if ok(last["ADX"], last["PDI"], last["MDI"]):
        a = last["ADX"]
        if a < 20:
            v, note = 0.0, "Αδύναμη τάση: η μετοχή κινείται πλάγια"
        else:
            v = 1.0 if last["PDI"] > last["MDI"] else -1.0
            strength = "ισχυρή" if a >= 25 else "μέτρια"
            note = f"{strength.capitalize()} {'ανοδική' if v > 0 else 'πτωτική'} τάση (+DI {'>' if v > 0 else '<'} −DI)"
        sig.append(Signal("Τάση", "ADX (δύναμη τάσης)", _f(a, 1), v, 1.0, note))

    if ok(last["SPAN_A"], last["SPAN_B"]):
        top, bot = max(last["SPAN_A"], last["SPAN_B"]), min(last["SPAN_A"], last["SPAN_B"])
        if c > top:
            v, note = 1.0, "Η τιμή είναι πάνω από το σύννεφο Ichimoku"
        elif c < bot:
            v, note = -1.0, "Η τιμή είναι κάτω από το σύννεφο Ichimoku"
        else:
            v, note = 0.0, "Η τιμή είναι μέσα στο σύννεφο (ουδέτερη ζώνη)"
        sig.append(Signal("Τάση", "Ichimoku cloud", f"{_f(bot)} – {_f(top)}", v, 1.0, note))

    if ok(last["PSAR"]):
        bull = bool(last["PSAR_BULL"])
        sig.append(Signal("Τάση", "Parabolic SAR", _f(last["PSAR"]), 1.0 if bull else -1.0, 0.5,
                          "Το SAR είναι κάτω από την τιμή (ανοδικό)" if bull else
                          "Το SAR είναι πάνω από την τιμή (πτωτικό)"))

    # Ορμή
    if ok(last["MACD_hist"], prev["MACD_hist"]):
        h, hp = last["MACD_hist"], prev["MACD_hist"]
        if h > 0 and h >= hp:
            v, note = 1.0, "MACD πάνω από το σήμα και η διαφορά μεγαλώνει"
        elif h > 0:
            v, note = 0.5, "MACD πάνω από το σήμα, αλλά η ορμή εξασθενεί"
        elif h < 0 and h <= hp:
            v, note = -1.0, "MACD κάτω από το σήμα και η διαφορά μεγαλώνει"
        else:
            v, note = -0.5, "MACD κάτω από το σήμα, αλλά η πτωτική ορμή εξασθενεί"
        if (h > 0) != (hp > 0):
            note += " · νέα διασταύρωση σήμερα"
        sig.append(Signal("Ορμή", "MACD (12, 26, 9)", f"ιστόγραμμα {_f(h, 3)}", v, 1.0, note))

    if ok(last["RSI"]):
        r = last["RSI"]
        if r >= 70:
            v, note = -1.0, "Υπεραγορασμένη: αυξημένος κίνδυνος διόρθωσης"
        elif r <= 30:
            v, note = 1.0, "Υπερπουλημένη: συχνά ακολουθεί αναπήδηση"
        elif r >= 50:
            v, note = 0.5, "Θετική ορμή (πάνω από 50)"
        else:
            v, note = -0.5, "Αρνητική ορμή (κάτω από 50)"
        sig.append(Signal("Ορμή", "RSI (14)", _f(r, 1), v, 1.0, note))

    if ok(last["STOCH_K"], last["STOCH_D"]):
        k, d = last["STOCH_K"], last["STOCH_D"]
        if k >= 80:
            v, note = (-1.0, "Υπεραγορασμένη και το %K γυρίζει κάτω") if k < d else (-0.25, "Υπεραγορασμένη ζώνη")
        elif k <= 20:
            v, note = (1.0, "Υπερπουλημένη και το %K γυρίζει πάνω") if k > d else (0.25, "Υπερπουλημένη ζώνη")
        else:
            v, note = (0.25 if k > d else -0.25), ("%K πάνω από %D" if k > d else "%K κάτω από %D")
        sig.append(Signal("Ορμή", "Stochastic (14, 3, 3)", f"%K {_f(k, 0)} / %D {_f(d, 0)}", v, 0.5, note))

    if ok(last["CCI"]):
        x = last["CCI"]
        if x > 200:
            v, note = -0.5, "Ακραία υψηλό: πιθανή υπερέκταση"
        elif x > 100:
            v, note = 0.5, "Ισχυρή ανοδική ορμή"
        elif x < -200:
            v, note = 0.5, "Ακραία χαμηλό: πιθανή υπερπώληση"
        elif x < -100:
            v, note = -0.5, "Ισχυρή πτωτική ορμή"
        else:
            v, note = 0.0, "Ουδέτερη ζώνη"
        sig.append(Signal("Ορμή", "CCI (20)", _f(x, 0), v, 0.5, note))

    if ok(last["WILLR"]):
        w = last["WILLR"]
        v, note = ((-0.5, "Υπεραγορασμένη") if w > -20 else (0.5, "Υπερπουλημένη") if w < -80
                   else (0.0, "Ουδέτερη ζώνη"))
        sig.append(Signal("Ορμή", "Williams %R (14)", _f(w, 0), v, 0.25, note))

    # Μεταβλητότητα
    if ok(last["BB_pctb"]):
        b = last["BB_pctb"]
        if b > 1:
            v, note = -0.5, "Πάνω από την άνω ζώνη: υπερέκταση"
        elif b < 0:
            v, note = 0.5, "Κάτω από την κάτω ζώνη: υπερπώληση"
        else:
            v, note = 0.0, f"Μέσα στις ζώνες ({b:.0%} της απόστασης από κάτω προς πάνω)"
        squeeze = ind["BB_width"].dropna().tail(126)
        if len(squeeze) > 20 and last["BB_width"] <= squeeze.quantile(0.1):
            note += " · στενές ζώνες (squeeze): συχνά προηγείται μεγάλης κίνησης"
        sig.append(Signal("Μεταβλητότητα", "Bollinger Bands (20, 2)", f"%B {_f(b, 2)}", v, 0.5, note))

    # Όγκος
    if ok(last["VOL20"]) and last["VOL20"] > 0:
        ratio = last["Volume"] / last["VOL20"]
        up_day = c >= prev["Close"]
        if ratio >= 1.5:
            v = 0.5 if up_day else -0.5
            note = f"Όγκος {_f(ratio, 1)}× του μέσου σε {'ανοδική' if up_day else 'πτωτική'} μέρα"
        else:
            v, note = 0.0, f"Όγκος {_f(ratio, 1)}× του μέσου 20 ημερών"
        sig.append(Signal("Όγκος", "Όγκος συναλλαγών", f"{_f(ratio, 2)}×", v, 0.5, note))

    obv_s = ind["OBV"]
    if len(obv_s) > 25:
        obv_ma = obv_s.rolling(20).mean().iloc[-1]
        v = 1.0 if obv_s.iloc[-1] > obv_ma else -1.0
        sig.append(Signal("Όγκος", "On-Balance Volume", "πάνω από μέσο" if v > 0 else "κάτω από μέσο", v, 0.5,
                          "Ο όγκος συσσωρεύεται σε ανοδικές μέρες (αγοραστές)" if v > 0 else
                          "Ο όγκος συσσωρεύεται σε πτωτικές μέρες (πωλητές)"))

    if ok(last["MFI"]):
        m = last["MFI"]
        v, note = ((-1.0, "Υπεραγορασμένη με βάση τιμή και όγκο") if m >= 80 else
                   (1.0, "Υπερπουλημένη με βάση τιμή και όγκο") if m <= 20 else
                   (0.0, "Ουδέτερη ζώνη"))
        sig.append(Signal("Όγκος", "Money Flow Index (14)", _f(m, 0), v, 0.5, note))

    # Θέση στο ετήσιο εύρος
    yr = ind.tail(252)
    hi, lo = yr["High"].max(), yr["Low"].min()
    if hi > lo:
        pos = (c - lo) / (hi - lo)
        if c >= hi * 0.97:
            v, note = 0.5, "Κοντά στο υψηλό 52 εβδομάδων (ισχυρή σχετική δύναμη)"
        elif c <= lo * 1.03:
            v, note = -0.5, "Κοντά στο χαμηλό 52 εβδομάδων"
        else:
            v, note = 0.0, f"Στο {pos:.0%} του ετήσιου εύρους"
        sig.append(Signal("Εύρος", "Θέση στο εύρος 52 εβδομάδων", f"{_f(lo)} – {_f(hi)}", v, 0.25, note))

    for name, v in candle_patterns(ind):
        sig.append(Signal("Κεριά", "Μοτίβο κεριού", name.split(" (")[0], v, 0.5, name))

    return sig


def score_signals(signals: list[Signal]) -> float:
    """Σταθμισμένος μέσος όρος σημάτων στην κλίμακα -100 … +100."""
    w = sum(s.weight for s in signals)
    if w == 0:
        return 0.0
    return 100 * sum(s.verdict * s.weight for s in signals) / w


def score_label(score: float) -> str:
    if score >= 40:
        return "Ισχυρά ανοδική"
    if score >= 15:
        return "Ανοδική"
    if score > -15:
        return "Ουδέτερη"
    if score > -40:
        return "Πτωτική"
    return "Ισχυρά πτωτική"


def intraday_signals(intra: pd.DataFrame) -> list[Signal]:
    """Σήματα της τρέχουσας συνεδρίασης (από κεριά 1 λεπτού)."""
    if intra is None or len(intra) < 15:
        return []
    bars = intra.resample("5min").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna(subset=["Close"])
    c = float(intra["Close"].iloc[-1])
    out: list[Signal] = []
    vw = vwap(intra).iloc[-1]
    if not pd.isna(vw):
        d = c / vw - 1
        out.append(Signal("Σήμερα", "Τιμή έναντι VWAP", _p(d, 2), 1.0 if d > 0 else -1.0, 1.0,
                          "Πάνω από τη μέση σταθμισμένη τιμή της ημέρας (αγοραστές στον έλεγχο)" if d > 0 else
                          "Κάτω από τη μέση σταθμισμένη τιμή της ημέρας (πωλητές στον έλεγχο)"))
    if len(bars) >= 21:
        e9, e21 = ema(bars["Close"], 9).iloc[-1], ema(bars["Close"], 21).iloc[-1]
        v = 1.0 if e9 > e21 else -1.0
        out.append(Signal("Σήμερα", "EMA 9 / EMA 21 (5λεπτα)", "9 > 21" if v > 0 else "9 < 21", v, 1.0,
                          "Βραχυπρόθεσμη ανοδική ορμή" if v > 0 else "Βραχυπρόθεσμη πτωτική ορμή"))
    if len(bars) >= 15:
        r = rsi(bars["Close"]).iloc[-1]
        if not pd.isna(r):
            v = -1.0 if r >= 75 else 1.0 if r <= 25 else (0.5 if r >= 50 else -0.5)
            out.append(Signal("Σήμερα", "RSI 5λεπτου", _f(r, 0), v, 0.5,
                              "Υπεραγορασμένη μέσα στη μέρα" if r >= 75 else
                              "Υπερπουλημένη μέσα στη μέρα" if r <= 25 else
                              "Θετική ορμή" if r >= 50 else "Αρνητική ορμή"))
    first = float(intra["Open"].iloc[0])
    if first > 0:
        d = c / first - 1
        out.append(Signal("Σήμερα", "Κίνηση από το άνοιγμα", _p(d, 2), 0.5 if d > 0 else -0.5, 0.5,
                          "Πάνω από την τιμή ανοίγματος" if d > 0 else "Κάτω από την τιμή ανοίγματος"))
    return out


# ---------------------------------------------------------------------------
# Πιθανότητες
# ---------------------------------------------------------------------------

HORIZONS = {1: "Αύριο", 5: "Σε 1 εβδομάδα", 20: "Σε 1 μήνα"}
HORIZON_DAYS = {1: "1 συνεδρίαση", 5: "5 συνεδριάσεις", 20: "20 συνεδριάσεις"}
FEATURES = ["ret1", "ret5", "ret20", "d_sma20", "d_sma50", "d_sma200", "rsi", "macd_h",
            "bb_pctb", "atr_pct", "vol_ratio", "stoch_k", "adx", "di_diff", "mfi"]


def feature_frame(ind: pd.DataFrame) -> pd.DataFrame:
    c = ind["Close"]
    f = pd.DataFrame(index=ind.index)
    f["ret1"] = c.pct_change(1)
    f["ret5"] = c.pct_change(5)
    f["ret20"] = c.pct_change(20)
    f["d_sma20"] = c / ind["SMA20"] - 1
    f["d_sma50"] = c / ind["SMA50"] - 1
    f["d_sma200"] = c / ind["SMA200"] - 1
    f["rsi"] = ind["RSI"] / 100
    f["macd_h"] = ind["MACD_hist"] / c
    f["bb_pctb"] = ind["BB_pctb"].clip(-1, 2)
    f["atr_pct"] = ind["ATR"] / c
    f["vol_ratio"] = np.log((ind["Volume"] + 1) / (ind["VOL20"] + 1))
    f["stoch_k"] = ind["STOCH_K"] / 100
    f["adx"] = ind["ADX"] / 100
    f["di_diff"] = (ind["PDI"] - ind["MDI"]) / 100
    f["mfi"] = ind["MFI"] / 100
    return f.replace([np.inf, -np.inf], np.nan)


def _rsi_bucket(r):
    if pd.isna(r):
        return np.nan
    return 0 if r < 30 else 1 if r < 45 else 2 if r < 55 else 3 if r < 70 else 4


RSI_BUCKET_TEXT = {0: "RSI < 30", 1: "RSI 30–45", 2: "RSI 45–55", 3: "RSI 55–70", 4: "RSI > 70"}


def regime_frame(ind: pd.DataFrame) -> pd.DataFrame:
    c = ind["Close"]
    s = pd.DataFrame(index=ind.index)
    s["above200"] = (c > ind["SMA200"]).where(ind["SMA200"].notna())
    s["above50"] = (c > ind["SMA50"]).where(ind["SMA50"].notna())
    s["rsi_b"] = ind["RSI"].map(_rsi_bucket)
    s["macd_pos"] = (ind["MACD_hist"] > 0).where(ind["MACD_hist"].notna())
    return s


def _describe_state(cols: list[str], row) -> str:
    parts = []
    for col in cols:
        val = row[col]
        if col == "above200":
            parts.append("πάνω από SMA200" if val else "κάτω από SMA200")
        elif col == "above50":
            parts.append("πάνω από SMA50" if val else "κάτω από SMA50")
        elif col == "rsi_b":
            parts.append(RSI_BUCKET_TEXT[int(val)])
        elif col == "macd_pos":
            parts.append("MACD θετικό" if val else "MACD αρνητικό")
    return ", ".join(parts)


def analog_probability(ind: pd.DataFrame, h: int, min_n: int = 40) -> dict | None:
    """Πόσο συχνά ανέβηκε η μετοχή σε h ημέρες όταν ήταν σε ίδια κατάσταση με σήμερα."""
    reg = regime_frame(ind)
    fwd = ind["Close"].shift(-h) / ind["Close"] - 1
    current = reg.iloc[-1]
    levels = [["above200", "above50", "rsi_b", "macd_pos"],
              ["above200", "above50", "rsi_b"],
              ["above200", "rsi_b"],
              ["above50", "rsi_b"],
              ["rsi_b"]]
    for cols in levels:
        if current[cols].isna().any():
            continue
        mask = fwd.notna()
        for col in cols:
            mask &= reg[col] == current[col]
        n = int(mask.sum())
        if n >= min_n:
            r = fwd[mask]
            return {"n": n, "p": float((r > 0).mean()), "median": float(r.median()),
                    "mean": float(r.mean()), "state": _describe_state(cols, current)}
    return None


def model_probability(ind: pd.DataFrame, h: int) -> dict | None:
    """Λογιστική παλινδρόμηση στους δείκτες, με έλεγχο σε δεδομένα που δεν είδε."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None
    feats = feature_frame(ind)
    future = ind["Close"].shift(-h)
    y = (future > ind["Close"]).astype(float).where(future.notna())
    data = feats.assign(y=y).dropna(subset=FEATURES)
    if data.empty:
        return None
    latest = data[FEATURES].iloc[[-1]]
    labeled = data.dropna(subset=["y"])
    if len(labeled) < 400 or labeled["y"].nunique() < 2:
        return None

    def new_model():
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))

    split = int(len(labeled) * 0.75)
    train = labeled.iloc[: max(split - h, 50)]  # κενό h ημερών ώστε να μη «κρυφοκοιτάζει»
    test = labeled.iloc[split:]
    if train["y"].nunique() < 2 or len(test) < 50:
        return None
    m = new_model().fit(train[FEATURES], train["y"])
    p_test = m.predict_proba(test[FEATURES])[:, 1]
    yt = test["y"].to_numpy()
    base = float(train["y"].mean())
    acc = float(((p_test > 0.5) == (yt == 1)).mean())
    acc_const = float(((base > 0.5) == (yt == 1)).mean())
    brier = float(np.mean((p_test - yt) ** 2))
    brier_base = float(np.mean((base - yt) ** 2))
    skill = 1 - brier / brier_base if brier_base > 0 else 0.0

    full = new_model().fit(labeled[FEATURES], labeled["y"])
    p_now = float(full.predict_proba(latest)[0, 1])
    coefs = dict(zip(FEATURES, full[-1].coef_[0]))
    z = full[0].transform(latest)[0]
    contrib = {k: float(coefs[k] * z[i]) for i, k in enumerate(FEATURES)}
    return {"p": p_now, "acc": acc, "acc_const": acc_const, "skill": float(skill),
            "n_train": len(train), "n_test": len(test), "contrib": contrib}


FEATURE_TEXT = {
    "ret1": "Απόδοση χθες", "ret5": "Απόδοση εβδομάδας", "ret20": "Απόδοση μήνα",
    "d_sma20": "Απόσταση από SMA20", "d_sma50": "Απόσταση από SMA50", "d_sma200": "Απόσταση από SMA200",
    "rsi": "RSI", "macd_h": "MACD", "bb_pctb": "Θέση στις Bollinger", "atr_pct": "Μεταβλητότητα (ATR)",
    "vol_ratio": "Όγκος έναντι μέσου", "stoch_k": "Stochastic", "adx": "ADX",
    "di_diff": "Κατεύθυνση τάσης (+DI−DI)", "mfi": "Money Flow",
}


def combined_probability(ind: pd.DataFrame, horizons=(1, 5, 20)) -> dict[int, dict]:
    """Συνδυάζει: ιστορική βάση + παρόμοιες καταστάσεις + μοντέλο.

    Κάθε μέθοδος "συρρικνώνεται" προς τη βασική συχνότητα όσο λιγότερα
    στοιχεία έχει ή όσο χειρότερα τα πήγε σε δεδομένα που δεν είχε δει.
    """
    out: dict[int, dict] = {}
    c = ind["Close"]
    for h in horizons:
        fwd = (c.shift(-h) / c - 1).dropna()
        if len(fwd) < 60:
            continue
        p0 = float((fwd > 0).mean())
        analog = analog_probability(ind, h)
        model = model_probability(ind, h)
        parts = []
        if analog:
            k = 100.0
            n_eff = analog["n"] / math.sqrt(h)  # οι επικαλυπτόμενες περίοδοι μετράνε λιγότερο
            analog["p_adj"] = (n_eff * analog["p"] + k * p0) / (n_eff + k)
            parts.append(analog["p_adj"])
        if model:
            # Όριο «τύχης»: τόση βελτίωση πετυχαίνει το μοντέλο και σε τυχαίες τιμές
            # (95ο εκατοστημόριο σε προσομοιώσεις τυχαίου περιπάτου).
            noise = 0.006 * math.sqrt(h) * math.sqrt(580 / max(model["n_test"], 100))
            model["noise"] = noise
            w = float(np.clip((model["skill"] - noise) / noise, 0, 1))
            model["weight"] = w
            model["p_adj"] = p0 + w * (model["p"] - p0)
            parts.append(model["p_adj"])
        p = float(np.mean(parts)) if parts else p0
        out[h] = {"p_up": p, "p_down": 1 - p, "base": p0, "analog": analog, "model": model,
                  "n_hist": len(fwd)}
    return out


def volatility_ranges(close: pd.Series, horizons=(1, 5, 20), lookback=252, sims=20000, seed=7):
    """Εύρος πιθανής κίνησης (bootstrap των αποδόσεων του τελευταίου έτους, χωρίς τάση)."""
    r = np.log(close).diff().dropna().tail(lookback)
    if len(r) < 60:
        return {}
    r = (r - r.mean()).to_numpy()
    rng = np.random.default_rng(seed)
    thresholds = {1: 0.02, 5: 0.05, 20: 0.10}
    out = {}
    price = float(close.iloc[-1])
    for h in horizons:
        total = rng.choice(r, size=(sims, h)).sum(axis=1)
        pct = np.exp(total) - 1
        t = thresholds.get(h, 0.05)
        out[h] = {
            "low": price * (1 + np.quantile(pct, 0.05)),
            "high": price * (1 + np.quantile(pct, 0.95)),
            "low_pct": float(np.quantile(pct, 0.05)),
            "high_pct": float(np.quantile(pct, 0.95)),
            "thr": t,
            "p_up_big": float((pct > t).mean()),
            "p_down_big": float((pct < -t).mean()),
        }
    return out


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def options_view(chains: list[dict], spot: float) -> dict | None:
    """Τι «προεξοφλεί» η αγορά δικαιωμάτων.

    chains: [{"expiry": date, "days": int, "calls": DataFrame, "puts": DataFrame}]
    """
    if not chains or not spot:
        return None
    vol_c = vol_p = oi_c = oi_p = 0.0
    for ch in chains:
        vol_c += float(ch["calls"].get("volume", pd.Series(dtype=float)).fillna(0).sum())
        vol_p += float(ch["puts"].get("volume", pd.Series(dtype=float)).fillna(0).sum())
        oi_c += float(ch["calls"].get("openInterest", pd.Series(dtype=float)).fillna(0).sum())
        oi_p += float(ch["puts"].get("openInterest", pd.Series(dtype=float)).fillna(0).sum())
    out = {"pc_volume": vol_p / vol_c if vol_c else None,
           "pc_oi": oi_p / oi_c if oi_c else None}

    target = next((ch for ch in chains if ch["days"] >= 5), chains[0])
    calls, puts = target["calls"], target["puts"]
    common = sorted(set(calls["strike"]) & set(puts["strike"]))
    if not common:
        return out
    k = min(common, key=lambda s: abs(s - spot))
    cr, pr = calls[calls["strike"] == k].iloc[0], puts[puts["strike"] == k].iloc[0]

    def num(x):
        return 0.0 if x is None or pd.isna(x) else float(x)

    def mid(row):
        bid, ask = num(row.get("bid")), num(row.get("ask"))
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return num(row.get("lastPrice"))

    straddle = mid(cr) + mid(pr)
    days = max(int(target["days"]), 1)
    out.update({"expiry": target["expiry"], "days": days, "strike": float(k),
                "move_pct": straddle / spot if straddle > 0 else None})
    ivs = [v for v in (cr.get("impliedVolatility"), pr.get("impliedVolatility"))
           if v is not None and not pd.isna(v) and 0.01 < v < 5]
    if ivs:
        iv = float(np.mean(ivs))
        sd = iv * math.sqrt(days / 365)
        out["iv"] = iv
        out["range_low"] = spot * math.exp(-sd)
        out["range_high"] = spot * math.exp(sd)
        # πιθανότητα (ουδέτερη ως προς τον κίνδυνο) για κίνηση πάνω από ±5%
        out["p_up5"] = 1 - _norm_cdf((math.log(1.05) + 0.5 * sd ** 2) / sd)
        out["p_down5"] = _norm_cdf((math.log(0.95) + 0.5 * sd ** 2) / sd)
    return out


# ---------------------------------------------------------------------------
# Ειδήσεις: συναίσθημα και θέματα
# ---------------------------------------------------------------------------

FIN_LEXICON = {
    "beat": 2.0, "beats": 2.0, "miss": -2.0, "misses": -2.0, "missed": -2.0,
    "upgrade": 2.0, "upgrades": 2.0, "upgraded": 2.0, "downgrade": -2.0, "downgrades": -2.0,
    "downgraded": -2.0, "surge": 2.5, "surges": 2.5, "soar": 2.5, "soars": 2.5, "plunge": -3.0,
    "plunges": -3.0, "tumble": -2.5, "tumbles": -2.5, "slump": -2.5, "slumps": -2.5,
    "rally": 2.0, "rallies": 2.0, "bullish": 2.5, "bearish": -2.5, "lawsuit": -2.0,
    "probe": -1.5, "layoffs": -1.5, "record": 1.5, "outperform": 2.0, "underperform": -2.0,
    "jumps": 2.0, "jump": 2.0, "gains": 1.5, "slides": -1.5, "sinks": -2.0, "sink": -2.0,
    "rises": 1.5, "falls": -1.5, "fall": -1.5, "drops": -1.5, "drop": -1.5, "crash": -3.0,
    "recall": -2.0, "bankruptcy": -3.0, "selloff": -2.0, "sell-off": -2.0, "buyback": 1.5,
    "raises": 1.0, "cuts": -1.0, "warns": -2.0, "warning": -1.5, "strong": 1.5, "weak": -1.5,
    "tops": 1.5, "lags": -1.0, "climbs": 1.5, "slips": -1.0, "rebound": 1.5, "rebounds": 1.5,
    "overweight": 1.5, "underweight": -1.5, "investigation": -2.0, "fraud": -3.0, "subpoena": -2.0,
}

TOPICS = [
    (r"\b(rumou?rs?|reportedly|people familiar|sources (?:say|said)|said to (?:be|weigh)|"
     r"considering|weighs|explores|exploring|in talks)\b", "Φήμη / ανεπιβεβαίωτο"),
    (r"\b(acquir\w*|merger|takeover|buyout|bid for)\b", "Εξαγορά / συγχώνευση"),
    (r"\b(earnings|quarterly|q[1-4]|revenue|eps|results|guidance|outlook|forecast)\b", "Αποτελέσματα / προβλέψεις"),
    (r"\b(upgrade[sd]?|raises? (?:its |the )?price target|outperform|overweight)\b", "Αναβάθμιση αναλυτή"),
    (r"\b(downgrade[sd]?|cuts? (?:its |the )?price target|underperform|underweight)\b", "Υποβάθμιση αναλυτή"),
    (r"\b(lawsuit|sued|sues|probe|investigation|sec|doj|ftc|antitrust|fined?|settlement|regulators?)\b",
     "Νομικά / ρυθμιστικά"),
    (r"\b(layoffs?|job cuts|restructur\w*)\b", "Περικοπές / αναδιάρθρωση"),
    (r"\b(buyback|repurchase|dividend)\b", "Μέρισμα / επαναγορά"),
    (r"\b(fda|approval|approved|clinical|trial)\b", "Εγκρίσεις"),
    (r"\b(partnership|partners with|contract|launch\w*|unveil\w*|new product)\b", "Προϊόντα / συνεργασίες"),
    (r"\b(ceo|cfo|resign\w*|steps down|appoint\w*|insider)\b", "Διοίκηση / insiders"),
    (r"\b(short seller|short report|short interest)\b", "Short sellers"),
    (r"\b(tariffs?|export controls?|sanctions?)\b", "Δασμοί / γεωπολιτικά"),
]


@dataclass
class NewsScorer:
    _vader: object = field(default=None, init=False)

    def __post_init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            self._vader.lexicon.update(FIN_LEXICON)
        except Exception:
            self._vader = None

    def score(self, text: str) -> float:
        if not text:
            return 0.0
        if self._vader is not None:
            return float(self._vader.polarity_scores(text)["compound"])
        words = re.findall(r"[a-z\-]+", text.lower())
        s = sum(FIN_LEXICON.get(w, 0.0) for w in words)
        return float(np.tanh(s / 4))


def topics_for(text: str) -> list[str]:
    t = (text or "").lower()
    return [label for pattern, label in TOPICS if re.search(pattern, t)]


def sentiment_label(s: float) -> str:
    return "Θετική" if s >= 0.15 else "Αρνητική" if s <= -0.15 else "Ουδέτερη"


def analyze_news(items: list[dict], scorer: NewsScorer | None = None) -> dict:
    """Βαθμολογεί κάθε είδηση (τίτλος βαραίνει διπλά) και βγάζει σύνοψη."""
    scorer = scorer or NewsScorer()
    enriched = []
    for it in items:
        title, summary = it.get("title", ""), it.get("summary", "") or ""
        s_title = scorer.score(title)
        s = (2 * s_title + scorer.score(summary[:400])) / 3 if summary else s_title
        enriched.append({**it, "sentiment": s, "label": sentiment_label(s),
                         "topics": topics_for(f"{title} {summary}")})
    if not enriched:
        return {"items": [], "avg": 0.0, "pos": 0, "neg": 0, "neu": 0, "topics": {}}
    avg = float(np.mean([e["sentiment"] for e in enriched]))
    counts: dict[str, int] = {}
    for e in enriched:
        for tp in e["topics"]:
            counts[tp] = counts.get(tp, 0) + 1
    return {
        "items": enriched,
        "avg": avg,
        "pos": sum(e["label"] == "Θετική" for e in enriched),
        "neg": sum(e["label"] == "Αρνητική" for e in enriched),
        "neu": sum(e["label"] == "Ουδέτερη" for e in enriched),
        "topics": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
    }

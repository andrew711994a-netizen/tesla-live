"""Γρήγορες συναλλαγές σε κεριά 5 λεπτών: μοτίβα κεριών, σχήματα γραφήματος και δύο γνωστά φαινόμενα της αγοράς.

Τα σήματα βγαίνουν στο κλείσιμο κάθε κεριού 5 λεπτών (μόνο με κεριά που έχουν κλείσει) και η εκτέλεση
προσομοιώνεται σε κεριά 1 λεπτού της Capital.com με bid/ask, όπως στο bot/scalp.py: αγορά στο ask, πώληση στο
bid, είσοδος στο επόμενο λεπτό, stop/στόχος λεπτό προς λεπτό. Καμία θέση το βράδυ. ΜΟΝΟ έρευνα: δεν ανοίγει θέσεις.

Οι κανόνες είναι γραμμένοι από πριν, με τιμές από τη βιβλιογραφία ή στρογγυλές, χωρίς βελτιστοποίηση πάνω στα
δεδομένα. Φίλτρο κόστους σε όλες: καμία συναλλαγή αν το ρίσκο της είναι κάτω από 4 φορές το spread της στιγμής.
Κρίση όπως στο bot/scalp.py: κερδίζει μετά το spread, με ολίσθηση, με 1′ καθυστέρηση, σε κάθε τρίτο της περιόδου,
με αρκετές συναλλαγές, σε τουλάχιστον 2 προϊόντα.

Εκτέλεση (με τα CAPITAL_* στο περιβάλλον):  python -m bot.scalp5   →   reports/scalp5.md
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .capital import Capital
from .scalp import (MIN_PF, MIN_TRADES, SCENARIOS, Trade, _mid, download, g, regular_hours, run_day,
                    summarize, verdict)

ROOT = Path(__file__).resolve().parent.parent
INSTRUMENTS = {"US100": "Nasdaq-100 (CFD δείκτη)", "US500": "S&P 500 (CFD δείκτη)", "AAPL": "Apple",
               "MSFT": "Microsoft", "NVDA": "Nvidia", "TSLA": "Tesla", "AMZN": "Amazon", "META": "Meta",
               "GOLD": "Χρυσός", "SILVER": "Ασήμι"}
DAYS = 550
COST_MULT = 4.0       # ρίσκο τουλάχιστον 4 × spread
START, LAST = 9 * 60 + 45, 15 * 60 + 10   # σήματα από κεριά που ξεκινούν 9:45–15:10 (εκτός από το momentum)


# ───────────── Κεριά 5 λεπτών και δείκτες ─────────────
def bars5(day1: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Κεριά 5 λεπτών (μέση τιμή bid/ask) και η θέση του τελευταίου λεπτού κάθε κεριού στη μέρα."""
    m = _mid(day1)
    key = m.index.floor("5min")
    grp = m.groupby(key)
    b = pd.DataFrame({"o": grp["o"].first(), "h": grp["h"].max(), "l": grp["l"].min(), "c": grp["c"].last(),
                      "v": grp["v"].sum()})
    last = pd.Series(np.arange(len(m)), index=m.index).groupby(key).max().to_numpy()
    return b, last


def features(days: list[tuple[object, pd.DataFrame]]) -> tuple[pd.DataFrame, dict]:
    """Όλα τα κεριά 5 λεπτών στη σειρά: οι μέσοι όροι και το ATR συνεχίζουν από μέρα σε μέρα, το VWAP ξεκινά κάθε μέρα."""
    parts, last = [], {}
    for d, day in days:
        b, lp = bars5(day)
        b["day"] = d
        b["k"] = np.arange(len(b))
        parts.append(b)
        last[d] = lp
    m = pd.concat(parts)
    c = m["c"]
    m["ema20"] = c.ewm(span=20, adjust=False).mean()
    m["ema50"] = c.ewm(span=50, adjust=False).mean()
    tr = pd.concat([m["h"] - m["l"], (m["h"] - c.shift()).abs(), (m["l"] - c.shift()).abs()], axis=1).max(axis=1)
    m["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    tp = (m["h"] + m["l"] + m["c"]) / 3
    w = m["v"].where(m.groupby("day")["v"].transform("sum") > 0, 1.0)
    m["vwap"] = (tp * w).groupby(m["day"]).cumsum() / w.groupby(m["day"]).cumsum()
    m["mod"] = m.index.hour * 60 + m.index.minute
    by = m.groupby("day")
    for s in (1, 2):
        for col in ("o", "h", "l", "c"):
            m[f"{col}{s}"] = by[col].shift(s)            # κεριά της ίδιας μέρας μόνο
    m["low12"] = by["l"].transform(lambda x: x.rolling(12, min_periods=1).min())
    m["high12"] = by["h"].transform(lambda x: x.rolling(12, min_periods=1).max())
    m["hod_prev"] = by["h"].transform(lambda x: x.cummax().shift())
    m["lod_prev"] = by["l"].transform(lambda x: x.cummin().shift())
    m["vavg_prev"] = by["v"].transform(lambda x: x.expanding().mean().shift())
    m["ema20_1"] = m["ema20"].shift()
    return m, last


# ───────────── Τεχνικές: κάθε μία επιστρέφει {θέση κεριού: (κατεύθυνση, stop, στόχος, λεπτά, ανά μέρα)} ─────────────
def _arr(m, *cols):
    return [m[c].to_numpy(float) for c in cols]


def _window(m) -> np.ndarray:
    mod = m["mod"].to_numpy()
    return (mod >= START) & (mod <= LAST)


def s_engulf(m):
    """Engulfing προς την τάση της ημέρας (πάνω/κάτω από VWAP, EMA20 πάνω/κάτω από EMA50). Stop στην άκρη των 2
    κεριών, στόχος 2R, έως 60′."""
    o, h, l, c, o1, h1, l1, c1, atr, vw, e20, e50 = _arr(m, "o", "h", "l", "c", "o1", "h1", "l1", "c1", "atr",
                                                         "vwap", "ema20", "ema50")
    body, body1 = np.abs(c - o), np.abs(c1 - o1)
    big = (body > body1) & (body >= 0.3 * atr)
    bull = (c1 < o1) & (c > o) & (c >= o1) & (o <= c1) & big & (c > vw) & (e20 > e50)
    bear = (c1 > o1) & (c < o) & (c <= o1) & (o >= c1) & big & (c < vw) & (e20 < e50)
    ok = _window(m)
    out = {}
    for i in np.flatnonzero(bull & ok):
        out[i] = (1, ("level", min(l[i], l1[i]) - 0.05 * atr[i]), ("r", 2.0), 60)
    for i in np.flatnonzero(bear & ok):
        out[i] = (-1, ("level", max(h[i], h1[i]) + 0.05 * atr[i]), ("r", 2.0), 60)
    return out


def s_hammer(m):
    """Σφυρί στο χαμηλό της τελευταίας ώρας κάτω από το VWAP (shooting star στο υψηλό πάνω από το VWAP):
    επιστροφή προς το VWAP. Stop πέρα από την ουρά, στόχος το VWAP (τουλάχιστον 1R) αλλιώς 1,5R, έως 60′."""
    o, h, l, c, atr, vw, lo12, hi12 = _arr(m, "o", "h", "l", "c", "atr", "vwap", "low12", "high12")
    body, rng = np.abs(c - o), h - l
    up_w, lo_w = h - np.maximum(o, c), np.minimum(o, c) - l
    meaningful = rng >= 0.6 * atr
    hammer = (lo_w >= 2 * body) & (up_w <= 0.5 * body + 0.1 * rng) & meaningful & (l <= lo12) & (c < vw)
    star = (up_w >= 2 * body) & (lo_w <= 0.5 * body + 0.1 * rng) & meaningful & (h >= hi12) & (c > vw)
    ok = _window(m)
    out = {}
    for i in np.flatnonzero(hammer & ok):
        stop = l[i] - 0.05 * atr[i]
        tgt = ("level", vw[i]) if vw[i] - c[i] >= c[i] - stop else ("r", 1.5)
        out[i] = (1, ("level", stop), tgt, 60)
    for i in np.flatnonzero(star & ok):
        stop = h[i] + 0.05 * atr[i]
        tgt = ("level", vw[i]) if c[i] - vw[i] >= stop - c[i] else ("r", 1.5)
        out[i] = (-1, ("level", stop), tgt, 60)
    return out


def s_pullback(m):
    """Επιστροφή στον EMA20 σε τάση: το κερί αγγίζει τον EMA20 και κλείνει δυνατά προς την τάση. Stop πέρα από το
    κερί, στόχος 2R, έως 60′."""
    o, h, l, c, atr, vw, e20, e50, e20p = _arr(m, "o", "h", "l", "c", "atr", "vwap", "ema20", "ema50", "ema20_1")
    rng = h - l
    up = (c > vw) & (e20 > e50) & (e20 > e20p) & (l <= e20) & (c > e20) & (c > o) & (c >= l + 0.6 * rng)
    dn = (c < vw) & (e20 < e50) & (e20 < e20p) & (h >= e20) & (c < e20) & (c < o) & (c <= h - 0.6 * rng)
    ok = _window(m) & (rng >= 0.4 * atr)
    out = {}
    for i in np.flatnonzero(up & ok):
        out[i] = (1, ("level", l[i] - 0.1 * atr[i]), ("r", 2.0), 60)
    for i in np.flatnonzero(dn & ok):
        out[i] = (-1, ("level", h[i] + 0.1 * atr[i]), ("r", 2.0), 60)
    return out


def s_star(m):
    """Morning star στο χαμηλό της τελευταίας ώρας / evening star στο υψηλό. Stop πέρα από τα 3 κεριά, στόχος 2R,
    έως 60′."""
    o, h, l, c, o1, h1, l1, c1, o2, h2, l2, c2, atr, lo12, hi12 = _arr(
        m, "o", "h", "l", "c", "o1", "h1", "l1", "c1", "o2", "h2", "l2", "c2", "atr", "low12", "high12")
    small = np.abs(c1 - o1) <= 0.3 * (h1 - l1)
    lo3 = np.fmin(np.fmin(l, l1), l2)
    hi3 = np.fmax(np.fmax(h, h1), h2)
    morning = (c2 < o2) & (o2 - c2 >= 0.5 * atr) & small & (c > o) & (c > (o2 + c2) / 2) & (lo3 <= lo12)
    evening = (c2 > o2) & (c2 - o2 >= 0.5 * atr) & small & (c < o) & (c < (o2 + c2) / 2) & (hi3 >= hi12)
    ok = _window(m)
    out = {}
    for i in np.flatnonzero(morning & ok):
        out[i] = (1, ("level", lo3[i] - 0.05 * atr[i]), ("r", 2.0), 60)
    for i in np.flatnonzero(evening & ok):
        out[i] = (-1, ("level", hi3[i] + 0.05 * atr[i]), ("r", 2.0), 60)
    return out


def s_inside(m):
    """Inside bar και σπάσιμο της «μητέρας» προς την τάση. Stop στην άλλη άκρη της μητέρας, στόχος 2R, έως 60′."""
    c, h1, l1, h2, l2, atr, vw, e20, e50 = _arr(m, "c", "h1", "l1", "h2", "l2", "atr", "vwap", "ema20", "ema50")
    inside = (h1 < h2) & (l1 > l2) & (h2 - l2 <= 2 * atr)
    up = inside & (c > h2) & (c > vw) & (e20 > e50)
    dn = inside & (c < l2) & (c < vw) & (e20 < e50)
    ok = _window(m)
    out = {}
    for i in np.flatnonzero(up & ok):
        out[i] = (1, ("level", l2[i] - 0.05 * atr[i]), ("r", 2.0), 60)
    for i in np.flatnonzero(dn & ok):
        out[i] = (-1, ("level", h2[i] + 0.05 * atr[i]), ("r", 2.0), 60)
    return out


def s_double(m):
    """Double bottom / top μέσα στη μέρα: δύο χαμηλά (υψηλά) σε απόσταση έως 0,3 ATR, με τουλάχιστον 4 κεριά
    ανάμεσα και ύψος σχήματος τουλάχιστον 1 ATR· σήμα στο πρώτο κλείσιμο πέρα από τη «λαιμόκοψη». Stop στη μέση του
    σχήματος, στόχος λαιμόκοψη + ύψος (κλασικός στόχος), έως 120′."""
    out = {}
    h_all, l_all, c_all, atr_all = _arr(m, "h", "l", "c", "atr")
    mod_all = m["mod"].to_numpy()
    starts = np.flatnonzero(m["k"].to_numpy() == 0)
    ends = np.append(starts[1:], len(m))
    for s, e in zip(starts, ends):
        h, l, c, atr, mod = h_all[s:e], l_all[s:e], c_all[s:e], atr_all[s:e], mod_all[s:e]
        n = len(c)
        lows = [j for j in range(2, n - 2) if l[j] == l[j - 2:j + 3].min()]
        highs = [j for j in range(2, n - 2) if h[j] == h[j - 2:j + 3].max()]
        for k in range(6, n):
            if not (START <= mod[k] <= LAST):
                continue
            a_ = atr[k]
            cand = [j for j in lows if k - 36 <= j <= k - 2]       # επιβεβαιωμένα (2 κεριά μετά), τις 3 τελευταίες ώρες
            for x in range(len(cand)):
                for y in range(x + 1, len(cand)):
                    a, b = cand[x], cand[y]
                    if b - a < 4 or abs(l[a] - l[b]) > 0.3 * a_:
                        continue
                    neck = h[a:b + 1].max()
                    height = neck - min(l[a], l[b])
                    if height >= a_ and c[k] > neck and c[k - 1] <= neck and (c[b:k] <= neck).all():
                        out[s + k] = (1, ("level", neck - 0.5 * height), ("level", neck + height), 120)
            cand = [j for j in highs if k - 36 <= j <= k - 2]
            for x in range(len(cand)):
                for y in range(x + 1, len(cand)):
                    a, b = cand[x], cand[y]
                    if b - a < 4 or abs(h[a] - h[b]) > 0.3 * a_:
                        continue
                    neck = l[a:b + 1].min()
                    height = max(h[a], h[b]) - neck
                    if height >= a_ and c[k] < neck and c[k - 1] >= neck and (c[b:k] >= neck).all():
                        out[s + k] = (-1, ("level", neck + 0.5 * height), ("level", neck - height), 120)
    return out


def s_momentum(m):
    """Ενδοσυνεδριακό momentum (Gao, Han, Li & Zhou, J. Financial Economics 2018): η κίνηση από το χθεσινό
    κλείσιμο ως τις 10:00 δείχνει την κατεύθυνση της τελευταίας μισής ώρας. Είσοδος 15:30, έξοδος 15:55,
    προστατευτικό stop 2 ATR. Μία συναλλαγή τη μέρα."""
    out = {}
    c_all, atr_all = _arr(m, "c", "atr")
    mod_all = m["mod"].to_numpy()
    days = m["day"].to_numpy()
    starts = np.flatnonzero(m["k"].to_numpy() == 0)
    prev_close = None
    prev_day = None
    for s, e in zip(starts, np.append(starts[1:], len(m))):
        mod = mod_all[s:e]
        i10 = np.flatnonzero(mod == 9 * 60 + 55)     # κερί 9:55–10:00
        i1530 = np.flatnonzero(mod == 15 * 60 + 25)  # κερί 15:25–15:30
        consecutive = prev_day is not None and (pd.Timestamp(days[s]) - pd.Timestamp(prev_day)).days <= 4
        if consecutive and len(i10) and len(i1530) and prev_close:
            r1 = c_all[s + i10[0]] / prev_close - 1
            if r1 != 0:
                k = s + i1530[0]
                out[k] = (1 if r1 > 0 else -1, ("dist", 2 * atr_all[k]), ("r", 10.0), 30)
        prev_close, prev_day = c_all[e - 1], days[s]
    return out


def s_trendday(m):
    """Μέρα τάσης: την πρώτη ώρα η τιμή έμεινε πάνω (κάτω) από το VWAP στο 80% των κεριών και κλείνει την ώρα πάνω
    (κάτω) από το άνοιγμα. Μετά τις 10:30, αγορά (πώληση) όταν ένα κερί αγγίζει το VWAP και κλείνει προς την τάση.
    Stop λίγο πέρα από το VWAP ή το κερί, στόχος 2R, έως 90′, έως 2 τη μέρα."""
    out = {}
    o_all, h_all, l_all, c_all, atr_all, vw_all = _arr(m, "o", "h", "l", "c", "atr", "vwap")
    mod_all = m["mod"].to_numpy()
    starts = np.flatnonzero(m["k"].to_numpy() == 0)
    for s, e in zip(starts, np.append(starts[1:], len(m))):
        o, h, l, c, atr, vw, mod = (x[s:e] for x in (o_all, h_all, l_all, c_all, atr_all, vw_all, mod_all))
        first = mod < 10 * 60 + 30
        if first.sum() < 10:
            continue
        above = (c[first] > vw[first]).mean()
        last_first = np.flatnonzero(first)[-1]
        if above >= 0.8 and c[last_first] > o[0]:
            side = 1
        elif above <= 0.2 and c[last_first] < o[0]:
            side = -1
        else:
            continue
        for k in range(last_first + 1, len(c)):
            if not (10 * 60 + 30 <= mod[k] <= LAST):
                continue
            if side > 0 and l[k] <= vw[k] < c[k] and c[k] > o[k]:
                out[s + k] = (1, ("level", min(l[k], vw[k]) - 0.25 * atr[k]), ("r", 2.0), 90)
            if side < 0 and h[k] >= vw[k] > c[k] and c[k] < o[k]:
                out[s + k] = (-1, ("level", max(h[k], vw[k]) + 0.25 * atr[k]), ("r", 2.0), 90)
    return out


def s_breakout(m):
    """Σπάσιμο του υψηλού (χαμηλού) της ημέρας μετά τις 10:30 με όγκο τουλάχιστον 1,5× τον μέσο της ημέρας και
    προς την πλευρά του VWAP. Stop 0,75 ATR ή κάτω από το κερί (όποιο πιο κοντά), στόχος 2R, έως 60′."""
    h, l, c, v, atr, vw, hod, lod, vavg = _arr(m, "h", "l", "c", "v", "atr", "vwap", "hod_prev", "lod_prev",
                                               "vavg_prev")
    mod = m["mod"].to_numpy()
    ok = (mod >= 10 * 60 + 30) & (mod <= LAST) & (vavg > 0) & (v >= 1.5 * vavg)
    up = ok & (c > hod) & (c > vw)
    dn = ok & (c < lod) & (c < vw)
    out = {}
    for i in np.flatnonzero(up):
        out[i] = (1, ("level", max(l[i] - 0.05 * atr[i], c[i] - 0.75 * atr[i])), ("r", 2.0), 60)
    for i in np.flatnonzero(dn):
        out[i] = (-1, ("level", min(h[i] + 0.05 * atr[i], c[i] + 0.75 * atr[i])), ("r", 2.0), 60)
    return out


# όνομα → (συνάρτηση, μέγιστες συναλλαγές τη μέρα)
ALL_STRATS = {
    "engulf": ("Engulfing προς την τάση", s_engulf, 3),
    "hammer": ("Σφυρί / shooting star σε ακραίο σημείο", s_hammer, 3),
    "pullback": ("Επιστροφή στον EMA20 σε τάση", s_pullback, 3),
    "star": ("Morning / evening star", s_star, 3),
    "inside": ("Inside bar σπάσιμο", s_inside, 3),
    "double": ("Double bottom / top", s_double, 2),
    "momentum": ("Momentum τελευταίας μισής ώρας (Gao κ.ά. 2018)", s_momentum, 1),
    "trendday": ("Μέρα τάσης: επιστροφή στο VWAP", s_trendday, 2),
    "breakout": ("Σπάσιμο υψηλού/χαμηλού ημέρας με όγκο", s_breakout, 2),
}
STRATS = dict(ALL_STRATS)


def configure() -> str:
    """SCALP_DAYS, SCALP_INSTRUMENTS, SCALP_STRATS (π.χ. engulf,double), SCALP_OUT (όνομα αναφοράς)."""
    global DAYS, INSTRUMENTS, STRATS
    days = os.environ.get("SCALP_DAYS", "").strip()
    if days:
        DAYS = max(5, min(int(days), 1500))
    inst = [x.strip().upper() for x in os.environ.get("SCALP_INSTRUMENTS", "").split(",") if x.strip()]
    if inst:
        if not all(x.isalnum() and len(x) <= 12 for x in inst):
            raise ValueError(f"Άκυρα σύμβολα: {inst}")
        INSTRUMENTS = {x: INSTRUMENTS.get(x, x) for x in inst}
    keys = [x.strip().lower() for x in os.environ.get("SCALP_STRATS", "").split(",") if x.strip()]
    if keys:
        unknown = [k for k in keys if k not in ALL_STRATS]
        if unknown:
            raise ValueError(f"Άγνωστες τεχνικές: {unknown}")
        STRATS = {k: ALL_STRATS[k] for k in keys}
    out = os.environ.get("SCALP_OUT", "").strip() or "scalp5"
    if not out.replace("_", "").isalnum():
        raise ValueError(f"Άκυρο όνομα αναφοράς: {out}")
    return out


# ───────────── Προσομοίωση ─────────────
def day_signals(sig: dict, m: pd.DataFrame, last: dict, days: dict) -> dict:
    """Σήματα ανά μέρα, στη θέση του τελευταίου λεπτού του κεριού 5′, μόνο όσα περνούν το φίλτρο κόστους."""
    per: dict = {}
    day_col, k_col, c_col = m["day"].to_numpy(), m["k"].to_numpy(), m["c"].to_numpy()
    for i, s in sig.items():
        d, k = day_col[i], int(k_col[i])
        day = days[d]
        pos = int(last[d][k])
        spr = float(day["oa"].iloc[pos] - day["ob"].iloc[pos])
        side, stop, _, _ = s
        risk = stop[1] if stop[0] == "dist" else side * (c_col[i] - stop[1])
        if not (risk > COST_MULT * spr):
            continue
        per.setdefault(d, {})[pos] = s
    return per


def evaluate(df: pd.DataFrame) -> dict:
    groups = [(d, day) for d, day in df.groupby(df.index.date) if len(day) >= 300]
    if len(groups) < 20:
        return {}
    days = dict(groups)
    m, last = features(groups)
    order = sorted(days)
    thirds = [set(x) for x in np.array_split(np.array(order, dtype=object), 3)]
    out = {}
    for key, (name, fn, per_day) in STRATS.items():
        per = day_signals(fn(m), m, last, days)
        res = {}
        for sc, kw in SCENARIOS.items():
            trades: list[Trade] = []
            for d, sigs in per.items():
                trades += run_day(days[d], sigs.get, max_trades=per_day, **kw)
            res[sc] = summarize(trades)
            if sc == "base":
                res["thirds"] = [summarize([t for t in trades if t.day in th]) for th in thirds]
                res["long"] = summarize([t for t in trades if t.side > 0])
                res["short"] = summarize([t for t in trades if t.side < 0])
        out[key] = res
    return out


HEAD = ("| Τεχνική | Συναλλαγές | Επιτυχία | Μέσο R | PF | Με ολίσθηση | Με 1′ καθυστέρηση | Τρίτα (μέσο R) | "
        "Αγορές / πωλήσεις (μέσο R) | Spread (R) | Λογαριασμός (0,5% ρίσκο) | Κρίση |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|")


def row(name: str, res: dict) -> str:
    b = res["base"]
    if not b["trades"]:
        return f"| {name} | 0 | – | – | – | – | – | – | – | – | – | ✘ καμία συναλλαγή |"
    ok, why = verdict(res)
    thirds = " / ".join(g(th["avg_r"], "+.2f") if th["trades"] else "–" for th in res["thirds"])
    ls = " / ".join(f"{g(x['avg_r'], '+.2f')} ({x['trades']})" if x["trades"] else "–"
                    for x in (res["long"], res["short"]))
    return (f"| {name} | {b['trades']} ({g(b['per_day'], '.1f')}/μέρα) | {b['win'] * 100:.0f}% | "
            f"{g(b['avg_r'], '+.3f')} | {g(b['pf'], '.2f')} | {g(res['slip']['avg_r'], '+.3f')} | "
            f"{g(res['delay']['avg_r'], '+.3f')} | {thirds} | {ls} | {g(b['cost_r'], '.2f')} | "
            f"{g(b['equity'] * 100, '+.1f')}% | {'✔ περνά' if ok else '✘ ' + '; '.join(why)} |")


def main() -> None:
    report = configure()
    cap = Capital(os.environ["CAPITAL_API_KEY"], os.environ["CAPITAL_IDENTIFIER"], os.environ["CAPITAL_PASSWORD"],
                  os.environ.get("CAPITAL_ENV", "demo"))
    out = [f"# Έρευνα: γρήγορες συναλλαγές σε κεριά 5 λεπτών ({DAYS} ημέρες ιστορικού)", "",
           "Σήματα στο κλείσιμο κάθε κεριού 5 λεπτών· εκτέλεση λεπτό προς λεπτό στα πραγματικά κεριά της Capital.com "
           "με bid/ask (αγορά στο ask, πώληση στο bid, είσοδος στο επόμενο λεπτό). Μόνο κανονικές ώρες Νέας Υόρκης "
           "(9:30–16:00), καμία θέση το βράδυ. Αν stop και στόχος πιάνονται στο ίδιο λεπτό, μετράει το stop. "
           f"Φίλτρο κόστους: καμία συναλλαγή με ρίσκο κάτω από {g(COST_MULT, '.0f')} φορές το spread. Οι κανόνες "
           "γράφτηκαν από πριν, χωρίς βελτιστοποίηση πάνω στα δεδομένα.", "",
           "**R** = πόσες φορές το ρίσκο της συναλλαγής· μέσο R πάνω από 0 = κερδίζει. **Με ολίσθηση**: μισό spread "
           "χειρότερη τιμή σε κάθε είσοδο και stop. **Με 1′ καθυστέρηση**: η είσοδος ένα λεπτό αργότερα. **Τρίτα**: "
           "η περίοδος σε τρία κομμάτια. **Spread (R)**: πόσο από το ρίσκο τρώει το spread.", "",
           f"**Αξιόπιστη** = τουλάχιστον {MIN_TRADES} συναλλαγές, profit factor ≥ {g(MIN_PF, '.2f')}, κερδίζει σε "
           "όλα τα σενάρια και σε κάθε τρίτο. Για να προχωρήσει σε demo, πρέπει να περνά σε τουλάχιστον 2 προϊόντα.",
           ""]
    out += ["## Οι τεχνικές", ""]
    out += [f"- **{name}**: {' '.join((fn.__doc__ or '').split())}" for name, fn, _ in STRATS.values()]
    out.append("")
    summary: dict = {}
    passes: dict[str, list[str]] = {k: [] for k in STRATS}
    grid: dict[str, dict[str, dict]] = {k: {} for k in STRATS}
    for epic, label in INSTRUMENTS.items():
        try:
            df = regular_hours(download(cap, epic, DAYS))
        except Exception as e:
            out += [f"### {label} ({epic})", "", f"Χωρίς δεδομένα: {e}", ""]
            continue
        res = evaluate(df) if not df.empty else {}
        if not res:
            out += [f"### {label} ({epic})", "", "Η Capital.com δεν έδωσε αρκετά κεριά λεπτού.", ""]
            continue
        days = sorted(set(df.index.date))
        spread = float(((df["oa"] - df["ob"]) / ((df["oa"] + df["ob"]) / 2)).median())
        summary[epic] = {"days": len(days), "from": str(days[0]), "to": str(days[-1]), "spread": spread,
                         "results": res}
        out += [f"### {label} ({epic})", "",
                f"Δεδομένα {days[0]:%d/%m/%Y} – {days[-1]:%d/%m/%Y}: {len(days)} μέρες. "
                f"Τυπικό spread {g(spread * 100, '.3f')}% της τιμής.", "", HEAD]
        for key, r in res.items():
            out.append(row(STRATS[key][0], r))
            grid[key][epic] = r
            if verdict(r)[0]:
                passes[key].append(epic)
        out.append("")
    if summary:
        epics = list(summary)
        out += ["## Σύνοψη (μέσο R ανά προϊόν· ✔ = περνά όλους τους ελέγχους)", "",
                "| Τεχνική | " + " | ".join(epics) + " |", "|---|" + "---|" * len(epics)]
        for key, (name, _, _) in STRATS.items():
            cells = []
            for e in epics:
                r = grid[key].get(e)
                if not r or not r["base"]["trades"]:
                    cells.append("–")
                else:
                    cells.append(f"{g(r['base']['avg_r'], '+.2f')}{' ✔' if verdict(r)[0] else ''}")
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        out.append("")
    good = {k: e for k, e in passes.items() if len(e) >= 2}
    out += ["## Συμπέρασμα", ""]
    if good:
        out += [f"- **{STRATS[k][0]}**: περνά όλους τους ελέγχους σε {', '.join(e)} → υποψήφια για δοκιμή σε demo."
                for k, e in good.items()]
    else:
        out.append("- **Καμία τεχνική δεν πέρασε όλους τους ελέγχους σε 2 ή περισσότερα προϊόντα.** Με αυτά τα "
                   "δεδομένα, γρήγορες συναλλαγές σε κεριά 5 λεπτών με αυτούς τους κανόνες δεν είναι αξιόπιστες "
                   "μετά τα κόστη.")
    for k, e in passes.items():
        if len(e) == 1:
            out.append(f"- {STRATS[k][0]}: πέρασε μόνο σε {e[0]} (ένα προϊόν δεν αρκεί, μπορεί να είναι τύχη).")
    n_tests = len(STRATS) * max(len(summary), 1)
    out += ["", f"Σημείωση: δοκιμάστηκαν {n_tests} συνδυασμοί τεχνικής και προϊόντος. Όσο περισσότερα δοκιμάζεις, τόσο "
            "πιο εύκολα κάτι «περνά» από τύχη· γι' αυτό ζητάμε να περνά όλους τους ελέγχους σε 2 προϊόντα και μετά "
            "δοκιμή σε demo πριν από οποιαδήποτε πραγματική χρήση.", ""]
    path = ROOT / "reports"
    path.mkdir(exist_ok=True)
    (path / f"{report}.md").write_text("\n".join(out), encoding="utf-8")
    (path / f"{report}.json").write_text(json.dumps({"summary": summary, "passes": passes}, ensure_ascii=False,
                                                indent=1, default=float), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()

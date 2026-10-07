"""Scalping και γρήγορες ενδοσυνεδριακές συναλλαγές: υπάρχει κάτι αξιόπιστο μετά τα πραγματικά κόστη;

Κατεβάζει από την Capital.com κεριά 1 λεπτού με τιμές bid/ask (άρα με το πραγματικό spread) και δοκιμάζει
γνωστές τεχνικές στις κανονικές ώρες της Νέας Υόρκης (9:30–16:00). Κάθε αγορά γίνεται στο ask και κάθε πώληση
στο bid, όπως στην πράξη. ΜΟΝΟ έρευνα: δεν ανοίγει θέσεις.

Μια τεχνική λέγεται αξιόπιστη μόνο αν κερδίζει: μετά το spread, με επιπλέον ολίσθηση, με 1 λεπτό καθυστέρηση
στην είσοδο, σε κάθε τρίτο της περιόδου, σε τουλάχιστον 2 προϊόντα, με αρκετές συναλλαγές.

Εκτέλεση (με τα CAPITAL_* στο περιβάλλον):  python -m bot.scalp   →   reports/scalp.md
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .capital import Capital, CapitalError

ROOT = Path(__file__).resolve().parent.parent
NY = "America/New_York"
INSTRUMENTS = {"US100": "Nasdaq-100 (CFD δείκτη)", "US500": "S&P 500 (CFD δείκτη)", "TSLA": "Tesla",
               "NVDA": "Nvidia", "AAPL": "Apple", "MSFT": "Microsoft"}
DAYS = 120
RISK = 0.005          # για την καμπύλη αξίας: 0,5% του λογαριασμού ανά συναλλαγή
MIN_TRADES = 50       # λιγότερες συναλλαγές = δεν λέει τίποτα
MIN_PF = 1.15
SCENARIOS = {"base": {"slip": 0.0, "delay": 0},      # μόνο το πραγματικό spread
             "slip": {"slip": 0.5, "delay": 0},      # + μισό spread χειρότερη τιμή σε είσοδο και stop
             "delay": {"slip": 0.0, "delay": 1}}     # είσοδος 1 λεπτό αργότερα


# ───────────── Δεδομένα ─────────────
def parse(rows: list[dict]) -> pd.DataFrame:
    recs = []
    for p in rows:
        o, h, l, c = (p.get(k) or {} for k in ("openPrice", "highPrice", "lowPrice", "closePrice"))
        recs.append({"t": p.get("snapshotTimeUTC") or p.get("snapshotTime"),
                     "ob": o.get("bid"), "oa": o.get("ask"), "hb": h.get("bid"), "ha": h.get("ask"),
                     "lb": l.get("bid"), "la": l.get("ask"), "cb": c.get("bid"), "ca": c.get("ask"),
                     "v": p.get("lastTradedVolume") or 0})
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs).dropna(subset=["t", "ob", "oa", "hb", "ha", "lb", "la", "cb", "ca"])
    df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(NY)
    df = df.drop_duplicates("t").set_index("t").sort_index()
    return df.astype(float)


def download(cap: Capital, epic: str, days: int = DAYS) -> pd.DataFrame:
    """Ένα αίτημα ανά εργάσιμη: 13:25–21:05 UTC καλύπτει τις ώρες της Νέας Υόρκης με θερινή και χειμερινή ώρα."""
    today = datetime.now(timezone.utc).date()
    rows: list[dict] = []
    fmt = "%Y-%m-%dT%H:%M:%S"
    for k in range(days, -1, -1):
        d = today - timedelta(days=k)
        if d.weekday() >= 5:
            continue
        a = datetime(d.year, d.month, d.day, 13, 25, tzinfo=timezone.utc)
        b = min(a + timedelta(minutes=460), datetime.now(timezone.utc).replace(second=0, microsecond=0))
        if b <= a:
            continue
        try:
            rows += cap.prices(epic, "MINUTE", a.strftime(fmt), b.strftime(fmt))
        except CapitalError:
            pass  # αργία ή πιο παλιά από όσο κρατά η Capital.com
    return parse(rows)


def regular_hours(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    m = df.index.hour * 60 + df.index.minute
    return df[(m >= 9 * 60 + 30) & (m < 16 * 60)]


# ───────────── Προσομοίωση με bid/ask ─────────────
@dataclass
class Trade:
    day: object
    side: int        # +1 αγορά, −1 πώληση
    entry: float
    stop: float
    exit: float
    reason: str
    spread: float

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def r(self) -> float:
        return self.side * (self.exit - self.entry) / self.risk if self.risk > 0 else 0.0

    @property
    def cost_r(self) -> float:
        return self.spread / self.risk if self.risk > 0 else 0.0


def run_day(day: pd.DataFrame, decide, max_trades: int = 10, last_entry=(15, 45), eod=(15, 55),
            slip: float = 0.0, delay: int = 0) -> list[Trade]:
    """decide(i) -> None ή (side, stop, target, max_bars) με stop/target ως ("level", τιμή) | ("dist", απόσταση)
    | ("r", πολλαπλάσιο του ρίσκου). Το σήμα βγαίνει στο κλείσιμο του κεριού i· η είσοδος γίνεται στο άνοιγμα
    του κεριού i+1+delay (αγορά στο ask, πώληση στο bid)."""
    rows = day.to_dict("records")
    idx = day.index
    n = len(rows)
    trades: list[Trade] = []
    pos = None
    pending = None
    entries = 0
    for i in range(n):
        bar, t = rows[i], idx[i]
        if pending is not None and pending[0] == i:
            _, side, stop_s, target_s, max_bars = pending
            pending = None
            spr = bar["oa"] - bar["ob"]
            entry = bar["oa"] + slip * spr if side > 0 else bar["ob"] - slip * spr
            stop = stop_s[1] if stop_s[0] == "level" else entry - side * stop_s[1]
            dist = side * (entry - stop)
            if dist > 0 and math.isfinite(dist):
                if target_s[0] == "level":
                    target = target_s[1]
                elif target_s[0] == "r":
                    target = entry + side * target_s[1] * dist
                else:
                    target = entry + side * target_s[1]
                if side * (target - entry) > 0:
                    pos = (side, entry, stop, target, i + max_bars, spr)
                    entries += 1
        if pos is not None:
            side, entry, stop, target, until, spr = pos
            ex = None
            if side > 0:
                if bar["ob"] <= stop:
                    ex = (bar["ob"] - slip * spr, "Stop")
                elif bar["lb"] <= stop:
                    ex = (stop - slip * spr, "Stop")
                elif bar["ob"] >= target:
                    ex = (bar["ob"], "Στόχος")
                elif bar["hb"] >= target:
                    ex = (target, "Στόχος")
            else:
                if bar["oa"] >= stop:
                    ex = (bar["oa"] + slip * spr, "Stop")
                elif bar["ha"] >= stop:
                    ex = (stop + slip * spr, "Stop")
                elif bar["oa"] <= target:
                    ex = (bar["oa"], "Στόχος")
                elif bar["la"] <= target:
                    ex = (target, "Στόχος")
            if ex is None and (i >= until or (t.hour, t.minute) >= eod or i == n - 1):
                ex = (bar["cb"] - slip * spr if side > 0 else bar["ca"] + slip * spr, "Χρόνος")
            if ex is not None:
                trades.append(Trade(t.date(), side, entry, stop, ex[0], ex[1], spr))
                pos = None
            continue
        if pending is not None or entries >= max_trades or (t.hour, t.minute) >= last_entry or i + 1 + delay >= n:
            continue
        sig = decide(i)
        if sig is not None:
            pending = (i + 1 + delay,) + tuple(sig)
    return trades


def _mid(day: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"o": (day["ob"] + day["oa"]) / 2, "h": (day["hb"] + day["ha"]) / 2,
                         "l": (day["lb"] + day["la"]) / 2, "c": (day["cb"] + day["ca"]) / 2,
                         "v": day["v"]}, index=day.index)


def _vwap(m: pd.DataFrame) -> pd.Series:
    tp = (m["h"] + m["l"] + m["c"]) / 3
    w = m["v"] if m["v"].sum() > 0 else pd.Series(1.0, index=m.index)
    return (tp * w).cumsum() / w.cumsum()


def _atr(m: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([m["h"] - m["l"], (m["h"] - m["c"].shift()).abs(), (m["l"] - m["c"].shift()).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _minutes(m: pd.DataFrame) -> np.ndarray:
    return np.asarray(m.index.hour * 60 + m.index.minute)


def strat_orb5(day: pd.DataFrame, **kw) -> list[Trade]:
    """Zarattini, Barbon & Aziz (2024): κατεύθυνση του πρώτου 5λεπτου κεριού, stop στην άλλη άκρη του,
    στόχος 10R (σπάνια) αλλιώς έξοδος στο κλείσιμο. Μία συναλλαγή τη μέρα."""
    m = _mid(day)
    first = m.iloc[:5]
    if len(first) < 5:
        return []
    o, c = first["o"].iloc[0], first["c"].iloc[-1]
    if c == o:
        return []
    side = 1 if c > o else -1
    stop = first["l"].min() if side > 0 else first["h"].max()
    return run_day(day, lambda i: (side, ("level", stop), ("r", 10.0), 10_000) if i == 4 else None,
                   max_trades=1, **kw)


def strat_orb15(day: pd.DataFrame, **kw) -> list[Trade]:
    """Σπάσιμο του εύρους του πρώτου 15λέπτου (9:30–9:45) έως τις 12:00: κλείσιμο λεπτού πάνω από το υψηλό
    → αγορά, κάτω από το χαμηλό → πώληση. Stop στη μέση του εύρους, στόχος 2R, αλλιώς έξοδος στο κλείσιμο."""
    m = _mid(day)
    mins = _minutes(m)
    rng = m[mins < 9 * 60 + 45]
    if len(rng) < 15:
        return []
    hi, lo = rng["h"].max(), rng["l"].min()
    midp = (hi + lo) / 2
    c = m["c"].to_numpy()

    def decide(i):
        if not (9 * 60 + 45 <= mins[i] < 12 * 60):
            return None
        if c[i] > hi:
            return (1, ("level", midp), ("r", 2.0), 10_000)
        if c[i] < lo:
            return (-1, ("level", midp), ("r", 2.0), 10_000)
        return None

    return run_day(day, decide, max_trades=1, **kw)


def _ema_scalp(day: pd.DataFrame, fast: int, slow: int, sl: float, tp: float, bars: int, **kw) -> list[Trade]:
    m = _mid(day)
    ef, es = m["c"].ewm(span=fast, adjust=False).mean(), m["c"].ewm(span=slow, adjust=False).mean()
    vwap, atr, mins = _vwap(m), _atr(m).to_numpy(), _minutes(m)
    up = ((ef > es) & (ef.shift() <= es.shift())).to_numpy()
    dn = ((ef < es) & (ef.shift() >= es.shift())).to_numpy()
    c, vw = m["c"].to_numpy(), vwap.to_numpy()

    def decide(i):
        if not (9 * 60 + 45 <= mins[i] <= 15 * 60 + 30) or i < slow:
            return None
        if up[i] and c[i] > vw[i]:
            return (1, ("dist", sl * atr[i]), ("dist", tp * atr[i]), bars)
        if dn[i] and c[i] < vw[i]:
            return (-1, ("dist", sl * atr[i]), ("dist", tp * atr[i]), bars)
        return None

    return run_day(day, decide, **kw)


def strat_ema_9_21(day, **kw):
    """Scalping ορμής 1′: διασταύρωση EMA 9/21 προς την πλευρά του VWAP, stop 1,5×ATR, στόχος 2×ATR, έως 20′."""
    return _ema_scalp(day, 9, 21, 1.5, 2.0, 20, **kw)


def strat_ema_5_13(day, **kw):
    """Πιο γρήγορο scalping: EMA 5/13 + VWAP, stop 1×ATR, στόχος 1,5×ATR, έως 10′."""
    return _ema_scalp(day, 5, 13, 1.0, 1.5, 10, **kw)


def _vwap_revert(day: pd.DataFrame, k: float, **kw) -> list[Trade]:
    m = _mid(day)
    vwap, mins = _vwap(m), _minutes(m)
    dev = m["c"] - vwap
    sd = dev.rolling(30, min_periods=20).std().to_numpy()
    d, vw = dev.to_numpy(), vwap.to_numpy()

    def decide(i):
        if not (10 * 60 <= mins[i] <= 15 * 60 + 30) or not (sd[i] > 0):
            return None
        if d[i] < -k * sd[i]:
            return (1, ("dist", 1.5 * sd[i]), ("level", vw[i]), 30)
        if d[i] > k * sd[i]:
            return (-1, ("dist", 1.5 * sd[i]), ("level", vw[i]), 30)
        return None

    return run_day(day, decide, **kw)


def strat_vwap_2(day, **kw):
    """Επιστροφή στο VWAP: απόκλιση πάνω από 2σ → συναλλαγή προς το VWAP, stop 1,5σ, έως 30′."""
    return _vwap_revert(day, 2.0, **kw)


def strat_vwap_25(day, **kw):
    """Ίδια με απόκλιση 2,5σ (λιγότερες, πιο ακραίες ευκαιρίες)."""
    return _vwap_revert(day, 2.5, **kw)


ALL_STRATS = {"orb5": ("ORB 5′ (Zarattini)", strat_orb5), "orb15": ("ORB 15′ σπάσιμο", strat_orb15),
              "ema921": ("EMA 9/21 + VWAP", strat_ema_9_21), "ema513": ("EMA 5/13 + VWAP", strat_ema_5_13),
              "vwap2": ("VWAP επιστροφή 2σ", strat_vwap_2), "vwap25": ("VWAP επιστροφή 2,5σ", strat_vwap_25)}
STRATS = {name: fn for name, fn in ALL_STRATS.values()}


def configure() -> str:
    """SCALP_DAYS, SCALP_INSTRUMENTS (π.χ. US100,US500), SCALP_STRATS (π.χ. orb5,orb15), SCALP_OUT (όνομα αναφοράς)."""
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
        STRATS = {ALL_STRATS[k][0]: ALL_STRATS[k][1] for k in keys}
    out = os.environ.get("SCALP_OUT", "").strip() or "scalp"
    if not out.replace("_", "").isalnum():
        raise ValueError(f"Άκυρο όνομα αναφοράς: {out}")
    return out


# ───────────── Στατιστικά και κρίση ─────────────
def summarize(trades: list[Trade]) -> dict:
    if not trades:
        return {"trades": 0, "avg_r": 0.0, "pf": 0.0}
    r = np.array([t.r for t in trades])
    curve = np.cumsum(r)
    peak = np.maximum.accumulate(np.concatenate([[0.0], curve]))[1:]
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    days = len({t.day for t in trades})
    return {"trades": int(len(r)), "per_day": len(r) / max(days, 1), "win": float((r > 0).mean()),
            "avg_r": float(r.mean()), "total_r": float(r.sum()),
            "pf": float(gains / losses) if losses > 0 else float("inf"),
            "max_dd_r": float((curve - peak).min()), "cost_r": float(np.mean([t.cost_r for t in trades])),
            "equity": float(np.prod(1 + RISK * r) - 1)}


def g(x: float, fmt: str) -> str:
    """Αριθμός με ελληνική υποδιαστολή."""
    return "∞" if x == float("inf") else format(x, fmt).replace(".", ",")


def verdict(res: dict) -> tuple[bool, list[str]]:
    """res: {"base","slip","delay": summary, "thirds": [summary×3]}. Επιστρέφει (αξιόπιστο, λόγοι αποτυχίας)."""
    why = []
    b = res["base"]
    if b["trades"] < MIN_TRADES:
        why.append(f"λίγες συναλλαγές ({b['trades']})")
    if b["avg_r"] <= 0:
        why.append("χάνει μετά το spread")
    elif b["pf"] < MIN_PF:
        why.append(f"profit factor {g(b['pf'], '.2f')}")
    if res["slip"]["avg_r"] <= 0:
        why.append("χάνει με ολίσθηση")
    if res["delay"]["avg_r"] <= 0:
        why.append("χάνει με 1′ καθυστέρηση")
    bad = [k + 1 for k, th in enumerate(res["thirds"]) if th["trades"] and th["avg_r"] <= 0]
    if bad:
        why.append("χάνει στο " + " και ".join(f"{k}ο" for k in bad) + " τρίτο")
    return (not why), why


HEAD = ("| Τεχνική | Συναλλαγές | Επιτυχία | Μέσο R | PF | Με ολίσθηση | Με 1′ καθυστέρηση | Τρίτα (μέσο R) | "
        "Spread (R) | Λογαριασμός (0,5% ρίσκο) | Κρίση |\n|---|---|---|---|---|---|---|---|---|---|---|")


def row(name: str, res: dict) -> str:
    b = res["base"]
    ok, why = verdict(res)
    if not b["trades"]:
        return f"| {name} | 0 | – | – | – | – | – | – | – | – | ✘ καμία συναλλαγή |"
    thirds = " / ".join(g(th["avg_r"], "+.2f") if th["trades"] else "–" for th in res["thirds"])
    return (f"| {name} | {b['trades']} ({g(b['per_day'], '.1f')}/μέρα) | {b['win'] * 100:.0f}% | "
            f"{g(b['avg_r'], '+.3f')} | {g(b['pf'], '.2f')} | {g(res['slip']['avg_r'], '+.3f')} | "
            f"{g(res['delay']['avg_r'], '+.3f')} | {thirds} | {g(b['cost_r'], '.2f')} | "
            f"{g(b['equity'] * 100, '+.1f')}% | {'✔ περνά' if ok else '✘ ' + '; '.join(why)} |")


def evaluate(df: pd.DataFrame) -> dict:
    """Όλες οι τεχνικές, όλα τα σενάρια, ανά τρίτο της περιόδου."""
    days = sorted(set(df.index.date))
    thirds = [set(x) for x in np.array_split(np.array(days, dtype=object), 3)]
    groups = [(d, day) for d, day in df.groupby(df.index.date) if len(day) >= 120]
    out = {}
    for name, fn in STRATS.items():
        res = {}
        for sc, kw in SCENARIOS.items():
            trades: list[Trade] = []
            for _, day in groups:
                trades += fn(day, **kw)
            res[sc] = summarize(trades)
            if sc == "base":
                res["thirds"] = [summarize([t for t in trades if t.day in th]) for th in thirds]
        out[name] = res
    return out


def main() -> None:
    report = configure()
    env = os.environ.get("CAPITAL_ENV", "demo")
    cap = Capital(os.environ["CAPITAL_API_KEY"], os.environ["CAPITAL_IDENTIFIER"], os.environ["CAPITAL_PASSWORD"], env)
    out = [f"# Έρευνα: scalping και γρήγορες ενδοσυνεδριακές συναλλαγές ({DAYS} ημέρες ιστορικού)", "",
           "Πραγματικά κεριά 1 λεπτού της Capital.com με bid/ask: κάθε αγορά στο ask, κάθε πώληση στο bid, άρα το "
           "spread είναι μέσα στο αποτέλεσμα. Μόνο κανονικές ώρες Νέας Υόρκης (9:30–16:00), καμία θέση το βράδυ. "
           "Αν stop και στόχος πιάνονται στο ίδιο λεπτό, μετράει το stop (συντηρητικά).", "",
           "**R** = πόσες φορές το ρίσκο της συναλλαγής· μέσο R πάνω από 0 = κερδίζει. **Με ολίσθηση**: μισό spread "
           "χειρότερη τιμή σε κάθε είσοδο και stop. **Με 1′ καθυστέρηση**: η είσοδος γίνεται ένα λεπτό αργότερα. "
           "**Τρίτα**: η περίοδος χωρισμένη σε τρία κομμάτια. **Spread (R)**: πόσο από το ρίσκο τρώει το spread.", "",
           f"**Αξιόπιστη** = τουλάχιστον {MIN_TRADES} συναλλαγές, profit factor ≥ {g(MIN_PF, '.2f')}, κερδίζει σε "
           "όλα τα σενάρια και σε κάθε τρίτο. Για να προχωρήσει σε demo, πρέπει να περνά σε τουλάχιστον 2 προϊόντα.",
           ""]
    summary: dict = {}
    passes: dict[str, list[str]] = {name: [] for name in STRATS}
    for epic, label in INSTRUMENTS.items():
        try:
            df = regular_hours(download(cap, epic, DAYS))
        except Exception as e:  # πρόβλημα με ένα σύμβολο δεν σταματά την έρευνα
            out += [f"### {label} ({epic})", "", f"Χωρίς δεδομένα: {e}", ""]
            continue
        if df.empty:
            out += [f"### {label} ({epic})", "", "Η Capital.com δεν έδωσε κεριά λεπτού.", ""]
            continue
        days = sorted(set(df.index.date))
        spread = float(((df["oa"] - df["ob"]) / ((df["oa"] + df["ob"]) / 2)).median())
        res = evaluate(df)
        summary[epic] = {"days": len(days), "from": str(days[0]), "to": str(days[-1]), "spread": spread,
                         "results": res}
        out += [f"### {label} ({epic})", "",
                f"Δεδομένα {days[0]:%d/%m/%Y} – {days[-1]:%d/%m/%Y}: {len(days)} μέρες, {len(df)} κεριά λεπτού. "
                f"Τυπικό spread {g(spread * 100, '.3f')}% της τιμής.", "", HEAD]
        for name, r in res.items():
            out.append(row(name, r))
            if verdict(r)[0]:
                passes[name].append(epic)
        out.append("")
    good = {n: e for n, e in passes.items() if len(e) >= 2}
    out += ["## Συμπέρασμα", ""]
    if good:
        out += [f"- **{n}**: περνά όλους τους ελέγχους σε {', '.join(e)} → υποψήφια για δοκιμή σε demo." for n, e in good.items()]
    else:
        out.append("- **Καμία τεχνική δεν πέρασε όλους τους ελέγχους σε 2 ή περισσότερα προϊόντα.** "
                   "Με αυτά τα δεδομένα, scalping με αυτές τις τεχνικές δεν είναι αξιόπιστο μετά τα κόστη.")
    single = {n: e for n, e in passes.items() if len(e) == 1}
    for n, e in single.items():
        out.append(f"- {n}: πέρασε μόνο σε {e[0]} (ένα προϊόν δεν αρκεί, μπορεί να είναι τύχη).")
    out.append("")
    path = ROOT / "reports"
    path.mkdir(exist_ok=True)
    (path / f"{report}.md").write_text("\n".join(out), encoding="utf-8")
    (path / f"{report}.json").write_text(json.dumps({"summary": summary, "passes": passes}, ensure_ascii=False,
                                                indent=1, default=float), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()

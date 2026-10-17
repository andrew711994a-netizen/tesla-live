"""Μεγάλη έρευνα: 5 ιδέες για καλύτερο σύνολο, πέρα από τους δείκτες.

Το bot (+17%/έτος στις μετοχές του, ~+10% σε άγνωστες) δεν βγάζει περισσότερα από την απλή κράτηση του Nasdaq-100
(+19%/έτος), απλώς πέφτει λιγότερο. Εδώ δοκιμάζουμε ιδέες με τεκμηρίωση στη χρηματοοικονομική έρευνα:
  1. Βάση + δορυφόρος: μέρος των χρημάτων στον δείκτη (μόνο όσο έχει ανοδική τάση), το υπόλοιπο στο bot.
  2. Μηνιαία εναλλαγή momentum: κάθε μήνα οι 8 δυνατότερες μετοχές των 12 μηνών (Jegadeesh–Titman).
  3. Ρίσκο ανάλογα με τη νευρικότητα της αγοράς (volatility targeting, Moreira–Muir / Barroso–Santa-Clara).
  4. Φίλτρα «περιβάλλοντος»: φόβος (VIX), ένταση στην αγορά (VIX/VIX3M), πιστωτική πίεση (HYG), άλμα επιτοκίων.
  5. Αγορά μετά από καλά αποτελέσματα με άλμα τιμής και όγκο (post-earnings drift).

Κάθε ιδέα κρίνεται όπως στο bot/research_params.py: καλύτερο Sharpe και στα τρία τεστ (μετοχές του bot 2016–21,
2022–σήμερα, 60 άγνωστες μετοχές 2016–σήμερα) χωρίς μεγαλύτερη πτώση, και μετά έλεγχος γειτονικών τιμών.
Δεν αλλάζει τίποτα στο bot.

Εκτέλεση:  python -m bot.research_ideas
Γράφει το reports/research_ideas.md και .json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import CONFIG, Costs, Risk, fills, num, pct, run
from .data import earnings_times, load_all
from .research_params import OTHER
from .strategy import Params, _ema, event_sessions

ROOT = Path(__file__).resolve().parent.parent
TESTS = [("A 2016–21", "A", "2016-01-01", "2021-12-31", "QQQ"),
         ("A 2022–σήμερα", "A", "2022-01-01", None, "QQQ"),
         ("B 2016–σήμερα", "B", "2016-01-01", None, "SPY")]
MACRO = ["^VIX", "^VIX3M", "HYG", "^TNX", "QQQ", "SPY"]


def curve_stats(eq: pd.Series, trades: int | None = None) -> dict:
    eq = eq.dropna()
    if len(eq) < 2:
        return {}
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    daily = eq.resample("D").last().dropna().pct_change().dropna()
    return {"cagr": float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1), "max_dd": float((eq / eq.cummax() - 1).min()),
            "sharpe": float(daily.mean() / daily.std() * math.sqrt(252)) if daily.std() > 0 else 0.0,
            "total": float(eq.iloc[-1] / eq.iloc[0] - 1), "trades": trades}


def window(idx: pd.DatetimeIndex, start: str, end: str | None) -> pd.DatetimeIndex:
    m = idx >= pd.Timestamp(start)
    if end:
        m &= idx <= pd.Timestamp(end)
    return idx[m]


# ───────────── 1. Δείκτης με φίλτρο τάσης ─────────────
def index_curve(px: pd.DataFrame, start: str, end: str | None, cost: float, ema_len: int | None) -> tuple[pd.Series, int]:
    """Κράτηση του δείκτη (ETF) όσο το κλείσιμο είναι πάνω από τον EMA· αλλαγή στο επόμενο άνοιγμα.
    ema_len=None: απλή αγορά και κράτηση."""
    c, o = px["Close"], px["Open"]
    want = (c > _ema(c, ema_len)) if ema_len else pd.Series(True, index=c.index)
    days = window(c.index, start, end)
    e, held, out, switches = 1.0, False, [], 0
    for d in days:
        i = c.index.get_loc(d)
        w = bool(want.iloc[i - 1]) if i > 0 else False
        if w and held:
            e *= c.iloc[i] / c.iloc[i - 1]
        elif w:
            e *= (1 - cost) * c.iloc[i] / o.iloc[i]
            held, switches = True, switches + 1
        elif held:
            e *= o.iloc[i] / c.iloc[i - 1] * (1 - cost)
            held, switches = False, switches + 1
        out.append(e)
    return pd.Series(out, index=days) * 10_000, switches


def mix(core: pd.Series, sat: pd.Series, w_core: float) -> pd.Series:
    """Δύο κομμάτια χωρίς εξισορρόπηση: w_core των χρημάτων στη βάση, τα υπόλοιπα στο bot."""
    core = core.reindex(sat.index).ffill().bfill()
    return 10_000 * (w_core * core / core.iloc[0] + (1 - w_core) * sat / sat.iloc[0])


# ───────────── 2. Μηνιαία εναλλαγή momentum ─────────────
def momentum_curve(univ: dict[str, pd.DataFrame], mkt_ok: pd.Series | None, start: str, end: str | None, cost: float,
                   top: int = 8, look: int = 252, skip: int = 21, absolute: bool = True) -> tuple[pd.Series, int]:
    """Την πρώτη μέρα κάθε μήνα (στο άνοιγμα): ίσα μερίδια στις top μετοχές με την καλύτερη απόδοση από look ως skip
    μέρες πριν. absolute: μόνο με θετική απόδοση. mkt_ok: όταν η αγορά δεν είναι σε τάση, όλα σε μετρητά."""
    closes = pd.DataFrame({s: df["Close"] for s, df in univ.items()}).sort_index()
    opens = pd.DataFrame({s: df["Open"] for s, df in univ.items()}).reindex(closes.index)
    alive = closes.notna()
    closes = closes.ffill()
    opens = opens.fillna(closes)
    mom = closes.shift(skip) / closes.shift(look) - 1
    days = window(closes.index, start, end)
    months = days.to_period("M")
    first = set(days[np.r_[True, months[1:] != months[:-1]]])
    cash, units, out, trades = 10_000.0, pd.Series(0.0, index=closes.columns), [], 0
    for d in days:
        i = closes.index.get_loc(d)
        if d in first and i > 0:
            op = opens.iloc[i]
            value = cash + float((units * op).sum())
            m = mom.iloc[i - 1].where(alive.iloc[i - 1])
            if absolute:
                m = m[m > 0]
            ok = True if mkt_ok is None else bool(mkt_ok.get(closes.index[i - 1], False))
            pick = list(m.dropna().sort_values(ascending=False).index[:top]) if ok else []
            new = pd.Series(0.0, index=closes.columns)
            for s in pick:
                new[s] = value / top / op[s]
            turnover = float((abs(new - units) * op).sum())
            trades += int(((new > 0) != (units > 0)).sum())
            cash = value - float((new * op).sum()) - turnover * cost
            units = new
        out.append(cash + float((units * closes.iloc[i]).sum()))
    return pd.Series(out, index=days), trades


# ───────────── 5. Αγορά μετά τα αποτελέσματα ─────────────
def pead_flags(df: pd.DataFrame, sg: pd.DataFrame, events: list[pd.Timestamp], gap: float) -> pd.Series:
    """Συνεδρίαση της αντίδρασης στα αποτελέσματα με άλμα ≥ gap στο άνοιγμα, που κράτησε ως το κλείσιμο, με όγκο
    ≥ 1,5× του μέσου 20 ημερών, πάνω από τον αργό EMA. Η αγορά γίνεται στο επόμενο άνοιγμα, όπως κάθε σήμα."""
    flag = pd.Series(False, index=df.index)
    vavg = df["Volume"].rolling(20).mean().shift(1)
    for e in events:
        if e not in df.index:
            continue
        i = df.index.get_loc(e)
        if i == 0:
            continue
        o, c, pc = df["Open"].iloc[i], df["Close"].iloc[i], df["Close"].iloc[i - 1]
        if (o / pc - 1 >= gap and c >= o and df["Volume"].iloc[i] >= 1.5 * vavg.iloc[i]
                and c > sg["ema_s"].iloc[i] and pd.notna(sg["atr"].iloc[i])):
            flag.iloc[i] = True
    return flag


def main() -> None:
    p = Params.from_dict(CONFIG["strategy"])
    rk = CONFIG["risk"]
    risk = Risk(rk["risk_per_trade"], rk["max_positions"], rk["max_total_risk"], rk["max_notional"], True)
    costs = Costs(**CONFIG["costs"])
    fl = fills()
    market_sym = CONFIG["market_symbol"]
    uni_a = list(CONFIG["universe"])
    uni_b = [s for s in OTHER if s not in uni_a]
    raw = {}
    for s in sorted(set(uni_a + uni_b + [market_sym] + MACRO)):
        try:
            raw.update(load_all([s]))
        except Exception as e:
            print(f"χωρίς δεδομένα {s}: {e}")
    market = raw[market_sym]
    mkt_ok = (market["Close"] > _ema(market["Close"], p.ema_slow)).fillna(False)
    U = {"A": {s: raw[s] for s in uni_a if s in raw},
         "B": {s: raw[s] for s in uni_b if s in raw and raw[s].index[0] <= pd.Timestamp("2015-01-01")}}

    cache: dict = {}

    def bot(t, **kw) -> tuple[pd.Series, int]:
        key = (t[0], repr(sorted((k, id(v) if not isinstance(v, (int, float, str)) else v) for k, v in kw.items())))
        if key not in cache:
            res = run(U[t[1]], p, risk, costs, t[2], t[3], market, **fl, **kw)
            cache[key] = (res["equity"], len(res["trades"]))
        return cache[key]

    base = {t[0]: curve_stats(*bot(t)) for t in TESTS}

    def evaluate(fn) -> dict:
        return {t[0]: curve_stats(*fn(t)) for t in TESTS}

    def verdict(res: dict) -> str:
        d_sh = [res[t]["sharpe"] - base[t]["sharpe"] for t, *_ in TESTS]
        d_dd = [res[t]["max_dd"] - base[t]["max_dd"] for t, *_ in TESTS]
        if all(x >= 0.05 for x in d_sh) and all(x >= -0.03 for x in d_dd):
            return "✅ καλύτερο και στα 3"
        if all(x <= -0.05 for x in d_sh):
            return "❌ χειρότερο και στα 3"
        return "➖ μικτό"

    # ── Οι ιδέες: (όνομα, συνάρτηση, [(γειτονική τιμή, συνάρτηση), ...]) ──
    c = costs.per_side
    ideas: list[tuple[str, list]] = []

    def core_mix(w, ema):
        return lambda t: (mix(index_curve(raw[t[4]], t[2], t[3], c, ema)[0], bot(t)[0], w), None)

    ideas.append(("1. Βάση (δείκτης) + δορυφόρος (bot)", [
        ("Δείκτης μόνο: αγορά και κράτηση (για σύγκριση)", lambda t: index_curve(raw[t[4]], t[2], t[3], c, None), []),
        ("Δείκτης μόνο, με φίλτρο τάσης EMA150", lambda t: index_curve(raw[t[4]], t[2], t[3], c, 150),
         [("EMA 100", lambda t: index_curve(raw[t[4]], t[2], t[3], c, 100)),
          ("EMA 200", lambda t: index_curve(raw[t[4]], t[2], t[3], c, 200))]),
        ("50% δείκτης με φίλτρο + 50% bot", core_mix(0.5, 150),
         [("40% δείκτης", core_mix(0.4, 150)), ("60% δείκτης", core_mix(0.6, 150)), ("EMA 200", core_mix(0.5, 200))]),
        ("70% δείκτης με φίλτρο + 30% bot", core_mix(0.7, 150),
         [("60% δείκτης", core_mix(0.6, 150)), ("80% δείκτης", core_mix(0.8, 150)), ("EMA 200", core_mix(0.7, 200))]),
        ("50% δείκτης χωρίς φίλτρο + 50% bot", core_mix(0.5, None), []),
    ]))

    def mom(top=8, look=252, absolute=True, filt=True):
        return lambda t: momentum_curve(U[t[1]], mkt_ok if filt else None, t[2], t[3], c, top, look, 21, absolute)

    ideas.append(("2. Μηνιαία εναλλαγή momentum (αντί για το bot)", [
        ("8 δυνατότερες, 12 μήνες, φίλτρο αγοράς", mom(),
         [("5 μετοχές", mom(top=5)), ("12 μετοχές", mom(top=12)), ("6 μήνες", mom(look=126)), ("9 μήνες", mom(look=189))]),
        ("8 δυνατότερες, 12 μήνες, χωρίς φίλτρο αγοράς", mom(filt=False), []),
        ("8 δυνατότερες, 6 μήνες, φίλτρο αγοράς", mom(look=126), [("5 μετοχές", mom(top=5, look=126)),
                                                                  ("12 μετοχές", mom(top=12, look=126))]),
    ]))

    nret = market["Close"].pct_change()

    def vol_scale(target, cap, win=20):
        vol = nret.rolling(win).std() * math.sqrt(252)
        return (target / vol).clip(upper=cap).shift(1).fillna(1.0)   # μέρα αγοράς ← νευρικότητα ως το χθεσινό κλείσιμο

    def volt(target, cap=1.0, win=20):
        sc = vol_scale(target, cap, win)
        return lambda t: bot(t, risk_scale=sc)

    ideas.append(("3. Ρίσκο ανάλογα με τη νευρικότητα της αγοράς", [
        ("Στόχος 20%/έτος, ποτέ πάνω από το σημερινό ρίσκο", volt(0.20),
         [("15%", volt(0.15)), ("25%", volt(0.25)), ("παράθυρο 60 ημερών", volt(0.20, 1.0, 60))]),
        ("Στόχος 15%/έτος, ποτέ πάνω από το σημερινό ρίσκο", volt(0.15),
         [("12%", volt(0.12)), ("18%", volt(0.18)), ("παράθυρο 60 ημερών", volt(0.15, 1.0, 60))]),
        ("Στόχος 20%/έτος, έως 1,5× όταν η αγορά είναι ήρεμη", volt(0.20, 1.5),
         [("έως 1,25×", volt(0.20, 1.25)), ("στόχος 18%", volt(0.18, 1.5)), ("στόχος 22%", volt(0.22, 1.5))]),
    ]))

    def ser(sym, col="Close"):
        return raw[sym][col] if sym in raw else None

    vix, vix3m, hyg, tnx = ser("^VIX"), ser("^VIX3M"), ser("HYG"), ser("^TNX")

    def allow_from(cond: pd.Series | None):
        if cond is None:
            return None
        sc = cond.reindex(market.index).ffill()
        return sc.where(sc.notna(), True).astype(bool)   # χωρίς δεδομένα (π.χ. πριν υπάρξει ο δείκτης): επιτρέπεται

    def filt(cond):
        al = allow_from(cond)
        return lambda t: bot(t, allow=al)

    macro = []
    if vix is not None:
        macro.append(("Όχι νέες αγορές όταν ο VIX > 30", filt(vix <= 30),
                      [("VIX > 25", filt(vix <= 25)), ("VIX > 35", filt(vix <= 35))]))
    if vix is not None and vix3m is not None:
        ratio = (vix / vix3m.reindex(vix.index)).dropna()
        macro.append(("Όχι νέες αγορές όταν VIX/VIX3M > 1 (ένταση)", filt(ratio <= 1.0),
                      [("> 0,95", filt(ratio <= 0.95)), ("> 1,05", filt(ratio <= 1.05))]))
    if hyg is not None:
        macro.append(("Όχι νέες αγορές όταν τα ομόλογα υψηλής απόδοσης (HYG) είναι κάτω από τον μέσο 50 ημερών",
                      filt(hyg > hyg.rolling(50).mean()),
                      [("μέσος 20", filt(hyg > hyg.rolling(20).mean())), ("μέσος 100", filt(hyg > hyg.rolling(100).mean()))]))
    if tnx is not None:
        macro.append(("Όχι νέες αγορές όταν το 10ετές επιτόκιο ανέβηκε > 0,40 μονάδες σε ένα μήνα", filt(tnx.diff(21) <= 0.40),
                      [("> 0,30", filt(tnx.diff(21) <= 0.30)), ("> 0,50", filt(tnx.diff(21) <= 0.50))]))
    ideas.append(("4. Φίλτρα περιβάλλοντος (φόβος, ένταση, πιστωτική πίεση, επιτόκια)", macro))

    # Αποτελέσματα εταιρειών (ημερομηνίες από το Yahoo)
    sessions = market.index
    events, missing_ev = {}, []
    for s in sorted(set(U["A"]) | set(U["B"])):
        try:
            events[s] = event_sessions(earnings_times(s), sessions)
        except Exception as e:
            missing_ev.append(s)
            print(f"χωρίς ημερομηνίες αποτελεσμάτων {s}: {e}")
    sig_cache: dict = {}

    def pead(gap, only=False):
        def mod(sym, sg):
            k = (sym, gap)
            if k not in sig_cache:
                df = raw[sym]
                sig_cache[k] = pead_flags(df, sg, events.get(sym, []), gap)
            f = sig_cache[k].reindex(sg.index, fill_value=False)
            sg = sg.copy()
            sg["long_sig"] = f if only else (sg["long_sig"].astype(bool) | f)
            return sg
        mod.__name__ = f"pead_{gap}_{only}"
        return lambda t: bot(t, sig_mod=mod)

    ideas.append(("5. Αγορά μετά από καλά αποτελέσματα (άλμα + όγκος)", [
        ("Επιπλέον σήμα: άλμα ≥ 3% μετά τα αποτελέσματα", pead(0.03), [("≥ 2%", pead(0.02)), ("≥ 5%", pead(0.05))]),
        ("Μόνο αυτό το σήμα (χωρίς τα σημερινά, για σύγκριση)", pead(0.03, only=True), []),
    ]))

    # ── Εκτέλεση ──
    summary, rows, passed = {"Σημερινό bot": base}, [], []
    for title, variants in ideas:
        block = []
        for name, fn, nbs in variants:
            res = evaluate(fn)
            summary[f"{title} | {name}"] = res
            v = verdict(res)
            nb_rows, robust = [], None
            if v.startswith("✅") and nbs:
                robust = True
                for desc, nfn in nbs:
                    nr = evaluate(nfn)
                    ok = all(nr[t]["sharpe"] >= base[t]["sharpe"] for t, *_ in TESTS)
                    robust &= ok
                    summary[f"{title} | {name} | γείτονας {desc}"] = nr
                    nb_rows.append((desc, nr, ok))
                if robust:
                    passed.append(f"{title.split('. ', 1)[1]}: {name}")
            block.append((name, res, v, nb_rows, robust))
            print(name, {k: round(x["sharpe"], 2) for k, x in res.items()}, v)
        rows.append((title, block))

    out = ["# Μεγάλη έρευνα: 5 ιδέες για καλύτερο σύνολο", "",
           f"Δεδομένα έως {market.index[-1]:%d/%m/%Y}. A = οι {len(U['A'])} μετοχές του bot· B = {len(U['B'])} μετοχές που "
           f"το bot δεν έχει δει ποτέ. Βάση για τον δείκτη: QQQ (Nasdaq-100) στα A, SPY (S&P 500) στο B, με μερίσματα. "
           f"Κόστος {c:.2%} ανά πράξη, στόχος/stop όπως στην Capital.com, χωρίς τόκο στα μετρητά.", "",
           "**Βελτίωση = καλύτερο Sharpe (κέρδος σε σχέση με τις διακυμάνσεις) και στα τρία τεστ, χωρίς μεγαλύτερη πτώση, "
           "και να κρατά με γειτονικές τιμές.** Ό,τι κερδίζει μόνο σε ένα ή δύο τεστ είναι μάλλον τύχη.", "",
           "| | " + " | ".join(f"{t}: ανά έτος / πτώση / Sharpe" for t, *_ in TESTS) + " |",
           "|---|" + "---|" * len(TESTS),
           "| **Σημερινό bot** | " + " | ".join(f"{pct(base[t]['cagr'])} / {pct(base[t]['max_dd'])} / {num(base[t]['sharpe'])}"
                                             for t, *_ in TESTS) + " |", ""]
    for title, block in rows:
        out += [f"## {title}", "", "| Παραλλαγή | " + " | ".join(f"{t}: ανά έτος / πτώση / Sharpe" for t, *_ in TESTS) +
                " | Ετυμηγορία |", "|---|" + "---|" * (len(TESTS) + 1)]
        for name, res, v, _, robust in block:
            cells = [f"{pct(res[t]['cagr'])} / {pct(res[t]['max_dd'])} / {num(res[t]['sharpe'])}" for t, *_ in TESTS]
            v2 = v + (" · κρατά με γειτονικές τιμές" if robust else " · ΔΕΝ κρατά με γειτονικές τιμές" if robust is False else "")
            out.append(f"| {name} | " + " | ".join(cells) + f" | {v2} |")
        nb = [(n, r) for n, _, _, r, _ in block if r]
        for n, nrows in nb:
            out += ["", f"Γειτονικές τιμές για «{n}» (Sharpe, πρέπει ≥ σημερινού και στα 3):", "",
                    "| Γειτονική τιμή | " + " | ".join(t for t, *_ in TESTS) + " | Κρατά; |",
                    "|---|" + "---|" * (len(TESTS) + 1)]
            for desc, nr, ok in nrows:
                out.append(f"| {desc} | " + " | ".join(num(nr[t]["sharpe"]) for t, *_ in TESTS) + f" | {'ναι' if ok else 'όχι'} |")
        out.append("")
    if missing_ev:
        out += [f"Χωρίς ημερομηνίες αποτελεσμάτων (η ιδέα 5 δεν ισχύει σε αυτές): {', '.join(missing_ev)}.", ""]
    out += ["## Συμπέρασμα", "",
            ("Πέρασαν όλους τους ελέγχους: " + "; ".join(passed) + ". Χρειάζονται την έγκρισή σου πριν μπουν στο bot."
             if passed else "Καμία ιδέα δεν πέρασε όλους τους ελέγχους."), "",
            "Σημείωση: όσο περισσότερα πράγματα δοκιμάζουμε, τόσο πιο εύκολα κάτι «περνά» από τύχη (σε τυχαία δεδομένα "
            "περίπου 2 στις 30 παραλλαγές). Γι' αυτό μετράμε μόνο ό,τι περνά και τα τρία τεστ ΚΑΙ τις γειτονικές τιμές.", ""]

    path = ROOT / "reports"
    path.mkdir(exist_ok=True)
    (path / "research_ideas.md").write_text("\n".join(out), encoding="utf-8")
    (path / "research_ideas.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float),
                                              encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()

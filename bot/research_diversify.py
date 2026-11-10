"""Έρευνα: διασπορά. Τα αδιάθετα χρήματα του bot σε χρυσό, ασήμι, ομόλογα, εμπορεύματα και δολάριο.

Το bot έχει κατά μέσο όρο το 20–35% του λογαριασμού αδιάθετο, και σχεδόν όλο όταν το φίλτρο αγοράς είναι κλειστό.
Εδώ δοκιμάζονται τρεις τρόποι να δουλεύουν αυτά τα χρήματα σε προϊόντα που δεν κινούνται μαζί με τις μετοχές
(ETF: τα αγοράζεις σαν μετοχές, οπότε στην Capital.com 1:1 δεν πρέπει να έχουν χρέωση νύχτας· ελέγχεται στο βήμα 0):

  Δ1. Οι ίδιοι κανόνες του bot (υποχώρηση/σπάσιμο, stop 2×ATR, στόχος 3×ATR, έως 20 μέρες) και σε αυτά τα προϊόντα.
      Δεν τα σταματά το φίλτρο του Nasdaq (έχουν το δικό τους φίλτρο τάσης). Οι μετοχές έχουν προτεραιότητα.
  Δ2. Όπως το Δ1, αλλά μετοχές και προϊόντα κατατάσσονται μαζί με την άνοδο 3 μηνών.
  Δ3. Κράτηση τάσης: όσο ένα προϊόν είναι σε ανοδική τάση (πάνω από τον μέσο 150 ημερών, και ο μέσος 50 πάνω
      από τον 150), το bot το κρατά με αδιάθετα χρήματα, έως 20% του λογαριασμού το καθένα. Όταν οι μετοχές
      χρειαστούν τα χρήματα, πουλιέται.
  Δ4. Όπως το Δ3, αλλά μόνο όσο το φίλτρο του Nasdaq είναι κλειστό, δηλαδή όταν το bot δεν αγοράζει μετοχές
      έτσι κι αλλιώς. Λιγότερες αγοραπωλησίες: τα προϊόντα δεν ανταγωνίζονται ποτέ τις μετοχές για τα χρήματα.
      (Προστέθηκε πριν τρέξει η έρευνα με πραγματικά δεδομένα: η δοκιμή με ψεύτικες τιμές έδειξε ότι το Δ3
      πληρώνει πολλά σε κόστος συναλλαγών, γιατί αλλάζει ποσά κάθε φορά που το bot αγοράζει ή πουλά μετοχή.)

Τα τεστ, το «με ίδιο ρίσκο» και ο κανόνας απόφασης είναι τα ίδια με το bot/research_edge.py, γραμμένα πριν τρέξει η
έρευνα: με ίδιο ρίσκο κερδίζει και στα τρία τεστ, p < 0,05 στα A και B μαζί, η χειρότερη πτώση δεν μεγαλώνει πάνω από
2 μονάδες, τα προϊόντα βγάζουν κέρδος και μόνα τους και στα δύο διαστήματα (2016–21, 2022–σήμερα), και οι γειτονικές
ρυθμίσεις (άλλα προϊόντα, άλλο όριο, άλλος μέσος) κερδίζουν κι αυτές.

Υποθέσεις των Δ3 και Δ4, όλες προς τη συντηρητική πλευρά: η τάση της μέρας t μετράει από το κλείσιμο της t+1 και μετά (η
αγορά γίνεται στο άνοιγμα της t+1 και το ενδιάμεσο κέρδος δεν μετριέται)· το ποσό είναι το μικρότερο αδιάθετο
ανάμεσα σε χθες και σήμερα· κάθε αλλαγή ποσού πληρώνει το κόστος του bot (0,07% ανά πράξη).

Δεν αλλάζει τίποτα στο bot. Με CAPITAL_API_KEY/IDENTIFIER/PASSWORD διαβάζει και τι προσφέρει η Capital.com (μόνο
ανάγνωση, καμία εντολή).

Εκτέλεση:  python -m bot.research_diversify
Γράφει το reports/research_diversify.md
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import CONFIG, Costs, Risk, every_signal, fills, num, pct, run, stats
from .data import load_all
from .research_edge import DD_TOL, KEYS, N_BOOT, PORT_HEAD, TESTS, cell, compare_curves, diff_cells, log_ret, verdict
from .research_ideas import curve_stats
from .research_params import OTHER
from .strategy import Params, _ema, market_ok

ROOT = Path(__file__).resolve().parent.parent
DIVERSIFIERS = {"GLD": "Χρυσός", "SLV": "Ασήμι", "TLT": "Ομόλογα ΗΠΑ 20+ ετών", "IEF": "Ομόλογα ΗΠΑ 7–10 ετών",
                "DBC": "Καλάθι εμπορευμάτων", "UUP": "Δολάριο έναντι άλλων νομισμάτων"}
SETS = {"μόνο χρυσός και ομόλογα": ["GLD", "TLT", "IEF"], "χωρίς ομόλογα": ["GLD", "SLV", "DBC", "UUP"]}
CAP_EACH = 0.20      # Δ3: έως τόσο του λογαριασμού σε κάθε προϊόν
FAST, SLOW = 50, 150  # Δ3: οι μέσοι της τάσης, όπως του bot
FEE_CHECK = 0.04     # ενημερωτικά: το Δ3 αν η Capital.com χρέωνε 4% τον χρόνο για τη νύχτα


# ───────────── 0. Τι προσφέρει η Capital.com ─────────────
def broker_check(symbols: list[str]) -> dict | None:
    """Για κάθε σύμβολο: υπάρχει στην Capital.com; τι τύπος είναι, τι ελάχιστο και τι spread έχει; Μόνο ανάγνωση."""
    keys = ("CAPITAL_API_KEY", "CAPITAL_IDENTIFIER", "CAPITAL_PASSWORD")
    if not all(os.environ.get(k) for k in keys):
        return None
    from .capital import Capital
    try:
        cap = Capital(*(os.environ[k] for k in keys), "demo")
    except Exception as e:
        print(f"Capital.com: {e}")
        return None
    out = {}
    for sym in symbols:
        info: dict = {"found": False}
        try:
            hits = cap.search(sym)
            info["candidates"] = [f"{m.get('epic')}: {m.get('instrumentName')} ({m.get('instrumentType')})" for m in hits[:4]]
            pick = next((m for m in hits if sym in (m.get("epic"), m.get("symbol"))), None)
            if pick:
                raw = cap._req("GET", f"/markets/{pick['epic']}")
                inst, rules, snap = raw.get("instrument", {}), raw.get("dealingRules", {}), raw.get("snapshot", {})
                fee = inst.get("overnightFee") or {}
                bid, offer = snap.get("bid"), snap.get("offer")
                info |= {"found": True, "epic": pick["epic"], "name": inst.get("name"), "type": inst.get("type"),
                         "currency": inst.get("currency"), "long_rate": fee.get("longRate"),
                         "min_size": (rules.get("minDealSize") or {}).get("value"),
                         "spread": (offer - bid) / offer if bid and offer else None}
        except Exception as e:
            info["error"] = str(e)[:160]
        out[sym] = info
    return out


# ───────────── Δ1/Δ2: οι κανόνες του bot και στα προϊόντα ─────────────
def mixed_mod(div_syms: set, mkt_ok: pd.Series, stocks_first: bool):
    """Το φίλτρο του Nasdaq ισχύει μόνο για τις μετοχές: σβήνει τα σήματά τους τις μέρες που είναι κλειστό, όπως
    ακριβώς κάνει το backtest. Τα προϊόντα διασποράς μένουν ελεύθερα· με stocks_first μπαίνουν τελευταία στη σειρά."""
    def mod(s: str, sg: pd.DataFrame) -> pd.DataFrame:
        if s in div_syms:
            return sg.assign(mom63=-1e9) if stocks_first else sg
        return sg.assign(long_sig=sg["long_sig"] & mkt_ok.reindex(sg.index).fillna(False).astype(bool))
    return mod


# ───────────── Δ3: κράτηση τάσης με τα αδιάθετα ─────────────
def trend_state(df: pd.DataFrame, fast: int = FAST, slow: int = SLOW) -> pd.Series:
    c = df["Close"]
    return ((c > _ema(c, slow)) & (_ema(c, fast) > _ema(c, slow))).fillna(False)


def sleeve_inputs(div: dict[str, pd.DataFrame], idx: pd.DatetimeIndex, fast: int, slow: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ημερήσιες αποδόσεις των προϊόντων και ποια κρατάμε κάθε μέρα (τάση δύο κλεισίματα πριν: δες docstring)."""
    ret = pd.DataFrame({s: df["Close"].reindex(idx).ffill().pct_change() for s, df in div.items()}).fillna(0.0)
    up = pd.DataFrame({s: trend_state(df, fast, slow).astype(float).reindex(idx).ffill() for s, df in div.items()})
    return ret, up.shift(2).fillna(0.0) > 0.5


def overlay(equity: pd.Series, cash: pd.Series, div: dict[str, pd.DataFrame], per_side: float,
            cap_each: float = CAP_EACH, fast: int = FAST, slow: int = SLOW, fee_annual: float = 0.0,
            active: pd.Series | None = None) -> dict:
    """Η αξία του λογαριασμού του bot αν τα αδιάθετα χρήματα κρατούσαν τα προϊόντα που είναι σε ανοδική τάση.
    active: μέρες (κλείσιμο) που επιτρέπεται η κράτηση, π.χ. όταν το φίλτρο του Nasdaq είναι κλειστό· ισχύει με
    την ίδια καθυστέρηση δύο κλεισιμάτων όπως η τάση."""
    idx = equity.index
    idle = (cash / equity).clip(0.0, 1.0)
    idle_eff = pd.concat([idle.shift(1), idle], axis=1).min(axis=1).fillna(0.0)   # ό,τι ήταν αδιάθετο και χθες και σήμερα
    ret, held = sleeve_inputs(div, idx, fast, slow)
    if active is not None:
        act = active.astype(float).reindex(idx).ffill().shift(2).fillna(0.0) > 0.5
        held = held.where(act, False, axis=0)
    n_up = held.sum(axis=1)
    each = np.minimum(cap_each, idle_eff / n_up.where(n_up > 0)).fillna(0.0)
    w = held.astype(float).mul(each, axis=0)
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()
    r = equity.pct_change().fillna(0.0) + (w * ret).sum(axis=1) - per_side * turnover - w.sum(axis=1) * fee_annual / 252
    return {"equity": equity.iloc[0] * (1 + r).cumprod(), "weights": w,
            "contrib": {s: float((w[s] * ret[s]).mean() * 252) for s in w.columns},
            "avg_weight": float(w.sum(axis=1).mean()), "cost": float(per_side * turnover.mean() * 252)}


def sleeve_alone(div: dict[str, pd.DataFrame], idx: pd.DatetimeIndex, per_side: float,
                 fast: int = FAST, slow: int = SLOW) -> pd.Series:
    """Τα προϊόντα μόνα τους: ίσα μερίδια σε όσα είναι σε ανοδική τάση, μετρητά όταν δεν είναι κανένα."""
    ret, held = sleeve_inputs(div, idx, fast, slow)
    n_up = held.sum(axis=1)
    w = held.astype(float).div(n_up.where(n_up > 0), axis=0).fillna(0.0)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    return (1 + (w * ret).sum(axis=1) - per_side * turnover).cumprod()


def main() -> None:
    p = Params.from_dict(CONFIG["strategy"])
    rk = CONFIG["risk"]
    risk = Risk(rk["risk_per_trade"], rk["max_positions"], rk["max_total_risk"], rk["max_notional"], True)
    risk_nf = replace(risk, use_market_filter=False)
    costs, fl = Costs(**CONFIG["costs"]), fills()
    market_sym = CONFIG["market_symbol"]
    uni_a = list(CONFIG["universe"])
    uni_b = [s for s in OTHER if s not in uni_a]
    raw = {}
    for s in sorted(set(uni_a + uni_b + [market_sym] + list(DIVERSIFIERS))):
        try:
            raw.update(load_all([s]))
        except Exception as e:
            print(f"χωρίς δεδομένα {s}: {e}")
    market = raw[market_sym]
    mkt = market_ok(market, p)
    U = {"A": {s: raw[s] for s in uni_a if s in raw},
         "B": {s: raw[s] for s in uni_b if s in raw and raw[s].index[0] <= pd.Timestamp("2015-01-01")}}

    # ── 0. Capital.com ──
    broker = broker_check(list(DIVERSIFIERS))
    usable = [s for s in DIVERSIFIERS if s in raw and (broker is None or broker[s]["found"])]
    div = {s: raw[s] for s in usable}
    summary: dict = {"data_to": f"{market.index[-1]:%Y-%m-%d}", "broker": broker, "usable": usable, "variants": {}}
    out = ["# Έρευνα: διασπορά με τα αδιάθετα χρήματα", "",
           f"Δεδομένα έως {market.index[-1]:%d/%m/%Y}. A = οι {len(U['A'])} μετοχές του bot· B = {len(U['B'])} μεγάλες "
           "μετοχές που το bot δεν έχει δει ποτέ. Ίδια όρια ρίσκου, κόστη και εκτέλεση με το bot.", "",
           "## 0. Τι προσφέρει η Capital.com", ""]
    if broker is None:
        out += ["⚠️ Δεν διαβάστηκε η Capital.com (λείπουν τα κλειδιά). Η έρευνα έτρεξε και με τα έξι προϊόντα· πριν "
                "από οποιαδήποτε χρήση πρέπει να ελεγχθεί ότι υπάρχουν και ότι δεν έχουν χρέωση νύχτας στο 1:1.", ""]
    else:
        out += ["| Προϊόν | Σύμβολο | Στην Capital.com | Τύπος | Νόμισμα | Ελάχιστο | Spread τώρα | Ποσοστό χρέωσης νύχτας (αγορά) |",
                "|---|---|---|---|---|---|---|---|"]
        for s, label in DIVERSIFIERS.items():
            b = broker[s]
            if b["found"]:
                out.append(f"| {label} | {s} | {b.get('name') or b['epic']} | {b.get('type') or '–'} | {b.get('currency') or '–'} | "
                           f"{b.get('min_size') if b.get('min_size') is not None else '–'} | "
                           f"{pct(b['spread']) if b.get('spread') is not None else '–'} | "
                           f"{b.get('long_rate') if b.get('long_rate') is not None else '–'} |")
            else:
                near = "; ".join(b.get("candidates") or []) or b.get("error") or "κανένα αποτέλεσμα"
                out.append(f"| {label} | {s} | ✘ δεν βρέθηκε ({near}) | – | – | – | – | – |")
        out += ["", "Η Capital.com λέει ότι δεν χρεώνει τη νύχτα σε μετοχές χωρίς μόχλευση (1:1). Αν ο τύπος ενός προϊόντος "
                "δεν είναι SHARES, ή το ποσοστό χρέωσης δεν είναι μηδέν, **ρώτησε την υποστήριξη αν ισχύει το ίδιο και "
                "για αυτό** πριν βασιστούμε εδώ. Στα τεστ μπήκαν μόνο όσα βρέθηκαν: " +
                (", ".join(usable) if usable else "κανένα") + ".", ""]
    if not div:
        out += ["Κανένα προϊόν διασποράς δεν είναι διαθέσιμο, οπότε η έρευνα σταματά εδώ.", ""]
        _write(out, summary)
        return

    # ── Σημερινό bot ──
    base = {}
    for label, u, start, end in TESTS:
        res = run(U[u], p, risk, costs, start, end, market, **fl)
        base[label] = {"stats": stats(res), "lr": log_ret(res["equity"]), "equity": res["equity"], "cash": res["cash"],
                       "invested": float(1 - (res["cash"] / res["equity"]).clip(0, 1).mean())}

    def mixed(names: list[str], stocks_first: bool) -> dict:
        d = {s: raw[s] for s in names}
        mod = mixed_mod(set(d), mkt, stocks_first)
        res_by = {}
        for label, u, start, end in TESTS:
            res = run({**U[u], **d}, p, risk_nf, costs, start, end, None, sig_mod=mod, **fl)
            res_by[label] = {"stats": stats(res), "lr": log_ret(res["equity"]),
                             "invested": float(1 - (res["cash"] / res["equity"]).clip(0, 1).mean())}
        return res_by

    def held(names: list[str], cap_each: float = CAP_EACH, fast: int = FAST, slow: int = SLOW, fee: float = 0.0,
             only_off: bool = False) -> dict:
        d = {s: raw[s] for s in names}
        res_by = {}
        for label, *_ in TESTS:
            ov = overlay(base[label]["equity"], base[label]["cash"], d, costs.per_side, cap_each, fast, slow, fee,
                         ~mkt if only_off else None)
            res_by[label] = {"stats": curve_stats(ov["equity"], base[label]["stats"]["trades"]), "lr": log_ret(ov["equity"]),
                             "invested": base[label]["invested"] + ov["avg_weight"], "detail": ov}
        return res_by

    check = compare_curves(mixed([], True), base, 200)
    exact = all(abs(v) < 1e-12 for v in check["raw_tests"].values())

    # ── 1. Τα προϊόντα μόνα τους ──
    span = {"2016–21": ("2016-01-01", "2021-12-31"), "2022–σήμερα": ("2022-01-01", None)}
    idx_all = market.index[market.index >= pd.Timestamp("2016-01-01")]
    alone = sleeve_alone(div, idx_all, costs.per_side)

    def cut(sr: pd.Series, start: str, end: str | None) -> pd.Series:
        sr = sr[sr.index >= pd.Timestamp(start)]
        return sr[sr.index <= pd.Timestamp(end)] if end else sr

    alone_st = {k: curve_stats(cut(alone, *v)) for k, v in span.items()}
    alone_ok = all(st.get("cagr", 0) > 0 for st in alone_st.values())
    bot_ret = pd.concat([base[TESTS[0][0]]["lr"], base[TESTS[1][0]]["lr"]]).sort_index()
    corr = float(np.log(alone).diff().reindex(bot_ret.index).corr(bot_ret))
    res = run(div, p, every_signal(risk_nf), costs, "2016-01-01", None, None, **fl)
    tr = pd.DataFrame([{"symbol": t["symbol"], "r": t["r"], "pct": t["pnl"] / (t["qty"] * t["entry"]),
                        "entry": t["entry_date"]} for t in res["trades"]])
    out += ["| | " + " | ".join(f"{t}: ανά έτος / πτώση / Sharpe" for t, *_ in TESTS) + " |", "|---|" + "---|" * len(TESTS),
            "| **Σημερινό bot** | " + " | ".join(cell(base[t]["stats"]) for t, *_ in TESTS) + " |", "",
            f"Αδιάθετα χρήματα του bot κατά μέσο όρο: " +
            " / ".join(f"{(1 - base[t]['invested']) * 100:.0f}%" for t, *_ in TESTS) + " (" + " / ".join(t for t, *_ in TESTS) + ").",
            f"Έλεγχος ακρίβειας: ο μηχανισμός «φίλτρο μόνο για τις μετοχές» δίνει {'ακριβώς το ίδιο' if exact else 'ΔΙΑΦΟΡΕΤΙΚΟ'} "
            "αποτέλεσμα με το backtest του bot όταν δεν προστεθεί κανένα προϊόν.", "",
            "## 1. Τα προϊόντα μόνα τους", "",
            "**Κράτηση τάσης** (ίσα μερίδια σε όσα είναι σε ανοδική τάση, μετρητά όταν δεν είναι κανένα):", "",
            "| | Ανά έτος | Χειρότερη πτώση | Sharpe |", "|---|---|---|---|"]
    for k, st in alone_st.items():
        out.append(f"| {k} | {pct(st['cagr'])} | {pct(st['max_dd'])} | {num(st['sharpe'])} |" if st else f"| {k} | – | – | – |")
    out += ["", f"Συσχέτιση με τις ημερήσιες αποδόσεις του bot (A): {num(corr)} (0 = άσχετα, 1 = κινούνται μαζί).", "",
            "**Οι κανόνες του bot σε αυτά τα προϊόντα**, κάθε σήμα σαν να αγοράστηκε (R = φορές το ρίσκο):", "",
            "| | Σήματα | Επιτυχία | Μέσο R | Μέσο αποτέλεσμα |", "|---|---|---|---|---|"]
    sig_ok, sig_facts = False, {}
    if not tr.empty:
        on = mkt.shift(1).reindex(tr["entry"]).fillna(False).to_numpy(dtype=bool)
        halves = {k: tr[(tr["entry"] >= pd.Timestamp(a)) & ((tr["entry"] <= pd.Timestamp(b)) if b else True)]
                  for k, (a, b) in span.items()}
        parts = ([("Όλα", tr)] + [(f"{DIVERSIFIERS[s]} ({s})", tr[tr["symbol"] == s]) for s in div] +
                 [("Όταν το φίλτρο του Nasdaq είναι ανοιχτό", tr[on]), ("Όταν είναι κλειστό", tr[~on])] + list(halves.items()))
        for name, part in parts:
            if part.empty:
                continue
            sig_facts[name] = {"n": len(part), "win": float((part["r"] > 0).mean()), "avg_r": float(part["r"].mean()),
                               "avg_pct": float(part["pct"].mean())}
            f = sig_facts[name]
            out.append(f"| {name} | {f['n']} | {f['win'] * 100:.0f}% | {num(f['avg_r'])} | {pct(f['avg_pct'])} |")
        sig_ok = all(len(h) > 0 and h["r"].mean() > 0 for h in halves.values())
    summary |= {"alone": alone_st, "corr_with_bot": corr, "signals": sig_facts}

    # ── 2. Χαρτοφυλάκιο ──
    variants = [("d1", "Δ1. Κανόνες του bot και στα προϊόντα, οι μετοχές πρώτες", lambda n: mixed(n, True), sig_ok),
                ("d2", "Δ2. Κανόνες του bot, μετοχές και προϊόντα μαζί στη σειρά", lambda n: mixed(n, False), sig_ok),
                ("d3", f"Δ3. Κράτηση τάσης με τα αδιάθετα, έως {CAP_EACH * 100:.0f}% το καθένα", lambda n: held(n), alone_ok),
                ("d4", "Δ4. Κράτηση τάσης μόνο όταν το φίλτρο του Nasdaq είναι κλειστό", lambda n: held(n, only_off=True), alone_ok)]
    out += ["", "## 2. Χαρτοφυλάκιο: το bot μαζί με τα προϊόντα", "", PORT_HEAD]
    passed, curves = [], {}
    for key, label, fn, ok_alone in variants:
        curves[key] = fn(usable)
        cmp = compare_curves(curves[key], base, N_BOOT)
        v, ok = verdict(cmp, ok_alone, len(variants))
        v = v.replace("δεν συμφωνεί η μέτρηση ανά σήμα", "τα προϊόντα δεν βγάζουν κέρδος μόνα τους και στα δύο διαστήματα")
        summary["variants"][key] = {"label": label, "verdict": v, **{k: cmp[k] for k in KEYS}}
        if ok:
            passed.append((key, label, fn))
        out.append(f"| {label} | " + " | ".join(cell(cmp["stats"][t]) for t, *_ in TESTS) + f" | {diff_cells(cmp)} | {v} |")
    for key, only_off in (("d3", False), ("d4", True)):
        fee_cmp = compare_curves(held(usable, fee=FEE_CHECK, only_off=only_off), base, N_BOOT)
        summary["variants"][f"{key}_fee"] = {k: fee_cmp[k] for k in KEYS}
        out.append(f"| Ενημερωτικά: το Δ{key[1]} αν υπήρχε χρέωση νύχτας {FEE_CHECK * 100:.0f}% τον χρόνο | "
                   + " | ".join(cell(fee_cmp["stats"][t]) for t, *_ in TESTS) + f" | {diff_cells(fee_cmp)} | – |")
    out += ["", "Μεταβολή της χειρότερης πτώσης σε σχέση με το σημερινό bot (θετικό = μικρότερη πτώση): " +
            "; ".join(f"{label.split('.')[0]}: " + " / ".join(
                pct(summary['variants'][key]['stats'][t]['max_dd'] - base[t]['stats']['max_dd']) for t, *_ in TESTS)
                for key, label, *_ in variants) + ".", ""]

    for key in ("d3", "d4"):
        dk = curves[key]
        out += [f"**Το Δ{key[1]} από μέσα** (" + " / ".join(t for t, *_ in TESTS) + "):", "",
                "- Μέσο ποσό σε προϊόντα διασποράς: " + " / ".join(f"{dk[t]['detail']['avg_weight'] * 100:.0f}%" for t, *_ in TESTS) + " του λογαριασμού.",
                "- Κόστος συναλλαγών του: " + " / ".join(pct(-dk[t]['detail']['cost']) for t, *_ in TESTS) + " τον χρόνο.",
                "- Συνεισφορά κάθε προϊόντος στην απόδοση του λογαριασμού, ανά έτος: " +
                "; ".join(f"{DIVERSIFIERS[s]} " + " / ".join(pct(dk[t]['detail']['contrib'][s]) for t, *_ in TESTS) for s in usable) + ".", ""]
        summary["variants"][key]["detail"] = {t: {k: dk[t]["detail"][k] for k in ("contrib", "avg_weight", "cost")} for t, *_ in TESTS}

    # ── 3. Γειτονικές ρυθμίσεις για όσες πέρασαν ──
    robust = []
    if passed:
        out += ["## 3. Γειτονικές ρυθμίσεις για όσες πέρασαν", "",
                "Κάθε γειτονική ρύθμιση πρέπει να κερδίζει το σημερινό bot με ίδιο ρίσκο στα A και B μαζί, χωρίς μεγαλύτερη πτώση.", "",
                "| Παραλλαγή | Γειτονική ρύθμιση | Σκέτη διαφορά ανά έτος (A και B μαζί) | Με ίδιο ρίσκο: "
                + " / ".join(t for t, *_ in TESTS) + " | Με ίδιο ρίσκο, A και B μαζί (εύρος 90%) | p | Κρατά; |",
                "|---|---|---|---|---|---|---|"]
        for key, label, fn in passed:
            nbs = [(name, lambda names=names: fn(names)) for name, names in
                   ((n, [s for s in syms if s in usable]) for n, syms in SETS.items()) if names and names != usable]
            if key in ("d3", "d4"):
                off = key == "d4"
                nbs += [("έως 15% το καθένα", lambda off=off: held(usable, cap_each=0.15, only_off=off)),
                        ("έως 25% το καθένα", lambda off=off: held(usable, cap_each=0.25, only_off=off)),
                        ("μέσος 100 ημερών", lambda off=off: held(usable, slow=100, only_off=off)),
                        ("μέσος 200 ημερών", lambda off=off: held(usable, slow=200, only_off=off))]
            all_ok = True
            for nb_label, nb_fn in nbs:
                cmp = compare_curves(nb_fn(), base, N_BOOT)
                ok = cmp["pooled"]["diff"] > 0 and cmp["dd_ok"]
                all_ok &= ok
                summary["variants"][key].setdefault("neighbors", {})[nb_label] = {k: cmp[k] for k in ("tests", "pooled", "raw", "dd_ok")}
                out.append(f"| {label.split('.')[0]} | {nb_label} | {diff_cells(cmp)} | {'ναι' if ok else 'όχι'} |")
            summary["variants"][key]["robust"] = all_ok
            if all_ok:
                robust.append(label)
        out.append("")

    out += ["## Συμπέρασμα", ""]
    if robust:
        out.append("Πέρασαν όλους τους ελέγχους: **" + ", ".join(robust) + "**. Χρειάζονται την έγκρισή σου, και νέο κώδικα "
                   "στο bot, πριν μπουν.")
    elif passed:
        out.append("Πέρασαν τα τεστ αλλά όχι τις γειτονικές ρυθμίσεις: " + ", ".join(x[1] for x in passed) + ". Δεν αρκεί για αλλαγή στο bot.")
    else:
        out.append("**Καμία παραλλαγή δεν πέρασε όλους τους ελέγχους.** Το «δεν αποδείχθηκε» δεν σημαίνει ότι είναι "
                   "χειρότερη· σημαίνει ότι με τα δεδομένα που έχουμε δεν ξεχωρίζει από την τύχη.")
    out += ["", f"Σημείωση: το όριο πτώσης είναι {DD_TOL * 100:.0f} μονάδες. Τα ίδια προϊόντα μπαίνουν και στα τρία τεστ, "
            "οπότε τα τεστ δεν είναι εντελώς ανεξάρτητα μεταξύ τους όπως στις έρευνες με μετοχές.", ""]
    summary |= {"passed": [x[1] for x in passed], "robust": robust}
    _write(out, summary)


def _write(out: list[str], summary: dict) -> None:
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items() if k not in ("boot", "weights", "equity")}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return float(o) if isinstance(o, (np.floating, np.integer)) else o

    path = ROOT / "reports"
    path.mkdir(exist_ok=True)
    (path / "research_diversify.md").write_text("\n".join(out), encoding="utf-8")
    (path / "research_diversify.json").write_text(json.dumps(clean(summary), ensure_ascii=False, indent=1, default=float),
                                                  encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()

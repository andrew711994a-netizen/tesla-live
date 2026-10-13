"""Μεγάλο τεστ παραμέτρων και δεικτών (MACD, ADX, Fibonacci, RSI, σπάσιμο, έξοδοι, κατάταξη, θέσεις).

Κάθε παραλλαγή αλλάζει ΜΙΑ ρύθμιση σε σχέση με το σημερινό bot και κρίνεται σε τρία ανεξάρτητα τεστ:
  1. τις 49 μετοχές του bot, 2016–2021 (τα χρόνια στα οποία φτιάχτηκε η στρατηγική),
  2. τις ίδιες μετοχές, 2022–σήμερα (χρόνια που η στρατηγική «δεν είχε δει»),
  3. 60 ΑΛΛΕΣ μεγάλες αμερικανικές μετοχές που το bot δεν έχει δει ποτέ, 2016–σήμερα.
Μια αλλαγή θεωρείται πραγματική βελτίωση μόνο αν είναι καλύτερη και στα τρία· αλλιώς μάλλον είναι τύχη
(όσο περισσότερα πράγματα δοκιμάζουμε, τόσο πιο εύκολα κάτι «κερδίζει» κατά τύχη).

Δεν αλλάζει τίποτα στο bot: οι νέοι δείκτες και έξοδοι υπάρχουν μόνο εδώ.

Εκτέλεση:  python -m bot.research_params
Γράφει το reports/research_params.md
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import CONFIG, Costs, Risk, num, pct, run, stats
from .data import load_all
from .strategy import Params, _ema, _rma, market_ok, signals

ROOT = Path(__file__).resolve().parent.parent

OTHER = ["LOW", "TGT", "UNP", "UPS", "RTX", "LMT", "MMM", "DE", "ADP", "AMGN", "GILD", "BMY", "TMO", "DHR",
         "ABT", "MDT", "ISRG", "SYK", "CVS", "CI", "ELV", "SCHW", "MS", "C", "WFC", "USB", "PNC", "AXP", "BLK",
         "SPGI", "MCO", "ICE", "CME", "INTU", "NOW", "AMAT", "LRCX", "KLAC", "ADI", "SNPS", "CDNS", "BKNG",
         "MAR", "CMG", "TJX", "ROST", "ORLY", "AZO", "SO", "DUK", "AEP", "SLB", "COP", "EOG", "LIN", "APD",
         "SHW", "ECL", "NSC", "CSX"]


# ── Δείκτες ──
def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    up, dn = h.diff(), -l.diff()
    plus = up.where((up > dn) & (up > 0), 0.0)
    minus = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = _rma(tr, n)
    pdi, mdi = 100 * _rma(plus, n) / atr, 100 * _rma(minus, n) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return _rma(dx, n)


def signals_ext(df: pd.DataFrame, p: Params, o: dict) -> pd.DataFrame:
    """Τα σήματα του bot, με προαιρετικούς επιπλέον δείκτες (o)."""
    sg = signals(df, p)
    c, h, l, op = df["Close"], df["High"], df["Low"], df["Open"]
    up = (c > sg["ema_s"]) & (sg["ema_f"] > sg["ema_s"])
    ok_atr = sg["atr"].notna()
    long = sg["long_sig"].astype(bool).copy()
    kind = sg["kind"].copy()

    macd = _ema(c, 12) - _ema(c, 26)
    msig = _ema(macd, 9)
    if o.get("macd_entry"):                        # επιπλέον είσοδος: MACD περνά πάνω από τη γραμμή σήματος
        add = up & (macd > msig) & (macd.shift() <= msig.shift()) & ok_atr
        kind = kind.where(long | ~add, "macd")
        long |= add.fillna(False)
    if o.get("fib_entry"):                         # επιπλέον είσοδος: υποχώρηση στη ζώνη Fibonacci 38,2–61,8%
        n = int(o.get("fib_len", 60))
        hi, lo = h.rolling(n).max(), l.rolling(n).min()
        lvl38, lvl62 = hi - 0.382 * (hi - lo), hi - 0.618 * (hi - lo)
        touch = (l <= lvl38) & (c >= lvl62) & (c > op) & ~(l.shift() <= lvl38.shift())
        add = (up & touch & ok_atr).fillna(False)
        kind = kind.where(long | ~add, "fib")
        long |= add
    if o.get("macd_filter") == "above_signal":
        long &= (macd > msig).fillna(False)
    if o.get("macd_filter") == "above_zero":
        long &= (macd > 0).fillna(False)
    if o.get("adx_min"):
        long &= (adx(df) > o["adx_min"]).fillna(False)

    sg["long_sig"] = long.fillna(False)
    sg["kind"] = kind
    sg["mom126"] = c / c.shift(126) - 1
    sg["mom21"] = c / c.shift(21) - 1
    return sg


# ── Backtest με επιπλέον επιλογές εξόδου/κατάταξης (ίδιο με το backtest.run όταν δεν δίνονται) ──
def run_ext(data, sigs, p: Params, risk: Risk, costs: Costs, start, end=None, market=None,
            trail: float = 0.0, no_target: bool = False, breakeven: float = 0.0, rank: str = "mom63",
            initial: float = 10_000.0) -> dict:
    mkt = market_ok(market, p) if (risk.use_market_filter and market is not None) else None
    end_ts = pd.Timestamp(end) if end else None
    dates = sorted(set().union(*[df.index for df in data.values()]))
    dates = [d for d in dates if d >= pd.Timestamp(start) and (end_ts is None or d <= end_ts)]
    cash, equity = initial, initial
    positions, pending, trades, curve = {}, [], [], []
    last_close, prev_d, days_invested = {}, None, 0

    def close_pos(s, px, d, reason):
        nonlocal cash
        pos = positions.pop(s)
        exit_cost = pos["qty"] * px * costs.per_side
        cash += pos["qty"] * px - exit_cost
        pnl = pos["qty"] * (px - pos["entry"]) - pos["entry_cost"] - exit_cost - pos["fin"]
        trades.append({"symbol": s, "entry_date": pos["entry_date"], "exit_date": d, "entry": pos["entry"],
                       "exit": px, "qty": pos["qty"], "pnl": pnl, "r": pnl / (pos["qty"] * pos["dist"]),
                       "reason": reason})

    def check_intraday(s, row, d):
        pos = positions[s]
        o, h, l = row["Open"], row["High"], row["Low"]
        if pos["entry_date"] != d and o <= pos["stop"]:
            close_pos(s, o, d, "Stop"); return True
        if l <= pos["stop"]:
            close_pos(s, pos["stop"], d, "Stop"); return True
        if pos["entry_date"] != d and o >= pos["tp"]:
            close_pos(s, o, d, "Στόχος"); return True
        if h >= pos["tp"]:
            close_pos(s, pos["tp"], d, "Στόχος"); return True
        return False

    for d in dates:
        for s in list(positions):
            df = data[s]
            if d not in df.index:
                continue
            row = df.loc[d]
            if positions[s]["exit_pending"]:
                close_pos(s, row["Open"], d, positions[s]["exit_pending"])
            else:
                check_intraday(s, row, d)
        for s, stop, tp, dist in pending:
            if s in positions or len(positions) >= risk.max_positions or d not in data[s].index:
                continue
            row = data[s].loc[d]
            o = row["Open"]
            if not (stop < o < tp) or dist <= 0:
                continue
            dist = o - stop   # όπως το bot: το ρίσκο μετριέται από την τιμή αγοράς ως το stop
            eq = equity
            qty = eq * risk.risk_per_trade / dist
            notional_used = sum(ps["qty"] * last_close.get(x, ps["entry"]) for x, ps in positions.items())
            qty = min(qty, max(0.0, risk.max_notional * eq - notional_used) / o)
            risk_used = sum(ps["qty"] * ps["dist"] for ps in positions.values())
            qty = min(qty, max(0.0, risk.max_total_risk * eq - risk_used) / dist)
            if qty * o < 0.02 * eq:
                continue
            entry_cost = qty * o * costs.per_side
            cash -= qty * o + entry_cost
            positions[s] = {"qty": qty, "entry": o, "stop": stop, "tp": math.inf if no_target else tp,
                            "dist": dist, "entry_date": d, "entry_cost": entry_cost, "fin": 0.0, "bars": 0,
                            "exit_pending": "", "hi": o}
            check_intraday(s, row, d)
        pending = []

        nights = (d - prev_d).days if prev_d is not None else 1
        for s, pos in positions.items():
            if d in data[s].index:
                last_close[s] = data[s].at[d, "Close"]
            px = last_close.get(s, pos["entry"])
            fin = pos["qty"] * px * costs.financing_annual * nights / 365
            cash -= fin
            pos["fin"] += fin
            if d not in sigs[s].index:
                continue
            # Κινούμενο stop (ισχύει από το επόμενο άνοιγμα, όπως μια εντολή που αλλάζει μετά το κλείσιμο)
            pos["hi"] = max(pos["hi"], px)
            if trail:
                pos["stop"] = max(pos["stop"], pos["hi"] - trail * sigs[s].at[d, "atr"])
            if breakeven and px >= pos["entry"] + breakeven * pos["dist"]:
                pos["stop"] = max(pos["stop"], pos["entry"])
            if pos["entry_date"] != d:
                pos["bars"] += 1
            if pos["bars"] >= p.max_bars:
                pos["exit_pending"] = "Χρόνος"
            elif p.trend_exit and bool(sigs[s].at[d, "trend_broken"]):
                pos["exit_pending"] = "Τάση"
        for s in data:
            if d in data[s].index:
                last_close[s] = data[s].at[d, "Close"]
        equity = cash + sum(ps["qty"] * last_close.get(s, ps["entry"]) for s, ps in positions.items())
        curve.append((d, equity, cash))
        if positions:
            days_invested += 1
        allowed = True if mkt is None else bool(mkt.get(d, False))
        if allowed:
            cands = []
            for s, sg in sigs.items():
                if s in positions or d not in sg.index or not bool(sg.at[d, "long_sig"]):
                    continue
                key = sg.at[d, rank]
                cands.append((key if not math.isnan(key) else -9, s, sg.at[d, "stop"], sg.at[d, "tp"],
                              p.sl_atr * sg.at[d, "atr"]))
            cands.sort(reverse=True)
            pending = [(s, stop, tp, dist) for _, s, stop, tp, dist in cands]
        prev_d = d

    if dates:
        for s in list(positions):
            close_pos(s, last_close.get(s, positions[s]["entry"]), dates[-1], "Τέλος")
    idx = [d for d, _, _ in curve]
    return {"trades": trades, "equity": pd.Series([e for _, e, _ in curve], index=idx, dtype=float),
            "initial": initial, "exposure": days_invested / max(len(dates), 1)}


# ── Παραλλαγές: (όνομα, αλλαγές στο Params, επιπλέον δείκτες, επιλογές backtest, αλλαγές στο Risk) ──
VARIANTS = [
    ("Σημερινό bot", {}, {}, {}, {}),
    # Δείκτες ως φίλτρα
    ("Φίλτρο MACD: πάνω από τη γραμμή σήματος", {}, {"macd_filter": "above_signal"}, {}, {}),
    ("Φίλτρο MACD: πάνω από το μηδέν", {}, {"macd_filter": "above_zero"}, {}, {}),
    ("Φίλτρο ADX > 20 (δυνατή τάση)", {}, {"adx_min": 20}, {}, {}),
    ("Φίλτρο ADX > 25", {}, {"adx_min": 25}, {}, {}),
    # Νέες είσοδοι
    ("+ Είσοδος MACD (διασταύρωση σε ανοδική τάση)", {}, {"macd_entry": True}, {}, {}),
    ("+ Είσοδος Fibonacci 38–62% (60 ημερών)", {}, {"fib_entry": True}, {}, {}),
    ("Fibonacci αντί για RSI", {"use_pullback": False}, {"fib_entry": True}, {}, {}),
    ("+ Fibonacci και + MACD", {}, {"fib_entry": True, "macd_entry": True}, {}, {}),
    # RSI
    ("RSI 35", {"rsi_level": 35}, {}, {}, {}),
    ("RSI 45", {"rsi_level": 45}, {}, {}, {}),
    ("RSI 50", {"rsi_level": 50}, {}, {}, {}),
    ("RSI 7 ημερών", {"rsi_len": 7}, {}, {}, {}),
    # Σπάσιμο
    ("Σπάσιμο 10 ημερών", {"brk_len": 10}, {}, {}, {}),
    ("Σπάσιμο 55 ημερών", {"brk_len": 55}, {}, {}, {}),
    ("Όγκος ×1,0", {"vol_mult": 1.0}, {}, {}, {}),
    ("Όγκος ×1,5", {"vol_mult": 1.5}, {}, {}, {}),
    ("EMA 20/150", {"ema_fast": 20}, {}, {}, {}),
    # Έξοδοι
    ("Stop 2,5×ATR", {"sl_atr": 2.5}, {}, {}, {}),
    ("Στόχος 4×ATR", {"tp_atr": 4.0}, {}, {}, {}),
    ("Στόχος 5×ATR", {"tp_atr": 5.0}, {}, {}, {}),
    ("Έως 30 μέρες", {"max_bars": 30}, {}, {}, {}),
    ("Κινούμενο stop 3×ATR (με στόχο 3×ATR)", {}, {}, {"trail": 3.0}, {}),
    ("Κινούμενο stop 3×ATR, χωρίς στόχο", {}, {}, {"trail": 3.0, "no_target": True}, {}),
    ("Κινούμενο stop 3×ATR, χωρίς στόχο, έως 40 μέρες", {"max_bars": 40}, {}, {"trail": 3.0, "no_target": True}, {}),
    ("Stop στην τιμή αγοράς μετά από +1R", {}, {}, {"breakeven": 1.0}, {}),
    # Κατάταξη
    ("Κατάταξη: άνοδος 6 μηνών", {}, {}, {"rank": "mom126"}, {}),
    ("Κατάταξη: άνοδος 1 μήνα", {}, {}, {"rank": "mom21"}, {}),
    # Χαρτοφυλάκιο
    ("6 θέσεις × 1%", {}, {}, {}, {"max_positions": 6, "max_total_risk": 0.06}),
    ("8 θέσεις × 0,75%", {}, {}, {}, {"max_positions": 8, "risk_per_trade": 0.0075, "max_total_risk": 0.06}),
    # Συνδυασμός των δύο που πέρασαν το πρώτο τρέξιμο
    ("Σπάσιμο 55 ημερών + 8 θέσεις × 0,75%", {"brk_len": 55}, {}, {},
     {"max_positions": 8, "risk_per_trade": 0.0075, "max_total_risk": 0.06}),
]

# Γειτονικές τιμές: μια πραγματική βελτίωση δεν εξαφανίζεται αν αλλάξει λίγο ο αριθμός
NEIGHBORS = {
    "Φίλτρο ADX > 20 (δυνατή τάση)": [({}, {"adx_min": 18}, {}), ({}, {"adx_min": 22}, {})],
    "Φίλτρο ADX > 25": [({}, {"adx_min": 23}, {}), ({}, {"adx_min": 27}, {})],
    "+ Είσοδος Fibonacci 38–62% (60 ημερών)": [({}, {"fib_entry": True, "fib_len": 40}, {}), ({}, {"fib_entry": True, "fib_len": 80}, {})],
    "Fibonacci αντί για RSI": [({"use_pullback": False}, {"fib_entry": True, "fib_len": 40}, {}),
                               ({"use_pullback": False}, {"fib_entry": True, "fib_len": 80}, {})],
    "RSI 35": [({"rsi_level": 33}, {}, {}), ({"rsi_level": 37}, {}, {})],
    "RSI 45": [({"rsi_level": 43}, {}, {}), ({"rsi_level": 47}, {}, {})],
    "RSI 50": [({"rsi_level": 48}, {}, {}), ({"rsi_level": 52}, {}, {})],
    "RSI 7 ημερών": [({"rsi_len": 6}, {}, {}), ({"rsi_len": 9}, {}, {})],
    "Σπάσιμο 10 ημερών": [({"brk_len": 8}, {}, {}), ({"brk_len": 12}, {}, {})],
    "Σπάσιμο 55 ημερών": [({"brk_len": 45}, {}, {}), ({"brk_len": 65}, {}, {})],
    "Όγκος ×1,0": [({"vol_mult": 0.9}, {}, {}), ({"vol_mult": 1.1}, {}, {})],
    "Όγκος ×1,5": [({"vol_mult": 1.4}, {}, {}), ({"vol_mult": 1.6}, {}, {})],
    "EMA 20/150": [({"ema_fast": 15}, {}, {}), ({"ema_fast": 25}, {}, {})],
    "Stop 2,5×ATR": [({"sl_atr": 2.25}, {}, {}), ({"sl_atr": 2.75}, {}, {})],
    "Στόχος 4×ATR": [({"tp_atr": 3.5}, {}, {}), ({"tp_atr": 4.5}, {}, {})],
    "Στόχος 5×ATR": [({"tp_atr": 4.5}, {}, {}), ({"tp_atr": 5.5}, {}, {})],
    "Έως 30 μέρες": [({"max_bars": 25}, {}, {}), ({"max_bars": 35}, {}, {})],
    "Κινούμενο stop 3×ATR (με στόχο 3×ATR)": [({}, {}, {"trail": 2.5}), ({}, {}, {"trail": 3.5})],
    "Κινούμενο stop 3×ATR, χωρίς στόχο": [({}, {}, {"trail": 2.5, "no_target": True}), ({}, {}, {"trail": 3.5, "no_target": True})],
    "Κινούμενο stop 3×ATR, χωρίς στόχο, έως 40 μέρες": [({"max_bars": 40}, {}, {"trail": 2.5, "no_target": True}),
                                                       ({"max_bars": 40}, {}, {"trail": 3.5, "no_target": True})],
    "Stop στην τιμή αγοράς μετά από +1R": [({}, {}, {"breakeven": 0.8}), ({}, {}, {"breakeven": 1.2})],
    "6 θέσεις × 1%": [({}, {}, {}, {"max_positions": 7, "max_total_risk": 0.07})],
    "8 θέσεις × 0,75%": [({}, {}, {}, {"max_positions": 7, "risk_per_trade": 0.0085, "max_total_risk": 0.06}),
                         ({}, {}, {}, {"max_positions": 10, "risk_per_trade": 0.006, "max_total_risk": 0.06})],
    "Σπάσιμο 55 ημερών + 8 θέσεις × 0,75%": [
        ({"brk_len": 45}, {}, {}, {"max_positions": 8, "risk_per_trade": 0.0075, "max_total_risk": 0.06}),
        ({"brk_len": 65}, {}, {}, {"max_positions": 8, "risk_per_trade": 0.0075, "max_total_risk": 0.06}),
        ({"brk_len": 55}, {}, {}, {"max_positions": 10, "risk_per_trade": 0.006, "max_total_risk": 0.06})],
}

TESTS = [("A 2016–21", "A", "2016-01-01", "2021-12-31"), ("A 2022–σήμερα", "A", "2022-01-01", None),
         ("B 2016–σήμερα", "B", "2016-01-01", None)]


def main() -> None:
    base_p = Params.from_dict(CONFIG["strategy"])
    rk = CONFIG["risk"]
    base_risk = Risk(rk["risk_per_trade"], rk["max_positions"], rk["max_total_risk"], rk["max_notional"], True)
    costs = Costs(**CONFIG["costs"])
    market_sym = CONFIG["market_symbol"]
    uni_a = list(CONFIG["universe"])
    uni_b = [s for s in OTHER if s not in uni_a]
    raw = {}
    for s in sorted(set(uni_a + uni_b + [market_sym])):
        try:
            raw.update(load_all([s]))
        except Exception as e:
            print(f"χωρίς δεδομένα {s}: {e}")
    market = raw[market_sym]
    universes = {"A": {s: raw[s] for s in uni_a if s in raw},
                 "B": {s: raw[s] for s in uni_b if s in raw and raw[s].index[0] <= pd.Timestamp("2015-01-01")}}

    # Έλεγχος: το run_ext χωρίς επιλογές δίνει ακριβώς ό,τι και το backtest του bot
    sig_a = {s: signals_ext(df, base_p, {}) for s, df in universes["A"].items()}
    ref = stats(run(universes["A"], base_p, base_risk, costs, "2022-01-01", None, market))
    mine = stats(run_ext(universes["A"], sig_a, base_p, base_risk, costs, "2022-01-01", None, market))
    same = abs(ref["total"] - mine["total"]) < 1e-9 and ref["trades"] == mine["trades"]

    def evaluate(pchg, ochg, bopt, rchg) -> dict:
        p = replace(base_p, **pchg)
        risk = replace(base_risk, **rchg)
        sg = {u: {s: signals_ext(df, p, ochg) for s, df in universes[u].items()} for u in universes}
        return {label: stats(run_ext(universes[u], sg[u], p, risk, costs, start, end, market, **bopt))
                for label, u, start, end in TESTS}

    rows, summary = [], {}
    for name, pchg, ochg, bopt, rchg in VARIANTS:
        res = evaluate(pchg, ochg, bopt, rchg)
        summary[name] = res
        rows.append((name, res))
        print(name, {k: round(v["sharpe"], 2) for k, v in res.items()})

    base = summary["Σημερινό bot"]

    def verdict(res: dict) -> str:
        d_sh = [res[t]["sharpe"] - base[t]["sharpe"] for t, *_ in TESTS]
        d_dd = [res[t]["max_dd"] - base[t]["max_dd"] for t, *_ in TESTS]      # αρνητικό = χειρότερη πτώση
        if all(x >= 0.05 for x in d_sh) and all(x >= -0.03 for x in d_dd):
            return "✅ καλύτερο και στα 3"
        if all(x <= -0.05 for x in d_sh):
            return "❌ χειρότερο και στα 3"
        return "➖ μικτό (μάλλον τύχη)"

    out = ["# Μεγάλο τεστ παραμέτρων και δεικτών", "",
           f"Δεδομένα έως {market.index[-1]:%d/%m/%Y}. A = οι {len(universes['A'])} μετοχές του bot· "
           f"B = {len(universes['B'])} άλλες μεγάλες μετοχές που το bot δεν έχει δει ποτέ "
           f"({', '.join(universes['B'])}).",
           "Ίδια όρια ρίσκου και κόστη με το bot. Κάθε παραλλαγή αλλάζει μία ρύθμιση. "
           "**Βελτίωση = καλύτερο Sharpe (κέρδος σε σχέση με τις διακυμάνσεις) και στα τρία τεστ**, χωρίς "
           "μεγαλύτερη πτώση. Ό,τι κερδίζει μόνο στο ένα ή στα δύο είναι πιθανότατα τύχη.",
           "", f"Έλεγχος ακρίβειας: η προσομοίωση εδώ δίνει {'ακριβώς τα ίδια' if same else 'ΔΙΑΦΟΡΕΤΙΚΑ'} "
           f"αποτελέσματα με το backtest του bot για τους σημερινούς κανόνες.", "",
           "| Παραλλαγή | " + " | ".join(f"{t}: ανά έτος / πτώση / Sharpe" for t, *_ in TESTS) + " | Συναλλαγές (A 2022–) | Ετυμηγορία |",
           "|---|" + "---|" * (len(TESTS) + 2)]
    for name, res in rows:
        cells = [f"{pct(res[t]['cagr'])} / {pct(res[t]['max_dd'])} / {num(res[t]['sharpe'])}" for t, *_ in TESTS]
        v = "—" if name == "Σημερινό bot" else verdict(res)
        out.append(f"| {name} | " + " | ".join(cells) + f" | {res['A 2022–σήμερα']['trades']} | {v} |")
    winners = [n for n, r in rows if n != "Σημερινό bot" and verdict(r).startswith("✅")]
    robust = []
    if winners:
        out += ["", "## Έλεγχος γειτονικών τιμών για όσες πέρασαν τα τρία τεστ", "",
                "Μια πραγματική βελτίωση δεν εξαφανίζεται αν αλλάξει λίγο ο αριθμός. Κάθε γειτονική τιμή πρέπει να "
                "είναι τουλάχιστον όσο καλή όσο το σημερινό bot (Sharpe) και στα τρία τεστ.", "",
                "| Παραλλαγή | Γειτονική τιμή | " + " | ".join(f"{t}: Sharpe" for t, *_ in TESTS) + " | Κρατά; |",
                "|---|---|" + "---|" * (len(TESTS) + 1)]
        for n in winners:
            nb = NEIGHBORS.get(n)
            if not nb:
                out.append(f"| {n} | (δεν έχει αριθμητική ρύθμιση) | " + " | ".join("–" for _ in TESTS) + " | – |")
                continue
            vname = next(v for v in VARIANTS if v[0] == n)
            all_ok = True
            for item in nb:
                pchg, ochg, bopt = item[:3]
                rchg = item[3] if len(item) > 3 else vname[4]
                res = evaluate(pchg, ochg, bopt, rchg)
                ok = all(res[t]["sharpe"] >= base[t]["sharpe"] for t, *_ in TESTS)
                all_ok &= ok
                desc = ", ".join(f"{k}={v}" for k, v in {**pchg, **ochg, **bopt, **(item[3] if len(item) > 3 else {})}.items())
                out.append(f"| {n} | {desc} | " + " | ".join(num(res[t]["sharpe"]) for t, *_ in TESTS) +
                           f" | {'ναι' if ok else 'όχι'} |")
                summary[f"{n} | γείτονας {desc}"] = res
            if all_ok:
                robust.append(n)
    out += ["", "## Συμπέρασμα", "",
            "Σημείωση: στο ίδιο τεστ με τυχαία (ψεύτικα) δεδομένα, περίπου 2 στις 30 παραλλαγές φαίνονται «καλύτερες "
            "και στα τρία» μόνο από τύχη. Γι' αυτό μετράμε ως πραγματική βελτίωση μόνο ό,τι περνά ΚΑΙ τον έλεγχο "
            "γειτονικών τιμών.", "",
            ("Πραγματικές βελτιώσεις (τρία τεστ + γειτονικές τιμές): " + ", ".join(robust) +
             ". Χρειάζονται την έγκρισή σου πριν μπουν στο bot.") if robust else
            ("Καμία παραλλαγή δεν πέρασε όλους τους ελέγχους" +
             (f" (στα τρία τεστ πέρασαν: {', '.join(winners)}, αλλά όχι με γειτονικές τιμές)" if winners else "") +
             ". Οι σημερινοί κανόνες είναι ήδη από τους καλύτερους που δοκιμάσαμε· οι «βελτιώσεις» που φαίνονται "
             "σε ένα ή δύο τεστ είναι πιθανότατα τύχη."), ""]

    path = ROOT / "reports"
    path.mkdir(exist_ok=True)
    (path / "research_params.md").write_text("\n".join(out), encoding="utf-8")
    (path / "research_params.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float),
                                               encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()

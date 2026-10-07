"""Πιο σφιχτά stop-loss / take-profit / λιγότερες μέρες σε θέση: βοηθούν ή χαλάνε;

Ίδιος αυστηρός έλεγχος: 2016–2021 για «εκπαίδευση», 2022–σήμερα ως «άγνωστα» χρόνια.
Ίδιοι κανόνες εισόδου και ίδιο ρίσκο ανά συναλλαγή (1% του λογαριασμού) σε όλες τις παραλλαγές:
αλλάζει μόνο το πού μπαίνει το stop, ο στόχος και πόσες μέρες κρατάμε μια θέση.
Εκτέλεση:  python -m bot.research_exits   →   reports/research_exits.md
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from .backtest import CONFIG, PERIODS, ROOT, Costs, Risk, num, pct, run, stats
from .strategy import Params

HEAD = ("| Παραλλαγή | Ανά έτος | Μέγ. πτώση | Sharpe | Συναλλαγές | Επιτυχία | Profit factor | "
        "Μέση ζημιά (R) | Χειρότερη (R) |\n|---|---|---|---|---|---|---|---|---|")


def line(name: str, st: dict, trades: list) -> str:
    r = np.array([t["r"] for t in trades]) if trades else np.array([])
    avg_loss = r[r < 0].mean() if (r < 0).any() else 0.0
    worst = r.min() if len(r) else 0.0
    return (f"| {name} | {pct(st['cagr'])} | {pct(st['max_dd'])} | {num(st['sharpe'])} | {st['trades']} | "
            f"{st['win'] * 100:.0f}% | {num(st['pf'])} | {num(avg_loss)} | {num(worst)} |")


def main() -> None:
    from .data import load_all

    uni = list(CONFIG["universe"])
    market_sym = CONFIG["market_symbol"]
    data = load_all(sorted(set(uni + [market_sym])))
    market = data[market_sym]
    stocks = {s: data[s] for s in uni if s in data}
    costs = Costs(**CONFIG["costs"])
    rk = CONFIG["risk"]
    base = Params.from_dict(CONFIG["strategy"])
    risk = Risk(rk["risk_per_trade"], rk["max_positions"], rk["max_total_risk"], rk["max_notional"], True)

    variants = {
        f"A. Τωρινό: stop {base.sl_atr:g}×ATR, στόχος {base.tp_atr:g}×ATR, έως {base.max_bars} μέρες": base,
        "B. Πιο κοντινό stop: 1,5×ATR (στόχος 3×ATR)": replace(base, sl_atr=1.5),
        "C. Κοντινό stop και στόχος: 1,5×ATR / 2,25×ATR": replace(base, sl_atr=1.5, tp_atr=2.25),
        "D. Πολύ σφιχτά: 1×ATR / 2×ATR": replace(base, sl_atr=1.0, tp_atr=2.0),
        "E. Πιο κοντινός στόχος: 2×ATR / 2×ATR": replace(base, tp_atr=2.0),
        "F. Πιο μακρινός στόχος: 2×ATR / 4×ATR": replace(base, tp_atr=4.0),
        "G. Λιγότερες μέρες: έως 10": replace(base, max_bars=10),
        "H. Πολύ λίγες μέρες: έως 5": replace(base, max_bars=5),
        "I. Κοντινό stop + έως 10 μέρες: 1,5×ATR / 3×ATR": replace(base, sl_atr=1.5, max_bars=10),
    }
    out = ["# Έρευνα: πιο σφιχτά stop-loss, στόχοι και διάρκεια θέσης", "",
           f"{len(stocks)} μετοχές, Capital.com 1:1, spread {costs.per_side:.2%} ανά πράξη. Σε όλες τις παραλλαγές "
           "το ρίσκο ανά συναλλαγή μένει 1% του λογαριασμού: πιο κοντινό stop σημαίνει μεγαλύτερη θέση για την ίδια "
           "ζημιά σε ευρώ, όχι μικρότερη ζημιά.", "",
           "R = πόσες φορές το αρχικό ρίσκο (1 R ≈ 1% του λογαριασμού). «Χειρότερη»: η μεγαλύτερη ζημιά μίας "
           "συναλλαγής, π.χ. όταν η τιμή ανοίγει με κενό κάτω από το stop.", ""]
    summary: dict[str, dict] = {}
    for period, (start, end) in PERIODS.items():
        rows = []
        for name, pv in variants.items():
            res = run(stocks, pv, risk, costs, start, end, market)
            rows.append((name, stats(res), res["trades"]))
        out += [f"### {period}", "", HEAD] + [line(n, st, tr) for n, st, tr in rows] + [""]
        summary[period] = {n: st for n, st, _ in rows}
    path = ROOT / "reports"
    path.mkdir(exist_ok=True)
    (path / "research_exits.md").write_text("\n".join(out), encoding="utf-8")
    (path / "research_exits.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float),
                                              encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()

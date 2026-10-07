"""Ο αργός μέσος (φίλτρο τάσης μετοχής και αγοράς): πώς αλλάζουν τα αποτελέσματα από 100 ως 200 ημέρες;

Αλλάζει ΜΙΑ ρύθμιση τη φορά γύρω από τις τωρινές. Αν οι γειτονικές ρυθμίσεις δίνουν παρόμοια αποτελέσματα,
οι κανόνες είναι γεροί· αν μόνο η τωρινή βγάζει καλά, ήταν τύχη. Ίδιο ρίσκο και ίδιες έξοδοι παντού.
Ίδιος αυστηρός έλεγχος: 2016–2021 για «εκπαίδευση», 2022–σήμερα ως «άγνωστα» χρόνια.
Εκτέλεση:  python -m bot.research_ema   →   reports/research_ema.md
"""

from __future__ import annotations

import json
from dataclasses import replace

from .backtest import CONFIG, PERIODS, ROOT, Costs, Risk, num, pct, run, stats
from .strategy import Params

HEAD = ("| Παραλλαγή | Ανά έτος | Μέγ. πτώση | Sharpe | Συναλλαγές | Επιτυχία | Profit factor |\n"
        "|---|---|---|---|---|---|---|")


def line(name: str, st: dict) -> str:
    return (f"| {name} | {pct(st['cagr'])} | {pct(st['max_dd'])} | {num(st['sharpe'])} | {st['trades']} | "
            f"{st['win'] * 100:.0f}% | {num(st['pf'])} |")


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

    def risk(filt: bool = True) -> Risk:
        return Risk(rk["risk_per_trade"], rk["max_positions"], rk["max_total_risk"], rk["max_notional"], filt)

    # (όνομα, ρυθμίσεις, δείκτης φίλτρου, με φίλτρο αγοράς) — ο αργός μέσος ισχύει και για το φίλτρο αγοράς
    variants = [(f"EMA {f}/{s}" + (" (τωρινό)" if (f, s) == (base.ema_fast, base.ema_slow) else ""),
                 replace(base, ema_fast=f, ema_slow=s), market, True)
                for f, s in [(50, 200), (50, 175), (50, 150), (50, 125), (50, 100), (30, 150), (70, 150)]]
    out = ["# Έρευνα: μήκος των κινητών μέσων (σταθερότητα)", "",
           f"{len(stocks)} μετοχές, Capital.com 1:1, spread {costs.per_side:.2%} ανά πράξη, 1% ρίσκο ανά συναλλαγή, "
           "ίδιες έξοδοι (stop 2×ATR, στόχος 3×ATR, έως 20 μέρες). Αλλάζουν μόνο τα μήκη των EMA· ο αργός EMA "
           "ισχύει και για το φίλτρο αγοράς (Nasdaq-100).", "",
           "Γεροί κανόνες = οι γειτονικές ρυθμίσεις βγάζουν παρόμοια. Μια αλλαγή αξίζει μόνο αν είναι καλύτερη "
           "**και στα δύο** διαστήματα (2016–2021 και 2022–σήμερα), όχι μόνο στο ένα.", ""]
    summary: dict[str, dict] = {}
    for period, (start, end) in PERIODS.items():
        rows = []
        for name, pv, mkt, filt in variants:
            rows.append((name, stats(run(stocks, pv, risk(filt), costs, start, end, mkt))))
        out += [f"### {period}", "", HEAD] + [line(n, st) for n, st in rows] + [""]
        summary[period] = {n: st for n, st in rows}
    path = ROOT / "reports"
    path.mkdir(exist_ok=True)
    (path / "research_ema.md").write_text("\n".join(out), encoding="utf-8")
    (path / "research_ema.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float),
                                                encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()

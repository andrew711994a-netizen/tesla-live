"""Είναι γεροί οι κανόνες εισόδου ή ένα τυχερό σημείο; Έλεγχος σταθερότητας.

Αλλάζει ΜΙΑ ρύθμιση τη φορά γύρω από τις τωρινές. Αν οι γειτονικές ρυθμίσεις δίνουν παρόμοια αποτελέσματα,
οι κανόνες είναι γεροί· αν μόνο η τωρινή βγάζει καλά, ήταν τύχη. Ίδιο ρίσκο και ίδιες έξοδοι παντού.
Ίδιος αυστηρός έλεγχος: 2016–2021 για «εκπαίδευση», 2022–σήμερα ως «άγνωστα» χρόνια.
Εκτέλεση:  python -m bot.research_entries   →   reports/research_entries.md
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
    alt_market = "^GSPC"
    data = load_all(sorted(set(uni + [market_sym, alt_market])))
    market, sp500 = data[market_sym], data[alt_market]
    stocks = {s: data[s] for s in uni if s in data}
    costs = Costs(**CONFIG["costs"])
    rk = CONFIG["risk"]
    base = Params.from_dict(CONFIG["strategy"])

    def risk(filt: bool = True) -> Risk:
        return Risk(rk["risk_per_trade"], rk["max_positions"], rk["max_total_risk"], rk["max_notional"], filt)

    # (όνομα, ρυθμίσεις, δείκτης φίλτρου, με φίλτρο αγοράς)
    variants = [
        ("A. Τωρινό: EMA 50/200, RSI 40, σπάσιμο 20 ημ., όγκος ×1,2, φίλτρο Nasdaq-100", base, market, True),
        ("B. Μόνο υποχώρηση (χωρίς σπάσιμο)", replace(base, use_breakout=False), market, True),
        ("C. Μόνο σπάσιμο (χωρίς υποχώρηση)", replace(base, use_pullback=False), market, True),
        ("D. RSI 35", replace(base, rsi_level=35), market, True),
        ("E. RSI 45", replace(base, rsi_level=45), market, True),
        ("F. Σπάσιμο 55 ημερών", replace(base, brk_len=55), market, True),
        ("G. Σπάσιμο χωρίς φίλτρο όγκου", replace(base, vol_mult=0.0), market, True),
        ("H. Πιο γρήγοροι μέσοι: EMA 20/100 (και στο φίλτρο αγοράς)", replace(base, ema_fast=20, ema_slow=100), market, True),
        ("I. EMA 50/150 (και στο φίλτρο αγοράς)", replace(base, ema_slow=150), market, True),
        ("J. Φίλτρο αγοράς S&P 500 αντί για Nasdaq-100", base, sp500, True),
        ("K. Χωρίς φίλτρο αγοράς", base, market, False),
    ]
    out = ["# Έρευνα: σταθερότητα των κανόνων εισόδου", "",
           f"{len(stocks)} μετοχές, Capital.com 1:1, spread {costs.per_side:.2%} ανά πράξη, 1% ρίσκο ανά συναλλαγή, "
           "ίδιες έξοδοι (stop 2×ATR, στόχος 3×ATR, έως 20 μέρες). Αλλάζει μία ρύθμιση εισόδου τη φορά.", "",
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
    (path / "research_entries.md").write_text("\n".join(out), encoding="utf-8")
    (path / "research_entries.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float),
                                                encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()

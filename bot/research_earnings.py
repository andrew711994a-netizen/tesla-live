"""Έρευνα: φίλτρο αποτελεσμάτων (earnings).

Στις 12/10/2026 το bot αγόρασε τη JPM μία μέρα πριν τα αποτελέσματά της· το πρωί της ανακοίνωσης η τιμή
έπεσε προσωρινά και έπιασε το stop. Εδώ ελέγχουμε στα 10 χρόνια αν βοηθά:
- να μην αγοράζει όταν έρχονται αποτελέσματα στις επόμενες λίγες συνεδριάσεις,
- και/ή να κλείνει τη θέση την προηγούμενη συνεδρίαση από τα αποτελέσματα.

Εκτέλεση:  python -m bot.research_earnings
Γράφει το reports/research_earnings.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .backtest import CONFIG, HEAD, PERIODS, Costs, Risk, num, row, run, stats
from .data import earnings_times, load_all
from .strategy import Params, event_sessions

ROOT = Path(__file__).resolve().parent.parent

VARIANTS = [
    ("Σημερινό bot (χωρίς φίλτρο)", 0, False),
    ("Όχι αγορά αν έχει αποτελέσματα την επόμενη συνεδρίαση", 1, False),
    ("Όχι αγορά αν έχει αποτελέσματα στις επόμενες 3", 3, False),
    ("Όχι αγορά αν έχει αποτελέσματα στις επόμενες 5", 5, False),
    ("Όχι αγορά 5 πριν + κλείσιμο την προηγούμενη συνεδρίαση", 5, True),
    ("Όχι αγορά 3 πριν + κλείσιμο την προηγούμενη συνεδρίαση", 3, True),
    ("Ποτέ μέσα στα αποτελέσματα (όχι αγορά 20 πριν)", 20, False),
]


def crossing_stats(trades: list[dict], events: dict[str, list]) -> dict:
    """Συναλλαγές που κράτησαν τη μετοχή πάνω από αποτελέσματα, σε σχέση με τις υπόλοιπες."""
    def crosses(t: dict) -> bool:
        return any(t["entry_date"] < e <= t["exit_date"] for e in events.get(t["symbol"], []))

    out = {}
    for name, sel in (("με αποτελέσματα", [t for t in trades if crosses(t)]),
                      ("χωρίς αποτελέσματα", [t for t in trades if not crosses(t)])):
        r = np.array([t["r"] for t in sel]) if sel else np.array([])
        out[name] = {"trades": len(sel), "win": float((r > 0).mean()) if len(r) else 0.0,
                     "avg_r": float(r.mean()) if len(r) else 0.0,
                     "worst_r": float(r.min()) if len(r) else 0.0,
                     "below_-1.5R": int((r < -1.5).sum())}
    return out


def main() -> None:
    p = Params.from_dict(CONFIG["strategy"])
    symbols = list(CONFIG["universe"])
    market_sym = CONFIG["market_symbol"]
    data = load_all(sorted(set(symbols + [market_sym])))
    market = data[market_sym]
    tradable = {s: data[s] for s in symbols}
    sessions = market.index

    events, missing = {}, []
    for s in symbols:
        try:
            events[s] = event_sessions(earnings_times(s), sessions)
        except Exception as e:
            missing.append(f"{s} ({str(e)[:60]})")
    firsts = sorted(ev[0] for ev in events.values() if ev)

    rk = CONFIG["risk"]
    risk = Risk(rk["risk_per_trade"], rk["max_positions"], rk["max_total_risk"], rk["max_notional"], True)
    costs = Costs(**CONFIG["costs"])

    out = ["# Έρευνα: φίλτρο αποτελεσμάτων (earnings)", "",
           f"Δεδομένα έως {market.index[-1]:%d/%m/%Y}. Ημερομηνίες αποτελεσμάτων από το ημερολόγιο του Yahoo για "
           f"{len(events)}/{len(symbols)} μετοχές" + (f" (λείπουν: {', '.join(missing)})" if missing else "") +
           (f"· οι παλαιότερες από {firsts[0]:%m/%Y}, οι μισές μετοχές από {firsts[len(firsts) // 2]:%m/%Y}." if firsts else "."),
           "", "Ίδια στρατηγική, ίδια όρια ρίσκου (5 θέσεις, 1% ανά συναλλαγή, φίλτρο αγοράς), κόστη Capital.com 1:1.", ""]
    summary = {}
    for period, (start, end) in PERIODS.items():
        out += [f"## {period}", "", HEAD]
        for name, block, exit_before in VARIANTS:
            res = run(tradable, p, risk, costs, start, end, market, events=events, block=block,
                      exit_before=exit_before)
            st = stats(res)
            summary[f"{period} | {name}"] = st
            out.append(row(name, st))
            if block == 0 and not exit_before:
                base_trades = res["trades"]
        out.append("")
        cs = crossing_stats(base_trades, events)
        out += [f"Σημερινό bot, {period}: συναλλαγές που κράτησαν τη μετοχή πάνω από αποτελέσματα σε σχέση με τις άλλες:", "",
                "| | Συναλλαγές | Επιτυχία | Μέσο R | Χειρότερη (R) | Χειρότερες από −1,5 R |", "|---|---|---|---|---|---|"]
        for k, v in cs.items():
            out.append(f"| {k} | {v['trades']} | {v['win'] * 100:.0f}% | {num(v['avg_r'])} | {num(v['worst_r'])} | "
                       f"{v['below_-1.5R']} |")
        out.append("")
        summary[f"{period} | crossing"] = cs

    path = ROOT / "reports"
    path.mkdir(exist_ok=True)
    (path / "research_earnings.md").write_text("\n".join(out), encoding="utf-8")
    (path / "research_earnings.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float),
                                                 encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
